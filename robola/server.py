"""
Robola WebSocket Server

提供本地 WebSocket 服务，用于与 Robola Web Editor 通信。
"""

import json
import asyncio
import logging
import base64
from pathlib import Path
from typing import Optional, Dict, Any

import mujoco
import numpy as np
import trimesh
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .handlers.model_data import pack_model_data
from .handlers.model_save import save_model_data

logger = logging.getLogger(__name__)


class RobolaServer:
    """Robola 本地 WebSocket 服务器"""

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        port: int = 9527,
        allowed_origin: str = "*",
    ):
        """
        初始化 Robola 服务器

        Args:
            mjcf_path: MJCF 文件的绝对路径 (可选，可以通过 WebSocket 消息加载)
            port: WebSocket 服务端口
            allowed_origin: 允许的 CORS 来源
        """
        self.mjcf_path: Optional[Path] = None
        self.port = port
        self.allowed_origin = allowed_origin

        if mjcf_path:
            self.mjcf_path = Path(mjcf_path).resolve()
            if not self.mjcf_path.exists():
                raise FileNotFoundError(f"MJCF file not found: {self.mjcf_path}")
            if not self.mjcf_path.suffix.lower() == ".xml":
                raise ValueError(f"File must be an XML file: {self.mjcf_path}")

        # MuJoCo 模型实例
        self.spec: Optional[mujoco.MjSpec] = None
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None

        # 工作目录（MJCF 文件所在目录）
        self.work_dir: Optional[Path] = self.mjcf_path.parent if self.mjcf_path else None

        # 如果指定了 MJCF 文件，预加载模型
        if self.mjcf_path:
            self._preload_model()

        # FastAPI 应用
        self.app = self._create_app()

    def _preload_model(self):
        """预加载 MJCF 模型"""
        try:
            print(f"[SERVER] Pre-loading MJCF model from: {self.mjcf_path}")
            logger.info(f"Pre-loading MJCF model from: {self.mjcf_path}")
            self.spec = mujoco.MjSpec.from_file(str(self.mjcf_path))
            self.model = self.spec.compile()
            self.data = mujoco.MjData(self.model)
            print(f"[SERVER] Model loaded successfully: {self.spec.modelname}")
            logger.info(f"Model loaded successfully: {self.spec.modelname}")
        except Exception as e:
            print(f"[SERVER] Failed to pre-load MJCF model: {e}")
            logger.error(f"Failed to pre-load MJCF model: {e}")
            raise

    def _create_app(self) -> FastAPI:
        """创建 FastAPI 应用"""
        app = FastAPI(
            title="Robola Local Server",
            description="Local WebSocket server for Robola MJCF Editor",
            version="0.1.0",
        )

        # 配置 CORS
        origins = [self.allowed_origin] if self.allowed_origin != "*" else ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/")
        async def root():
            return {
                "status": "running",
                "mjcf_file": str(self.mjcf_path) if self.mjcf_path else None,
                "work_dir": str(self.work_dir) if self.work_dir else None,
            }

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            logger.info("WebSocket client connected")
            print("[SERVER] WebSocket client connected")

            try:
                # 如果已经加载了模型，自动发送模型数据
                if self.spec is not None and self.model is not None:
                    print(f"[SERVER] Auto-sending model data, model: {self.spec.modelname}")
                    logger.info("Auto-sending model data to new client")
                    await self._send_model_loaded(websocket)
                    await self._handle_get_model_data(websocket, {})
                    print("[SERVER] Model data sent successfully")
                else:
                    print(f"[SERVER] No model loaded yet, spec={self.spec}, model={self.model}")
                
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self._handle_message(websocket, message)

            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                try:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )
                except:
                    pass

        return app

    async def _send_model_loaded(self, websocket: WebSocket):
        """发送模型加载完成消息"""
        await websocket.send_text(
            json.dumps(
                {
                    "type": "model_loaded",
                    "message": f"Successfully loaded {self.mjcf_path.name if self.mjcf_path else 'model'}",
                    "model_info": {
                        "name": self.spec.modelname if self.spec else "",
                        "file": str(self.mjcf_path) if self.mjcf_path else "",
                    },
                }
            )
        )

    async def _handle_message(self, websocket: WebSocket, message: dict):
        """处理 WebSocket 消息"""
        message_type = message.get("type")

        handlers = {
            "load_xml": self._handle_load_xml,
            "get_model_data": self._handle_get_model_data,
            "save_model": self._handle_save_model,
            "request_mesh": self._handle_mesh_request,
            "request_texture": self._handle_texture_request,
            "upload_mesh": self._handle_mesh_upload,
            "upload_texture": self._handle_texture_upload,
            "step_simulation": self._handle_step_simulation,
        }

        handler = handlers.get(message_type)
        if handler:
            await handler(websocket, message)
        else:
            await websocket.send_text(
                json.dumps(
                    {"type": "error", "message": f"Unknown message type: {message_type}"}
                )
            )

    async def _handle_load_xml(self, websocket: WebSocket, message: dict):
        """加载 MJCF 模型"""
        try:
            # 支持从消息中获取 xml_file 路径，或使用初始化时指定的路径
            xml_file = message.get("xml_file")
            if xml_file:
                self.mjcf_path = Path(xml_file).resolve()
                self.work_dir = self.mjcf_path.parent
            
            if not self.mjcf_path:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "No MJCF file specified"})
                )
                return
            
            if not self.mjcf_path.exists():
                await websocket.send_text(
                    json.dumps({"type": "error", "message": f"File not found: {self.mjcf_path}"})
                )
                return
            
            logger.info(f"Loading MJCF model from: {self.mjcf_path}")

            self.spec = mujoco.MjSpec.from_file(str(self.mjcf_path))
            self.model = self.spec.compile()
            self.data = mujoco.MjData(self.model)

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "model_loaded",
                        "message": f"Successfully loaded {self.mjcf_path.name}",
                        "model_info": {
                            "name": self.spec.modelname,
                            "file": str(self.mjcf_path),
                        },
                    }
                )
            )

            # 立即发送模型数据
            await self._handle_get_model_data(websocket, {})

        except Exception as e:
            logger.error(f"Failed to load MJCF model: {e}")
            await websocket.send_text(
                json.dumps({"type": "error", "message": f"Failed to load model: {str(e)}"})
            )

    async def _handle_get_model_data(self, websocket: WebSocket, message: dict):
        """发送模型数据"""
        if self.spec is None or self.model is None or self.data is None:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "No model loaded"})
            )
            return

        try:
            payload = pack_model_data(self.spec, self.model, self.data)
            await websocket.send_text(
                json.dumps({"type": "model_data", "payload": payload}, separators=(",", ":"))
            )
        except Exception as e:
            logger.error(f"Failed to send model data: {e}")
            await websocket.send_text(
                json.dumps({"type": "error", "message": f"Failed to send model data: {str(e)}"})
            )

    async def _handle_save_model(self, websocket: WebSocket, message: dict):
        """保存模型"""
        try:
            model_data = message.get("modelData")
            if model_data is None:
                await websocket.send_text(
                    json.dumps(
                        {"type": "save_model_result", "status": "error", "message": "Missing model data"}
                    )
                )
                return

            logger.info(f"Saving model to: {self.mjcf_path}")

            save_model_data(self.spec, self.model, model_data, str(self.mjcf_path))

            # 重新加载模型以获取最新状态
            self.spec = mujoco.MjSpec.from_file(str(self.mjcf_path))
            self.model = self.spec.compile()
            self.data = mujoco.MjData(self.model)

            await websocket.send_text(
                json.dumps({"type": "save_model_result", "status": "ok"})
            )

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            await websocket.send_text(
                json.dumps(
                    {"type": "save_model_result", "status": "error", "message": str(e)}
                )
            )

    async def _handle_mesh_request(self, websocket: WebSocket, message: dict):
        """处理 mesh 文件请求"""
        try:
            mesh_name = message.get("meshName")
            request_id = message.get("requestId")

            if not mesh_name or not request_id:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "mesh_data",
                            "requestId": request_id,
                            "meshData": None,
                            "error": "Missing meshName or requestId",
                        }
                    )
                )
                return

            mesh_spec = self.spec.mesh(mesh_name)

            if mesh_spec.file == "":
                # 从模型数据生成 mesh
                mesh_id = self.model.mesh(mesh_name).id
                vertadr = self.model.mesh_vertadr[mesh_id]
                vertnum = self.model.mesh_vertnum[mesh_id]
                faceadr = self.model.mesh_faceadr[mesh_id]
                facenum = self.model.mesh_facenum[mesh_id]

                verts = self.model.mesh_vert[vertadr : vertadr + vertnum].reshape(-1, 3)
                faces = self.model.mesh_face[faceadr : faceadr + facenum].reshape(-1, 3)

                # 坐标转换
                verts[:, [0, 1, 2]] = verts[:, [2, 1, 0]]
                faces[:, [0, 1, 2]] = faces[:, [2, 1, 0]]

                tmesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                stl_bytes = tmesh.export(file_type="stl")
                file_data = stl_bytes if isinstance(stl_bytes, bytes) else stl_bytes.encode("utf-8")
                file_size = len(file_data)
                base64_data = base64.b64encode(file_data).decode("utf-8")
            else:
                # 从文件读取 mesh
                mesh_dir = self.spec.meshdir if self.spec.meshdir else ""
                mesh_path = self.work_dir / mesh_dir / mesh_spec.file

                if not mesh_path.exists():
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "mesh_data",
                                "requestId": request_id,
                                "meshData": None,
                                "error": f"Mesh file not found: {mesh_name}",
                            }
                        )
                    )
                    return

                file_size = mesh_path.stat().st_size
                if file_size > 100 * 1024 * 1024:  # 100MB limit
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "mesh_data",
                                "requestId": request_id,
                                "meshData": None,
                                "error": f"Mesh file too large: {mesh_name}",
                            }
                        )
                    )
                    return

                with open(mesh_path, "rb") as f:
                    file_data = f.read()
                    base64_data = base64.b64encode(file_data).decode("utf-8")

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "mesh_data",
                        "requestId": request_id,
                        "meshName": mesh_name,
                        "meshData": base64_data,
                        "fileSize": file_size,
                    }
                )
            )
            logger.debug(f"Sent mesh data: {mesh_name} ({file_size} bytes)")

        except Exception as e:
            logger.error(f"Failed to handle mesh request: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "mesh_data",
                        "requestId": message.get("requestId"),
                        "meshData": None,
                        "error": f"Failed to load mesh: {str(e)}",
                    }
                )
            )

    async def _handle_texture_request(self, websocket: WebSocket, message: dict):
        """处理 texture 文件请求"""
        try:
            texture_name = message.get("textureName")
            request_id = message.get("requestId")

            if not texture_name or not request_id:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "texture_data",
                            "requestId": request_id,
                            "textureData": None,
                            "error": "Missing textureName or requestId",
                        }
                    )
                )
                return

            texture_file = ""
            try:
                tex = self.spec.texture(texture_name)
                if tex and tex.file:
                    texture_file = tex.file
            except:
                texture_file = texture_name

            if not texture_file:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "texture_data",
                            "requestId": request_id,
                            "textureData": None,
                            "error": f"Texture file not found for: {texture_name}",
                        }
                    )
                )
                return

            texture_dir = self.spec.texturedir if self.spec.texturedir else ""
            texture_path = self.work_dir / texture_dir / texture_file

            if not texture_path.exists():
                # 尝试直接在工作目录下查找
                texture_path = self.work_dir / texture_file
                if not texture_path.exists():
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "texture_data",
                                "requestId": request_id,
                                "textureData": None,
                                "error": f"Texture file not found: {texture_name}",
                            }
                        )
                    )
                    return

            file_size = texture_path.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "texture_data",
                            "requestId": request_id,
                            "textureData": None,
                            "error": f"Texture file too large: {texture_name}",
                        }
                    )
                )
                return

            ext = texture_path.suffix.lower()
            mime_type = "image/png"
            if ext in (".jpg", ".jpeg"):
                mime_type = "image/jpeg"
            elif ext == ".bmp":
                mime_type = "image/bmp"

            with open(texture_path, "rb") as f:
                file_data = f.read()
                base64_data = base64.b64encode(file_data).decode("utf-8")
                data_url = f"data:{mime_type};base64,{base64_data}"

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "texture_data",
                        "requestId": request_id,
                        "textureName": texture_name,
                        "textureData": data_url,
                        "fileSize": file_size,
                    }
                )
            )
            logger.debug(f"Sent texture data: {texture_name} ({file_size} bytes)")

        except Exception as e:
            logger.error(f"Failed to handle texture request: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "texture_data",
                        "requestId": message.get("requestId"),
                        "textureData": None,
                        "error": f"Failed to load texture: {str(e)}",
                    }
                )
            )

    async def _handle_mesh_upload(self, websocket: WebSocket, message: dict):
        """处理 mesh 文件上传"""
        try:
            mesh_name = message.get("meshName")
            file_name = message.get("fileName")
            mesh_data_base64 = message.get("meshData")
            request_id = message.get("requestId")

            if not all([mesh_name, file_name, mesh_data_base64, request_id]):
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "requestId": request_id,
                            "message": "Missing required fields for mesh upload",
                        }
                    )
                )
                return

            # 确定保存路径
            mesh_dir = self.work_dir
            if self.spec and self.spec.meshdir:
                mesh_dir = self.work_dir / self.spec.meshdir

            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = mesh_dir / file_name

            # 解码并保存
            mesh_data = base64.b64decode(mesh_data_base64)
            with open(mesh_path, "wb") as f:
                f.write(mesh_data)

            logger.info(f"Mesh file saved: {mesh_path} ({len(mesh_data)} bytes)")

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "mesh_uploaded",
                        "requestId": request_id,
                        "meshName": mesh_name,
                        "fileName": file_name,
                        "fileSize": len(mesh_data),
                        "savedPath": str(mesh_path),
                    }
                )
            )

        except Exception as e:
            logger.error(f"Failed to handle mesh upload: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "requestId": message.get("requestId"),
                        "message": f"Failed to upload mesh: {str(e)}",
                    }
                )
            )

    async def _handle_texture_upload(self, websocket: WebSocket, message: dict):
        """处理 texture 文件上传"""
        try:
            texture_name = message.get("textureName")
            file_name = message.get("fileName")
            texture_data_base64 = message.get("textureData")
            request_id = message.get("requestId")

            if not all([texture_name, file_name, texture_data_base64, request_id]):
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "requestId": request_id,
                            "message": "Missing required fields for texture upload",
                        }
                    )
                )
                return

            # 确定保存路径
            texture_dir = self.work_dir
            if self.spec and self.spec.texturedir:
                texture_dir = self.work_dir / self.spec.texturedir

            texture_dir.mkdir(parents=True, exist_ok=True)
            texture_path = texture_dir / file_name

            # 解码并保存
            texture_data = base64.b64decode(texture_data_base64)
            with open(texture_path, "wb") as f:
                f.write(texture_data)

            logger.info(f"Texture file saved: {texture_path} ({len(texture_data)} bytes)")

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "texture_uploaded",
                        "requestId": request_id,
                        "textureName": texture_name,
                        "fileName": file_name,
                        "fileSize": len(texture_data),
                        "savedPath": str(texture_path),
                    }
                )
            )

        except Exception as e:
            logger.error(f"Failed to handle texture upload: {e}")
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "requestId": message.get("requestId"),
                        "message": f"Failed to upload texture: {str(e)}",
                    }
                )
            )

    async def _handle_step_simulation(self, websocket: WebSocket, message: dict):
        """执行仿真步进"""
        if self.model is None or self.data is None:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "No model loaded"})
            )
            return

        try:
            mujoco.mj_step(self.model, self.data)
            await self._handle_get_model_data(websocket, {})
        except Exception as e:
            logger.error(f"Simulation step failed: {e}")
            await websocket.send_text(
                json.dumps({"type": "error", "message": f"Simulation step failed: {str(e)}"})
            )

    def run(self):
        """启动服务器"""
        print(f"\n{'='*60}")
        print(f"  Robola Local Server")
        print(f"{'='*60}")
        if self.mjcf_path:
            print(f"  MJCF File: {self.mjcf_path}")
            print(f"  Work Dir:  {self.work_dir}")
        else:
            print(f"  MJCF File: (none - waiting for file selection)")
        print(f"  Port:      {self.port}")
        print(f"{'='*60}")
        print(f"  WebSocket: ws://localhost:{self.port}/ws")
        print(f"  Health:    http://localhost:{self.port}/health")
        print(f"{'='*60}")
        print(f"\n  Open Robola Editor in your browser and connect!")
        print(f"  Press Ctrl+C to stop the server.\n")

        uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="info")


def serve(
    mjcf_path: Optional[str] = None,
    port: int = 9527,
    allowed_origin: str = "*",
):
    """
    启动 Robola 本地服务

    Args:
        mjcf_path: MJCF 文件路径 (可选)
        port: WebSocket 服务端口
        allowed_origin: 允许的 CORS 来源
    """
    server = RobolaServer(
        mjcf_path=mjcf_path,
        port=port,
        allowed_origin=allowed_origin,
    )
    server.run()
