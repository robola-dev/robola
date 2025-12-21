"""
Robola WebSocket Server

Provides a local WebSocket service for communication with the Robola Web Editor.
"""

import json
import logging
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List

import mujoco
import trimesh
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .handlers.model_data import pack_model_data
from .handlers.model_save import save_model_data
from .handlers.simulation_runtime import SimulationRuntime

logger = logging.getLogger(__name__)


class RobolaServer:
    """Robola Local WebSocket Server"""

    COMPILE_TAG = "robola compile"

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        port: int = 9527,
        allowed_origin: str = "*",
        simulation_fps: float = 60.0,
    ):
        """
        Init Robola Server

        Args:
            mjcf_path: Absolute path to the MJCF file (optional; can be loaded via a WebSocket message)
            port: WebSocket server port
            allowed_origin: Allowed CORS origin
        """
        self.mjcf_path: Optional[Path] = None
        self.port = port
        self.allowed_origin = allowed_origin
        if simulation_fps <= 0 or simulation_fps > 60:
            raise ValueError("simulation_fps must be between 1 and 60 Hz")
        self.simulation_fps = simulation_fps

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

        self.work_dir: Optional[Path] = self.mjcf_path.parent if self.mjcf_path else None

        # Simulation runtime controller
        self.simulation = SimulationRuntime(self, interval_hz=self.simulation_fps)

        if self.mjcf_path:
            self._preload_model()

        # FastAPI Application
        self.app = self._create_app()

    def _preload_model(self):
        """pre-load MJCF model from file"""
        try:
            if not self.mjcf_path:
                raise ValueError("MJCF path is not set")
            print(f"[SERVER] Pre-loading MJCF model from: {self.mjcf_path}")
            logger.info(f"Pre-loading MJCF model from: {self.mjcf_path}")
            base_spec = mujoco.MjSpec.from_file(str(self.mjcf_path))
            target_path = self._ensure_compile_comment(base_spec, self.mjcf_path)

            if target_path == self.mjcf_path:
                self.spec = base_spec
            else:
                self.mjcf_path = target_path
                self.work_dir = self.mjcf_path.parent
                self.spec = mujoco.MjSpec.from_file(str(self.mjcf_path))

            self.model = self.spec.compile()
            self.data = mujoco.MjData(self.model)
            if self.simulation:
                self.simulation.invalidate_metadata()
            print(f"[SERVER] Model loaded successfully: {self.spec.modelname}")
            logger.info(f"Model loaded successfully: {self.spec.modelname}")
        except Exception as e:
            print(f"[SERVER] Failed to pre-load MJCF model: {e}")
            logger.error(f"Failed to pre-load MJCF model: {e}")
            raise

    def _ensure_compile_comment(self, spec: mujoco.MjSpec, source_path: Path) -> Path:
        """Ensure exported MJCF carries the compile marker before loading."""
        target_path = source_path.with_name(
            f"{source_path.stem}_robola_compile{source_path.suffix}"
        )

        if target_path.exists():
            try:
                compiled_spec = mujoco.MjSpec.from_file(str(target_path))
            except Exception as exc:  # pragma: no cover - best effort check
                logger.warning(
                    "Unable to load existing compiled MJCF %s: %s",
                    target_path,
                    exc,
                )
            else:
                if self._has_compile_tag(compiled_spec):
                    logger.info(
                        "Reusing existing compiled MJCF with valid tag: %s",
                        target_path,
                    )
                    return target_path
                logger.info(
                    "Existing compiled MJCF missing compile tag, regenerating: %s",
                    target_path,
                )

        if self._has_compile_tag(spec):
            return source_path

        trimmed = (getattr(spec, "comment", "") or "").strip()
        updated_comment = f"{trimmed}\n{self.COMPILE_TAG}" if trimmed else self.COMPILE_TAG
        spec.comment = updated_comment

        logger.info(
            "Comment missing compile tag; exporting updated MJCF to %s",
            target_path,
        )
        print(f"[SERVER] Exporting MJCF with compile tag to: {target_path}")
        spec.compile()
        spec.to_file(str(target_path))
        return target_path

    def _has_compile_tag(self, spec: mujoco.MjSpec) -> bool:
        """Return True if the spec already contains the compile tag comment."""
        comment = (getattr(spec, "comment", "") or "").lower()
        return self.COMPILE_TAG.lower() in comment

    def _create_app(self) -> FastAPI:
        """Create FastAPI Application"""
        app = FastAPI(
            title="Robola Local Server",
            description="Local WebSocket server for Robola MJCF Editor",
            version="0.1.0",
        )

        # Config CORS
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
            finally:
                await self.simulation.handle_disconnect(websocket)

        return app

    def _build_body_name_map(self) -> List[Dict[str, Any]]:
        """Return a list of runtime body id/name pairs for the current model."""
        if not self.model:
            return []
        mapping: List[Dict[str, Any]] = []
        for body_id in range(self.model.nbody):
            try:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            except Exception:  # pragma: no cover - best effort lookup
                name = None
            mapping.append({"id": body_id, "name": name or f"body_{body_id}"})
        return mapping

    async def _send_model_loaded(self, websocket: WebSocket):
        """Send model loaded notification"""
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
        """Handle incoming WebSocket message"""
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
            "start_simulation": self._handle_start_simulation,
            "pause_simulation": self._handle_pause_simulation,
            "stop_simulation": self._handle_stop_simulation,
            "reset_simulation": self._handle_reset_simulation,
            "set_joint_position": self._handle_set_joint_position,
            "set_equality_active": self._handle_set_equality_active,
            "set_actuator_controls": self._handle_set_actuator_controls,
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
        """Load MJCF model from XML file"""
        try:
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
            self.simulation.invalidate_metadata()

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

            await self._handle_get_model_data(websocket, {})

        except Exception as e:
            logger.error(f"Failed to load MJCF model: {e}")
            await websocket.send_text(
                json.dumps({"type": "error", "message": f"Failed to load model: {str(e)}"})
            )

    async def _handle_get_model_data(self, websocket: WebSocket, message: dict):
        """Send packed model data to client"""
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
        """Save modified model data to MJCF file"""
        try:
            model_data = message.get("modelData")
            if model_data is None:
                await websocket.send_text(
                    json.dumps(
                        {"type": "save_model_result", "status": "error", "message": "Missing model data"}
                    )
                )
                return

            save_model_data(self.spec, self.model, model_data, str(self.mjcf_path))

            self.spec = mujoco.MjSpec.from_file(str(self.mjcf_path))
            self.model = self.spec.compile()
            self.data = mujoco.MjData(self.model)
            self.simulation.invalidate_metadata()

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "save_model_result",
                        "status": "ok",
                        "body_name_map": self._build_body_name_map(),
                    },
                    separators=(",", ":"),
                )
            )
            logger.info(f"Model saved successfully: {self.mjcf_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            await websocket.send_text(
                json.dumps(
                    {"type": "save_model_result", "status": "error", "message": str(e)}
                )
            )

    async def _handle_mesh_request(self, websocket: WebSocket, message: dict):
        """Process mesh file request"""
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
                mesh_id = self.model.mesh(mesh_name).id
                vertadr = self.model.mesh_vertadr[mesh_id]
                vertnum = self.model.mesh_vertnum[mesh_id]
                faceadr = self.model.mesh_faceadr[mesh_id]
                facenum = self.model.mesh_facenum[mesh_id]

                verts = self.model.mesh_vert[vertadr : vertadr + vertnum].reshape(-1, 3)
                faces = self.model.mesh_face[faceadr : faceadr + facenum].reshape(-1, 3)

                verts[:, [0, 1, 2]] = verts[:, [2, 1, 0]]
                faces[:, [0, 1, 2]] = faces[:, [2, 1, 0]]

                tmesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                stl_bytes = tmesh.export(file_type="stl")
                file_data = stl_bytes if isinstance(stl_bytes, bytes) else stl_bytes.encode("utf-8")
                file_size = len(file_data)
                base64_data = base64.b64encode(file_data).decode("utf-8")
            else:
                # Load mesh from file
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
        """ texture request """
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
        """mesh file upload"""
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

            mesh_dir = self.work_dir
            if self.spec and self.spec.meshdir:
                mesh_dir = self.work_dir / self.spec.meshdir

            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = mesh_dir / file_name

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
        """texture file upload"""
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

            texture_dir = self.work_dir
            if self.spec and self.spec.texturedir:
                texture_dir = self.work_dir / self.spec.texturedir

            texture_dir.mkdir(parents=True, exist_ok=True)
            texture_path = texture_dir / file_name

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
        await self.simulation.step(websocket)

    async def _handle_start_simulation(self, websocket: WebSocket, message: dict):
        await self.simulation.start(websocket)

    async def _handle_pause_simulation(self, websocket: WebSocket, message: dict):
        await self.simulation.pause(websocket)

    async def _handle_stop_simulation(self, websocket: WebSocket, message: dict):
        await self.simulation.stop(websocket, reset=True)

    async def _handle_reset_simulation(self, websocket: WebSocket, message: dict):
        await self.simulation.reset(websocket)

    async def _handle_set_joint_position(self, websocket: WebSocket, message: dict):
        await self.simulation.set_joint_position(
            websocket,
            message.get("joint_id"),
            message.get("value"),
        )

    async def _handle_set_equality_active(self, websocket: WebSocket, message: dict):
        await self.simulation.set_equality_active(
            websocket,
            message.get("equality_id"),
            message.get("active"),
        )

    async def _handle_set_actuator_controls(self, websocket: WebSocket, message: dict):
        await self.simulation.set_actuator_controls(websocket, message.get("controls"))

    def run(self):
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
        print(f"  Editor:    https://www.robolaweb.com/editor")
        print(f"{'='*60}")
        print(f"\n  Open Robola Editor: https://www.robolaweb.com/editor in your browser and connect!")
        print(f"  Press Ctrl+C to stop the server.\n")

        uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="info")


def serve(
    mjcf_path: Optional[str] = None,
    port: int = 9527,
    allowed_origin: str = "*",
    fps: float = 60.0,
):
    """
    Start the Robola local server

    Args:
        mjcf_path: Path to the MJCF file (optional)
        port: WebSocket server port
        allowed_origin: Allowed CORS origin
    """
    if fps <= 0 or fps > 60:
        raise ValueError("fps must be between 1 and 60 Hz")

    server = RobolaServer(
        mjcf_path=mjcf_path,
        port=port,
        allowed_origin=allowed_origin,
        simulation_fps=fps,
    )
    server.run()
