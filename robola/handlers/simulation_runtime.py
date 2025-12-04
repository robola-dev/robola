"""Simulation runtime handling utilities for Robola."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import mujoco
import numpy as np
from fastapi import WebSocket

from .model_data import mujoco2WebPos, mujoco2WebQuat

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from ..server import RobolaServer

logger = logging.getLogger(__name__)


def _quat_inverse(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    a, b, c, d = np.asarray(q1, dtype=float)
    e, f, g, h = np.asarray(q2, dtype=float)
    return np.array(
        [
            a * e - b * f - c * g - d * h,
            a * f + b * e + c * h - d * g,
            a * g - b * h + c * e + d * f,
            a * h + b * g - c * f + d * e,
        ],
        dtype=float,
    )


def _quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    qvec = np.array([0.0, *np.asarray(vec, dtype=float)], dtype=float)
    qinv = _quat_inverse(quat)
    return _quat_multiply(_quat_multiply(quat, qvec), qinv)[1:]


def _compute_body_relative_pose(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    parent_id = int(model.body_parentid[body_id])
    if parent_id == -1:
        return np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    child_pos = np.asarray(data.xpos[body_id], dtype=float)
    parent_pos = np.asarray(data.xpos[parent_id], dtype=float)
    child_quat = np.asarray(data.xquat[body_id], dtype=float)
    parent_quat = np.asarray(data.xquat[parent_id], dtype=float)

    parent_inv = _quat_inverse(parent_quat)
    rel_quat = _quat_multiply(parent_inv, child_quat)
    rel_pos = _quat_rotate(parent_inv, child_pos - parent_pos)
    return rel_pos, rel_quat


class SimulationRuntime:
    """Encapsulates MuJoCo simulation lifecycle management."""

    def __init__(self, server: "RobolaServer", interval_hz: float = 30.0) -> None:
        self._server = server
        self._interval = 1.0 / max(interval_hz, 1.0)
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._state: str = "idle"
        self._client: Optional[WebSocket] = None
        self._control_payload: Optional[Dict[str, Any]] = None
        self._actuator_ranges: Dict[int, Tuple[float, float]] = {}

    @property
    def state(self) -> str:
        return self._state

    def invalidate_metadata(self) -> None:
        self._control_payload = None
        self._actuator_ranges.clear()

    async def start(self, websocket: WebSocket) -> None:
        if not self._server.model or not self._server.data:
            await self._send_error(websocket, "No model loaded")
            return

        if self._task:
            await self._stop(target_state=self._state, notify=False)

        self._client = websocket
        await self._send_control_metadata(websocket)
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._simulation_loop(websocket, self._stop_event))
        self._state = "running"
        await self._notify_status("running", primary=websocket)

    async def pause(self, websocket: WebSocket) -> None:
        if self._state != "running" or not self._task:
            await self._notify_status(self._state or "idle", primary=websocket)
            return

        await self._stop(target_state="paused", primary=websocket)

    async def stop(self, websocket: WebSocket, *, reset: bool = True) -> None:
        await self._stop(target_state="idle", reset_model=reset, primary=websocket)

    async def reset(self, websocket: WebSocket) -> None:
        if not self._server.model or not self._server.data:
            await self._send_error(websocket, "No model loaded")
            return

        mujoco.mj_resetData(self._server.model, self._server.data)
        mujoco.mj_forward(self._server.model, self._server.data)

        frame = self._build_simulation_frame()
        payload = json.dumps({"type": "simulation_reset", "frame": frame}, separators=(",", ":"))

        targets = [ws for ws in (websocket, self._client) if ws is not None]
        sent: set[WebSocket] = set()
        for ws in targets:
            if ws in sent:
                continue
            try:
                await ws.send_text(payload)
            except Exception as exc:
                logger.debug("Failed sending reset notification: %s", exc)
            sent.add(ws)

    async def step(self, websocket: WebSocket) -> None:
        if not self._server.model or not self._server.data:
            await self._send_error(websocket, "No model loaded")
            return

        try:
            mujoco.mj_step(self._server.model, self._server.data)
            await self._server._handle_get_model_data(websocket, {})
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Simulation step failed: %s", exc)
            await self._send_error(websocket, f"Simulation step failed: {exc}")

    async def set_joint_position(self, websocket: WebSocket, joint_id: Optional[int], value: Optional[float]) -> None:
        if not self._server.model or not self._server.data:
            await self._send_error(websocket, "No model loaded")
            return

        if self._state == "running":
            await self._send_error(websocket, "Pause simulation before editing joints")
            return

        if joint_id is None or value is None:
            await self._send_error(websocket, "joint_id and value are required")
            return

        try:
            joint_index = int(joint_id)
            target_value = float(value)
        except (TypeError, ValueError):
            await self._send_error(websocket, "Invalid joint_id or value")
            return

        if joint_index < 0 or joint_index >= self._server.model.njnt:
            await self._send_error(websocket, f"Joint id {joint_index} out of range")
            return

        joint_type = int(self._server.model.jnt_type[joint_index])
        supported_types = (
            mujoco.mjtJoint.mjJNT_HINGE.value,
            mujoco.mjtJoint.mjJNT_SLIDE.value,
        )
        if joint_type not in supported_types:
            await self._send_error(
                websocket,
                "Only hinge and slide joints can be positioned from the editor",
            )
            return

        qpos_adr = int(self._server.model.jnt_qposadr[joint_index])
        self._server.data.qpos[qpos_adr] = target_value
        mujoco.mj_forward(self._server.model, self._server.data)

        await websocket.send_text(
            json.dumps(
                {
                    "type": "joint_position_updated",
                    "joint_id": joint_index,
                    "value": target_value,
                },
                separators=(",", ":"),
            )
        )

    async def set_equality_active(self, websocket: WebSocket, equality_id: Optional[int], active: Optional[bool]) -> None:
        if not self._server.model or not self._server.data:
            await self._send_error(websocket, "No model loaded")
            return

        if self._state == "running":
            await self._send_error(websocket, "Pause simulation before toggling equality constraints")
            return

        if equality_id is None or active is None:
            await self._send_error(websocket, "equality_id and active are required")
            return

        try:
            eq_index = int(equality_id)
            active_flag = 1 if bool(active) else 0
        except (TypeError, ValueError):
            await self._send_error(websocket, "Invalid equality_id or active flag")
            return

        if eq_index < 0 or eq_index >= self._server.model.neq:
            await self._send_error(websocket, f"Equality id {eq_index} out of range")
            return

        self._server.data.eq_active[eq_index] = active_flag
        mujoco.mj_forward(self._server.model, self._server.data)

        await websocket.send_text(
            json.dumps(
                {
                    "type": "equality_state_updated",
                    "equality_id": eq_index,
                    "active": bool(active_flag),
                },
                separators=(",", ":"),
            )
        )

    async def set_actuator_controls(self, websocket: WebSocket, controls: Optional[List[Dict[str, Any]]]) -> None:
        if not self._server.model or not self._server.data:
            await self._send_error(websocket, "No model loaded")
            return

        if self._state != "running":
            await self._send_error(websocket, "Simulation must be running to drive actuators")
            return

        if not isinstance(controls, list) or not controls:
            await self._send_error(websocket, "controls payload is required")
            return

        applied: List[Dict[str, float]] = []
        for entry in controls:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("id")
            value = entry.get("value")
            try:
                actuator_index = int(idx)
                target_value = float(value)
            except (TypeError, ValueError):
                continue

            if actuator_index < 0 or actuator_index >= self._server.model.nu:
                continue

            low, high = self._lookup_actuator_range(actuator_index)
            clamped = min(max(target_value, low), high)
            self._server.data.ctrl[actuator_index] = clamped
            applied.append({"id": actuator_index, "value": clamped})

        if not applied:
            await self._send_error(websocket, "No valid actuator controls provided")
            return

        await websocket.send_text(
            json.dumps(
                {"type": "actuator_controls_applied", "controls": applied},
                separators=(",", ":"),
            )
        )

    async def handle_disconnect(self, websocket: WebSocket) -> None:
        if self._client == websocket:
            await self._stop(target_state="idle", notify=False)

    async def _send_control_metadata(self, websocket: WebSocket) -> None:
        payload = self._get_control_payload()
        if not payload:
            return
        try:
            await websocket.send_text(json.dumps(payload, separators=(",", ":")))
        except Exception as exc:  # pragma: no cover - best effort notification
            logger.debug("Failed sending control metadata: %s", exc)

    def _get_control_payload(self) -> Optional[Dict[str, Any]]:
        if not self._server.model or not self._server.data:
            return None
        if self._control_payload is None:
            control_body = self._build_control_payload()
            self._control_payload = {"type": "simulation_controls", "payload": control_body}
        return self._control_payload

    def _build_control_payload(self) -> Dict[str, Any]:
        model = self._server.model
        assert model is not None

        joints: List[Dict[str, Any]] = []
        joint_types = (
            mujoco.mjtJoint.mjJNT_HINGE.value,
            mujoco.mjtJoint.mjJNT_SLIDE.value,
        )
        for joint_id in range(model.njnt):
            joint_type = int(model.jnt_type[joint_id])
            if joint_type not in joint_types:
                continue

            if not int(model.jnt_limited[joint_id]):
                continue

            range_min, range_max = self._read_range(model.jnt_range, joint_id)
            if math.isclose(range_min, range_max, rel_tol=0.0, abs_tol=1e-9):
                continue

            joints.append(
                {
                    "id": joint_id,
                    "name": self._get_obj_name(mujoco.mjtObj.mjOBJ_JOINT, joint_id, "joint"),
                    "range": [range_min, range_max],
                    "type": joint_type,
                    "qpos_index": int(model.jnt_qposadr[joint_id]),
                }
            )

        actuators: List[Dict[str, Any]] = []
        self._actuator_ranges.clear()
        for actuator_id in range(model.nu):
            limited = bool(int(model.actuator_ctrllimited[actuator_id]))
            if limited:
                ctrl_min, ctrl_max = self._read_range(model.actuator_ctrlrange, actuator_id)
            else:
                ctrl_min, ctrl_max = -1.0, 1.0

            if math.isclose(ctrl_min, ctrl_max, rel_tol=0.0, abs_tol=1e-9):
                ctrl_max = ctrl_min

            self._actuator_ranges[actuator_id] = (ctrl_min, ctrl_max)
            actuators.append(
                {
                    "id": actuator_id,
                    "name": self._get_obj_name(mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id, "actuator"),
                    "range": [ctrl_min, ctrl_max],
                    "limited": limited,
                }
            )

        equalities: List[Dict[str, Any]] = []
        for eq_id in range(model.neq):
            equalities.append(
                {
                    "id": eq_id,
                    "name": self._get_obj_name(mujoco.mjtObj.mjOBJ_EQUALITY, eq_id, "equality"),
                    "type": int(model.eq_type[eq_id]),
                }
            )

        return {"joints": joints, "actuators": actuators, "equalities": equalities}

    def _get_obj_name(self, obj_type: mujoco.mjtObj, obj_id: int, fallback_prefix: str) -> str:
        model = self._server.model
        if model is None:
            return f"{fallback_prefix}_{obj_id}"
        try:
            name = mujoco.mj_id2name(model, obj_type, obj_id)
        except Exception:  # pragma: no cover - best effort lookup
            name = None
        return name or f"{fallback_prefix}_{obj_id}"

    def _lookup_actuator_range(self, actuator_index: int) -> Tuple[float, float]:
        if actuator_index not in self._actuator_ranges:
            self._get_control_payload()
        return self._actuator_ranges.get(actuator_index, (-1.0, 1.0))

    def _read_range(self, vector: Any, index: int) -> Tuple[float, float]:
        arr = np.asarray(vector)
        if arr.ndim == 1:
            start = 2 * index
            return float(arr[start]), float(arr[start + 1])
        pair = arr[index]
        return float(pair[0]), float(pair[1])

    async def _notify_status(self, status: str, *, primary: Optional[WebSocket] = None) -> None:
        payload = json.dumps({"type": "simulation_status", "status": status}, separators=(",", ":"))
        targets = []
        if primary is not None:
            targets.append(primary)
        if self._client and self._client not in targets:
            targets.append(self._client)

        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception as exc:  # pragma: no cover - best effort notification
                logger.debug("Failed sending simulation status: %s", exc)

    async def _stop(
        self,
        target_state: str = "idle",
        *,
        reset_model: bool = False,
        primary: Optional[WebSocket] = None,
        notify: bool = True,
    ) -> None:
        if self._task:
            if self._stop_event:
                self._stop_event.set()
            task = self._task
            self._task = None
            try:
                await task
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Simulation loop finished with %s", exc)

        self._stop_event = None
        self._state = target_state

        if reset_model and self._server.model and self._server.data:
            mujoco.mj_resetData(self._server.model, self._server.data)
            mujoco.mj_forward(self._server.model, self._server.data)

        if notify:
            await self._notify_status(target_state, primary=primary)

        if target_state == "idle":
            self._client = None

    async def _simulation_loop(self, websocket: WebSocket, stop_event: asyncio.Event) -> None:
        logger.info("Simulation loop started")
        try:
            while not stop_event.is_set():
                if not self._server.model or not self._server.data:
                    break

                start = time.perf_counter()
                sim_loop_time = self._server.data.time + self._interval
                while self._server.data.time <= sim_loop_time:
                    mujoco.mj_step(self._server.model, self._server.data)
                frame = self._build_simulation_frame()
                try:
                    await websocket.send_text(json.dumps(frame, separators=(",", ":")))
                except Exception as exc:
                    logger.error("Failed to stream simulation frame: %s", exc)
                    break

                elapsed = time.perf_counter() - start
                sleep_duration = max(0.0, self._interval - elapsed)
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            logger.debug("Simulation loop cancelled")
            raise
        finally:
            logger.info("Simulation loop stopped")

    def _build_simulation_frame(self):
        if not self._server.model or not self._server.data:
            return {"type": "simulation_frame", "time": 0.0, "bodies": []}

        bodies = []
        for body_id in range(1, self._server.model.nbody):
            rel_pos, rel_quat = _compute_body_relative_pose(self._server.model, self._server.data, body_id)
            pos = mujoco2WebPos(rel_pos.tolist())
            quat = mujoco2WebQuat(rel_quat.tolist())
            bodies.append({"id": body_id, "pos": pos, "quat": quat})

        joint_qpos = []
        if self._server.model.nq > 0:
            joint_qpos = np.asarray(self._server.data.qpos[: self._server.model.nq], dtype=float).tolist()

        return {
            "type": "simulation_frame",
            "time": float(self._server.data.time),
            "bodies": bodies,
            "joint_qpos": joint_qpos,
        }

    async def _send_error(self, websocket: WebSocket, message: str) -> None:
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": message}, separators=(",", ":"))
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Failed to send error message: %s", exc)
