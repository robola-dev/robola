import json
import logging
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .model_data import (
    ACTUATOR_SHORTCUT_TAGS,
    extract_tendon_path_segments,
    infer_actuator_shortcut_from_dict,
)

logger = logging.getLogger(__name__)

_SHORTCUT_FLOAT_TOL = 1e-6
COMPILE_COMMENT_TAG = "robola compile"


def _describe_spec_object(obj: Any) -> str:
    try:
        type_name = obj.__class__.__name__
    except Exception:
        type_name = type(obj).__name__
    name = getattr(obj, "name", None)
    parent = getattr(obj, "parent", None)
    parent_name = getattr(parent, "name", None) if parent is not None else None
    details = [type_name]
    if name:
        details.append(f"name='{name}'")
    if parent_name:
        details.append(f"parent='{parent_name}'")
    return " ".join(details)


def _apply_spec_settings(target: Any, settings: Dict[str, Any]) -> None:
    if not target or not settings:
        return
    for key, value in settings.items():
        if value is None:
            continue
        if not hasattr(target, key):
            continue
        attr = getattr(target, key)
        if isinstance(value, dict):
            _apply_spec_settings(attr, value)
            continue
        try:
            setattr(target, key, value)
        except Exception:
            if isinstance(value, (list, tuple)):
                try:
                    if hasattr(attr, "__len__") and len(attr) == len(value):
                        for idx, component in enumerate(value):
                            attr[idx] = component
                        continue
                except Exception:
                    pass
            try:
                setattr(target, key, np.asarray(value))
            except Exception:
                logger.warning(
                    "Failed applying spec setting '%s' on %s with value %r",
                    key,
                    target.__class__.__name__,
                    value,
                    exc_info=True,
                )


def _to_sequence(values: Optional[Iterable[Any]], *, cast: Optional[type] = None) -> Optional[List[Any]]:
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        list_values = values.tolist()
    else:
        list_values = list(values)
    if cast is not None:
        return [cast(v) for v in list_values]
    return list_values


def _convert_numeric_sequence(
    values: Optional[Iterable[Any]],
    *,
    cast: type = float,
    default: Any = 0.0,
    attr_name: Optional[str] = None,
    obj: Any = None,
    raw_values: Optional[Iterable[Any]] = None,
) -> Optional[List[Any]]:
    if values is None:
        return None
    result: List[Any] = []
    sequence = list(values)
    context = _describe_spec_object(obj) if obj is not None else "spec object"
    attr = attr_name or "sequence"
    raw_snapshot = list(raw_values) if raw_values is not None else sequence
    for idx, item in enumerate(sequence):
        if item is None:
            # logger.warning(
            #     "Component for attribute '%s' on %s is None at index %d (raw=%r); defaulting to %r",
            #     attr,
            #     context,
            #     idx,
            #     raw_snapshot,
            #     default,
            # )
            try:
                result.append(cast(default))
            except Exception:
                result.append(default)
            continue
        try:
            result.append(cast(item))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid value %r for attribute '%s' on %s at index %d (raw=%r); defaulting to %r",
                item,
                attr,
                context,
                idx,
                raw_snapshot,
                default,
            )
            try:
                result.append(cast(default))
            except Exception:
                result.append(default)
    return result


def _normalize_size_sequence(values: Optional[Iterable[Any]], *, min_length: int = 3) -> Optional[List[float]]:
    if values is None:
        return None
    size_values = _convert_numeric_sequence(values, cast=float, default=0.0, attr_name="size")
    if size_values is None:
        return None
    while len(size_values) < min_length:
        size_values.append(0.0)
    return size_values


def _to_numpy_array(values: Optional[Iterable[Any]], *, dtype: Optional[Any] = None) -> Optional[np.ndarray]:
    if values is None:
        return None
    try:
        if dtype is not None:
            return np.asarray(values, dtype=dtype)
        return np.asarray(values)
    except Exception as exc:
        logger.error("Failed converting value to numpy array", exc_info=True)
        raise


def _convert_pos(values: Optional[Iterable[float]]) -> Optional[List[float]]:
    if values is None:
        return None
    converted = webPos2Mujoco(values)
    return _convert_numeric_sequence(
        converted,
        cast=float,
        default=0.0,
        attr_name="pos",
        raw_values=values,
    )


def _convert_quat_from_euler(values: Optional[Iterable[float]]) -> Optional[List[float]]:
    if values is None:
        return None
    quat = babylonEuler2Quat(values)
    if isinstance(quat, np.ndarray):
        return quat.tolist()
    return list(quat)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        return float(value)
    except (TypeError, ValueError):
        return default


def _pick_numeric(seq: Optional[Iterable[Any]], index: int, default: float = 0.0) -> float:
    if seq is None:
        return default
    if isinstance(seq, list):
        seq_list = seq
    elif isinstance(seq, tuple):
        seq_list = list(seq)
    else:
        try:
            seq_list = list(seq)
        except TypeError:
            return default
    try:
        value = seq_list[index]
    except IndexError:
        return default
    return _safe_float(value, default)


def _ensure_float_list(values: Optional[Iterable[Any]]) -> List[float]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return [float(x) for x in values.tolist()]
    result: List[float] = []
    for item in values:
        result.append(_safe_float(item, 0.0))
    return result


def _sequence_has_nonzero(values: Optional[Iterable[Any]], *, tol: float = 0.0) -> bool:
    if not values:
        return False
    for item in values:
        if item is None:
            continue
        if abs(_safe_float(item, 0.0)) > tol:
            return True
    return False


def _ensure_shortcut_payload(
    data: Dict[str, Any],
    key: str,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    payload = data.get(key)
    result = defaults.copy()
    if isinstance(payload, dict):
        for field in defaults.keys():
            if field in payload:
                result[field] = payload[field]
    return result


def _pad_float_list(values: Optional[Iterable[Any]], *, length: int, fill: float = 0.0) -> List[float]:
    result: List[float] = []
    if values is not None:
        for item in values:
            if len(result) >= length:
                break
            result.append(_safe_float(item, fill))
    while len(result) < length:
        result.append(fill)
    if len(result) > length:
        result = result[:length]
    return result


def _normalize_range(values: Optional[Iterable[Any]]) -> Optional[List[float]]:
    if values is None:
        return None
    normalized = _pad_float_list(values, length=2, fill=0.0)
    return normalized


def _make_param_vector(values: Optional[Iterable[Any]], *, length: int = 10, fill: float = 0.0) -> List[float]:
    return _pad_float_list(values, length=length, fill=fill)


def _build_motor_parameters(_: Dict[str, Any]) -> Dict[str, Any]:
    gain = _make_param_vector([1.0, 0.0, 0.0])
    bias = _make_param_vector([0.0, 0.0, 0.0])
    dyn = _make_param_vector([1.0, 0.0, 0.0])
    return {
        "gaintype": mujoco.mjtGain.mjGAIN_FIXED.value,
        "biastype": mujoco.mjtBias.mjBIAS_NONE.value,
        "dyntype": mujoco.mjtDyn.mjDYN_NONE.value,
        "dynprm": dyn,
        "gainprm": gain,
        "biasprm": bias,
    }


def _build_position_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = _ensure_shortcut_payload(
        data,
        "shortcut_position",
        {"kp": 1.0, "kv": 0.0, "dampratio": 0.0, "timeconst": 0.0, "inheritrange": 0},
    )
    kp = _safe_float(payload.get("kp"), 1.0)
    kv_raw = payload.get("kv")
    dampratio_raw = payload.get("dampratio")
    timeconst = _safe_float(payload.get("timeconst"), 0.0)
    inheritrange = int(bool(payload.get("inheritrange")))

    kv_value = _safe_float(kv_raw, 0.0) if kv_raw is not None else 0.0
    dampratio_value = _safe_float(dampratio_raw, 0.0) if dampratio_raw is not None else 0.0
    using_kv = kv_raw is not None

    bias_third = -kv_value if using_kv else dampratio_value
    dyntype_value = (
        mujoco.mjtDyn.mjDYN_FILTEREXACT.value
        if timeconst > _SHORTCUT_FLOAT_TOL
        else mujoco.mjtDyn.mjDYN_NONE.value
    )

    dyn = _make_param_vector([timeconst, 0.0, 0.0])
    gain = _make_param_vector([kp, 0.0, 0.0])
    bias = _make_param_vector([0.0, -kp, bias_third])

    return {
        "gaintype": mujoco.mjtGain.mjGAIN_FIXED.value,
        "biastype": mujoco.mjtBias.mjBIAS_AFFINE.value,
        "dyntype": dyntype_value,
        "dynprm": dyn,
        "gainprm": gain,
        "biasprm": bias,
        "inheritrange": inheritrange,
    }


def _build_velocity_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = _ensure_shortcut_payload(data, "shortcut_velocity", {"kv": 1.0})
    kv = _safe_float(payload.get("kv"), 1.0)
    dyn = _make_param_vector([1.0, 0.0, 0.0])
    gain = _make_param_vector([kv, 0.0, 0.0])
    bias = _make_param_vector([0.0, 0.0, -kv])
    return {
        "gaintype": mujoco.mjtGain.mjGAIN_FIXED.value,
        "biastype": mujoco.mjtBias.mjBIAS_AFFINE.value,
        "dyntype": mujoco.mjtDyn.mjDYN_NONE.value,
        "dynprm": dyn,
        "gainprm": gain,
        "biasprm": bias,
    }


def _build_intvelocity_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = _ensure_shortcut_payload(
        data,
        "shortcut_intvelocity",
        {"kp": 1.0, "kv": 0.0, "dampratio": 0.0, "inheritrange": False},
    )
    kp = _safe_float(payload.get("kp"), 1.0)
    kv_raw = payload.get("kv")
    dampratio_raw = payload.get("dampratio")
    inheritrange = int(bool(payload.get("inheritrange")))

    kv_value = _safe_float(kv_raw, 0.0) if kv_raw is not None else 0.0
    dampratio_value = _safe_float(dampratio_raw, 0.0) if dampratio_raw is not None else 0.0
    using_kv = kv_raw is not None
    bias_third = -kv_value if using_kv else dampratio_value

    dyn = _make_param_vector([1.0, 0.0, 0.0])
    gain = _make_param_vector([kp, 0.0, 0.0])
    bias = _make_param_vector([0.0, -kp, bias_third])

    return {
        "gaintype": mujoco.mjtGain.mjGAIN_FIXED.value,
        "biastype": mujoco.mjtBias.mjBIAS_AFFINE.value,
        "dyntype": mujoco.mjtDyn.mjDYN_INTEGRATOR.value,
        "dynprm": dyn,
        "gainprm": gain,
        "biasprm": bias,
        "actlimited": 1,
        "inheritrange": inheritrange,
    }


def _build_damper_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = _ensure_shortcut_payload(data, "shortcut_damper", {"kv": 1.0})
    kv = _safe_float(payload.get("kv"), 1.0)
    ctrlrange_existing = _normalize_range(data.get("ctrlrange"))
    if not _sequence_has_nonzero(ctrlrange_existing, tol=_SHORTCUT_FLOAT_TOL):
        ctrlrange = [0.0, 1.0]
    else:
        ctrlrange = ctrlrange_existing or [0.0, 1.0]
    dyn = _make_param_vector([1.0, 0.0, 0.0])
    gain = _make_param_vector([0.0, 0.0, -kv])
    bias = _make_param_vector([0.0, 0.0, 0.0])
    return {
        "gaintype": mujoco.mjtGain.mjGAIN_AFFINE.value,
        "biastype": mujoco.mjtBias.mjBIAS_NONE.value,
        "dyntype": mujoco.mjtDyn.mjDYN_NONE.value,
        "dynprm": dyn,
        "gainprm": gain,
        "biasprm": bias,
        "ctrllimited": 1,
        "ctrlrange": ctrlrange,
    }


def _build_cylinder_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = _ensure_shortcut_payload(
        data,
        "shortcut_cylinder",
        {"timeconst": 1.0, "area": 1.0, "bias": [0.0, 0.0, 0.0]},
    )
    timeconst = _safe_float(payload.get("timeconst"), 1.0)
    area = _safe_float(payload.get("area"), 1.0)
    bias_values = payload.get("bias")
    bias_source = None if isinstance(bias_values, str) else bias_values
    dyn = _make_param_vector([timeconst, 0.0, 0.0])
    gain = _make_param_vector([area, 0.0, 0.0])
    bias_short = _pad_float_list(bias_source, length=3, fill=0.0)
    bias = _make_param_vector(bias_short)
    return {
        "gaintype": mujoco.mjtGain.mjGAIN_FIXED.value,
        "biastype": mujoco.mjtBias.mjBIAS_AFFINE.value,
        "dyntype": mujoco.mjtDyn.mjDYN_FILTER.value,
        "dynprm": dyn,
        "gainprm": gain,
        "biasprm": bias,
    }


def _build_muscle_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = _ensure_shortcut_payload(
        data,
        "shortcut_muscle",
        {
            "timeconst": [0.01, 0.04],
            "tausmooth": 0.0,
            "range": [0.75, 1.05],
            "force": -1.0,
            "scale": 200.0,
            "lmin": 0.5,
            "lmax": 1.6,
            "vmax": 1.5,
            "fpmax": 1.3,
            "fvmax": 1.2,
        },
    )
    timeconst_vals = _pad_float_list(payload.get("timeconst"), length=2, fill=0.0)
    tausmooth = _safe_float(payload.get("tausmooth"), 0.0)
    range_vals = _pad_float_list(payload.get("range"), length=2, fill=-1.0)
    force = _safe_float(payload.get("force"), -1.0)
    scale = _safe_float(payload.get("scale"), -1.0)
    lmin = _safe_float(payload.get("lmin"), -1.0)
    lmax = _safe_float(payload.get("lmax"), -1.0)
    vmax = _safe_float(payload.get("vmax"), -1.0)
    fpmax = _safe_float(payload.get("fpmax"), -1.0)
    fvmax = _safe_float(payload.get("fvmax"), -1.0)

    gain_params = [
        range_vals[0],
        range_vals[1],
        force,
        scale,
        lmin,
        lmax,
        vmax,
        fpmax,
        fvmax,
    ]

    dyn = _make_param_vector([timeconst_vals[0], timeconst_vals[1], tausmooth])
    gain = _make_param_vector(gain_params, fill=0.0)
    bias = _make_param_vector(gain_params, fill=0.0)

    return {
        "gaintype": mujoco.mjtGain.mjGAIN_MUSCLE.value,
        "biastype": mujoco.mjtBias.mjBIAS_MUSCLE.value,
        "dyntype": mujoco.mjtDyn.mjDYN_MUSCLE.value,
        "dynprm": dyn,
        "gainprm": gain,
        "biasprm": bias,
    }


def _build_adhesion_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = _ensure_shortcut_payload(data, "shortcut_adhesion", {"gain": 1.0})
    gain = _safe_float(payload.get("gain"), 1.0)
    ctrlrange_existing = _normalize_range(data.get("ctrlrange"))
    if not _sequence_has_nonzero(ctrlrange_existing, tol=_SHORTCUT_FLOAT_TOL):
        ctrlrange = [0.0, 1.0]
    else:
        ctrlrange = ctrlrange_existing or [0.0, 1.0]
    dyn = _make_param_vector([1.0, 0.0, 0.0])
    gain_vec = _make_param_vector([gain, 0.0, 0.0])
    bias_vec = _make_param_vector([0.0, 0.0, 0.0])
    return {
        "gaintype": mujoco.mjtGain.mjGAIN_FIXED.value,
        "biastype": mujoco.mjtBias.mjBIAS_NONE.value,
        "dyntype": mujoco.mjtDyn.mjDYN_NONE.value,
        "dynprm": dyn,
        "gainprm": gain_vec,
        "biasprm": bias_vec,
        "trntype": mujoco.mjtTrn.mjTRN_BODY.value,
        "ctrllimited": 1,
        "ctrlrange": ctrlrange,
    }


_SHORTCUT_PARAMETER_BUILDERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "motor": _build_motor_parameters,
    "position": _build_position_parameters,
    "velocity": _build_velocity_parameters,
    "intvelocity": _build_intvelocity_parameters,
    "damper": _build_damper_parameters,
    "cylinder": _build_cylinder_parameters,
    "muscle": _build_muscle_parameters,
    "adhesion": _build_adhesion_parameters,
}


def _apply_actuator_shortcut(actuator: mujoco.MjsActuator, data: Dict[str, Any]) -> None:
    shortcut_value = str(data.get("shortcut") or "").strip().lower()
    if shortcut_value not in ACTUATOR_SHORTCUT_TAGS:
        shortcut_value = infer_actuator_shortcut_from_dict(data)
    builder = _SHORTCUT_PARAMETER_BUILDERS.get(shortcut_value)
    if builder is None:
        return

    try:
        updates = builder(data)
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "Failed to derive actuator shortcut '%s' for %s",
            shortcut_value,
            _describe_spec_object(actuator),
            exc_info=True,
        )
        return

    if updates:
        data.update(updates)
    data["shortcut"] = shortcut_value


def _set_enum(obj: Any, attr: str, enum_cls: Any, value: Optional[Any]) -> None:
    if value is None:
        return
    if isinstance(value, enum_cls):
        enum_value = value
    else:
        enum_value = enum_cls(value)
    setattr(obj, attr, enum_value)


def _maybe_set(obj: Any, attr: str, value: Optional[Any], *, transform=None) -> None:
    if value is None:
        return
    original_value = value
    if transform is not None:
        try:
            value = transform(value)
        except Exception as exc:
            logger.error(
                "Failed transforming attribute '%s' on %s with value %r",
                attr,
                _describe_spec_object(obj),
                original_value,
                exc_info=True,
            )
            raise
        if value is None:
            return
    try:
        setattr(obj, attr, value)
    except Exception as exc:
        logger.error(
            "Failed setting attribute '%s' on %s with value %r",
            attr,
            _describe_spec_object(obj),
            value,
            exc_info=True,
        )
        raise


def _group_by(items: Optional[Iterable[Dict[str, Any]]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    if not items:
        return grouped
    for item in items:
        grouped[item.get(key)].append(item)
    return grouped


def webPos2Mujoco(position):
    """
    Convert a mujoco position to a web-compatible format.
    """
    return [position[0], position[2], position[1]]

def babylonEuler2Quat(euler, degrees=True, invert=True):
    """
    Euler (roll,pitch,yaw) -> MuJoCo quat [w,x,y,z], using scipy, xyz order.
    """
    euler = np.asarray(euler, dtype=float)
    if euler.shape != (3,):
        raise ValueError("euler must be length-3")
    if invert:
        euler[1], euler[2] = euler[2], euler[1]
        euler = -euler
    # create Rotation from euler (xyz)
    rot = R.from_euler('xyz', euler, degrees=degrees)
    q_xyzw = rot.as_quat()   # scipy -> [x, y, z, w]
    # convert to MuJoCo order [w, x, y, z]
    q_mujoco = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
    return q_mujoco


def _clear_spec(save_spec: mujoco.MjSpec) -> None:
    world = save_spec.worldbody
    for geom in list(world.geoms):
        save_spec.delete(geom)
    for site in list(world.sites):
        save_spec.delete(site)
    for camera in list(world.cameras):
        save_spec.delete(camera)
    for light in list(world.lights):
        save_spec.delete(light)
    for joint in list(world.joints):
        save_spec.delete(joint)

    for body in list(save_spec.bodies):
        if body is world:
            continue
        save_spec.delete(body)

    for collection in (
        save_spec.actuators,
        save_spec.tendons,
        save_spec.equalities,
        save_spec.pairs,
        save_spec.excludes,
        save_spec.sensors,
        save_spec.keys,
    ):
        for item in list(collection):
            save_spec.delete(item)

    for asset_collection in (save_spec.textures, save_spec.materials, save_spec.meshes):
        for item in list(asset_collection):
            save_spec.delete(item)


def _apply_body_properties(body: mujoco.MjsBody, data: Dict[str, Any]) -> None:
    if data.get("name"):
        body.name = data["name"]
    _maybe_set(body, "pos", data.get("pos"), transform=_convert_pos)
    if data.get("euler") is not None:
        _maybe_set(body, "quat", data.get("euler"), transform=_convert_quat_from_euler)
    elif data.get("quat") is not None:
        _maybe_set(body, "quat", _to_sequence(data.get("quat")))
    skipinertial = False
    if "explicitinertial" in data:
        body.explicitinertial = bool(data["explicitinertial"])
        if body.explicitinertial:
            skipinertial = True
    if skipinertial:
        _maybe_set(body, "mass", data.get("mass"))
        _maybe_set(body, "ipos", data.get("ipos"), transform=_convert_pos)
        if data.get("ieuler") is not None:
            _maybe_set(body, "iquat", data.get("ieuler"), transform=_convert_quat_from_euler)
        elif data.get("iquat") is not None:
            _maybe_set(body, "iquat", _to_sequence(data.get("iquat")))
        _maybe_set(body, "inertia", data.get("inertia"), transform=_convert_pos)
    if "mocap" in data:
        body.mocap = bool(data["mocap"])
    if "gravcomp" in data:
        body.gravcomp = float(data["gravcomp"])
    if "userdata" in data:
        body.userdata = _to_sequence(data.get("userdata")) or []
    if "explicitinertial" in data:
        body.explicitinertial = bool(data["explicitinertial"])


def _apply_joint_properties(joint: mujoco.MjsJoint, data: Dict[str, Any]) -> None:
    if data.get("name"):
        joint.name = data["name"]
    _set_enum(joint, "type", mujoco.mjtJoint, data.get("type"))
    _maybe_set(joint, "pos", data.get("pos"), transform=_convert_pos)
    _maybe_set(joint, "axis", data.get("axis"), transform=_convert_pos)
    _maybe_set(joint, "ref", data.get("ref"))
    _maybe_set(joint, "stiffness", data.get("stiffness"))
    _maybe_set(joint, "springref", data.get("springref"))
    _maybe_set(joint, "springdamper", data.get("springdamper"), transform=lambda v: _to_sequence(v, cast=float))
    if "limited" in data:
        joint.limited = data["limited"]
    _maybe_set(joint, "range", data.get("range"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(joint, "margin", data.get("margin"))
    _maybe_set(joint, "solref_limit", data.get("solref_limit"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(joint, "solimp_limit", data.get("solimp_limit"), transform=lambda v: _to_sequence(v, cast=float))
    if "actfrclimited" in data:
        joint.actfrclimited = data["actfrclimited"]
    _maybe_set(joint, "actfrcrange", data.get("actfrcrange"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(joint, "armature", data.get("armature"))
    _maybe_set(joint, "damping", data.get("damping"))
    _maybe_set(joint, "frictionloss", data.get("frictionloss"))
    _maybe_set(joint, "solref_friction", data.get("solref_friction"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(joint, "solimp_friction", data.get("solimp_friction"), transform=lambda v: _to_sequence(v, cast=float))
    if "group" in data:
        joint.group = data["group"]
    if "actgravcomp" in data:
        joint.actgravcomp = bool(data["actgravcomp"])
    if "userdata" in data:
        joint.userdata = _to_sequence(data.get("userdata")) or []


def _apply_geom_properties(geom: mujoco.MjsGeom, data: Dict[str, Any]) -> None:
    if data.get("name"):
        geom.name = data["name"]
    _set_enum(geom, "type", mujoco.mjtGeom, data.get("type"))
    enable_fromto = bool(data.get("enableFromTo"))
    fromto = data.get("fromto") if enable_fromto else None
    use_fromto = False
    if fromto is not None and not _sequence_has_nonzero(fromto):
        fromto = None
    if fromto is not None:
        # MuJoCo uses a different fromto order than the web format
        fromto[1], fromto[2], fromto[4], fromto[5] = fromto[2], fromto[1], fromto[5], fromto[4]
        supported_fromto_types = {
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_BOX,
            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        }
        geom_type_enum = None
        try:
            geom_type_enum = mujoco.mjtGeom(int(geom.type))
        except Exception:
            try:
                geom_type_enum = (
                    geom.type if isinstance(geom.type, mujoco.mjtGeom) else None
                )
            except Exception:
                geom_type_enum = None

        if geom_type_enum in supported_fromto_types:
            cleaned = _convert_numeric_sequence(
                fromto,
                cast=float,
                default=np.nan,
                attr_name="fromto",
                obj=geom,
            )
            if cleaned is not None:
                geom.fromto = cleaned
                use_fromto = True
        else:
            type_label = None
            if geom_type_enum is not None and hasattr(geom_type_enum, "name"):
                type_label = geom_type_enum.name
            else:
                type_label = str(getattr(geom, "type", "unknown"))
            # logger.warning(
            #     "Skipping fromto for %s because geom type %s does not support it",
            #     _describe_spec_object(geom),
            #     type_label,
            # )
    if not use_fromto:
        _maybe_set(geom, "pos", data.get("pos"), transform=_convert_pos)
        if data.get("euler") is not None:
            _maybe_set(geom, "quat", data.get("euler"), transform=_convert_quat_from_euler)
        elif data.get("quat") is not None:
            _maybe_set(geom, "quat", _to_sequence(data.get("quat")))
    if data.get("type") == 4 or data.get("type") == 6:
        _maybe_set(geom, "size", [data.get("size")[0], data.get("size")[2], data.get("size")[1]], transform=_normalize_size_sequence)
    else:
        _maybe_set(geom, "size", data.get("size"), transform=_normalize_size_sequence)
    if "contype" in data:
        geom.contype = data["contype"]
    if "conaffinity" in data:
        geom.conaffinity = data["conaffinity"]
    if "condim" in data:
        geom.condim = data["condim"]
    if "priority" in data:
        geom.priority = data["priority"]
    _maybe_set(geom, "friction", data.get("friction"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(geom, "solmix", data.get("solmix"))
    _maybe_set(geom, "solref", data.get("solref"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(geom, "solimp", data.get("solimp"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(geom, "margin", data.get("margin"))
    _maybe_set(geom, "gap", data.get("gap"))
    if data.get("mass") is not None:
        geom.mass = float(data["mass"])
    if data.get("density") is not None:
        geom.density = float(data["density"])
    if data.get("typeinertia") is not None:
        _set_enum(geom, "typeinertia", mujoco.mjtGeomInertia, data.get("typeinertia"))
    if data.get("fluid_ellipsoid") is not None:
        geom.fluid_ellipsoid = bool(data["fluid_ellipsoid"])
    _maybe_set(geom, "fluid_coefs", data.get("fluid_coefs"), transform=lambda v: _to_sequence(v, cast=float))
    material_value = data.get("material")
    if material_value is not None:
        geom.material = material_value
    use_material = isinstance(material_value, str) and material_value.strip() != ""
    if not use_material:
        _maybe_set(geom, "rgba", data.get("rgba"), transform=lambda v: _to_sequence(v, cast=float))
    if "group" in data:
        geom.group = data["group"]
    if data.get("hfieldname") is not None:
        geom.hfieldname = data["hfieldname"]
    if data.get("meshname") is not None:
        geom.meshname = data["meshname"]
    if data.get("fitscale") is not None:
        geom.fitscale = data["fitscale"]
    if "userdata" in data:
        geom.userdata = _to_sequence(data.get("userdata")) or []


def _apply_site_properties(site: mujoco.MjsSite, data: Dict[str, Any]) -> None:
    if data.get("name"):
        site.name = data["name"]
    _set_enum(site, "type", mujoco.mjtGeom , data.get("type"))
    enable_fromto = bool(data.get("enableFromTo"))
    fromto = data.get("fromto") if enable_fromto else None
    if fromto is not None and not _sequence_has_nonzero(fromto):
        fromto = None
    use_fromto = False
    if fromto is not None:
        fromto[1], fromto[2], fromto[4], fromto[5] = fromto[2], fromto[1], fromto[5], fromto[4]
        supported_fromto_types = {
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_BOX,
            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        }
        geom_type_enum = None
        try:
            geom_type_enum = mujoco.mjtGeom(int(site.type))
        except Exception:
            try:
                geom_type_enum = (
                    site.type if isinstance(site.type, mujoco.mjtGeom) else None
                )
            except Exception:
                geom_type_enum = None

        if geom_type_enum in supported_fromto_types:
            cleaned = _convert_numeric_sequence(
                fromto,
                cast=float,
                default=np.nan,
                attr_name="fromto",
                obj=site,
            )
            if cleaned is not None:
                site.fromto = cleaned
                use_fromto = True
        else:
            type_label = None
            if geom_type_enum is not None and hasattr(geom_type_enum, "name"):
                type_label = geom_type_enum.name
            else:
                type_label = str(getattr(site, "type", "unknown"))
            # logger.warning(
            #     "Skipping fromto for %s because geom type %s does not support it",
            #     _describe_spec_object(site),
            #     type_label,
            # )
    if not use_fromto:
        _maybe_set(site, "pos", data.get("pos"), transform=_convert_pos)
        if data.get("euler") is not None:
            _maybe_set(site, "quat", data.get("euler"), transform=_convert_quat_from_euler)
        elif data.get("quat") is not None:
            _maybe_set(site, "quat", _to_sequence(data.get("quat")))
    if data.get("type") == 4 or data.get("type") == 6:
        _maybe_set(site, "size", [data.get("size")[0], data.get("size")[2], data.get("size")[1]], transform=_normalize_size_sequence)
    else:
        _maybe_set(site, "size", data.get("size"), transform=_normalize_size_sequence)
    material_value = data.get("material")
    if material_value is not None:
        site.material = material_value
    use_material = isinstance(material_value, str) and material_value.strip() != ""
    if "group" in data:
        site.group = data["group"]
    if not use_material:
        _maybe_set(site, "rgba", data.get("rgba"), transform=lambda v: _to_sequence(v, cast=float))
    if "userdata" in data:
        site.userdata = _to_sequence(data.get("userdata")) or []


def _apply_camera_properties(camera: mujoco.MjsCamera, data: Dict[str, Any]) -> None:
    if data.get("name"):
        camera.name = data["name"]
    _maybe_set(camera, "pos", data.get("pos"), transform=_convert_pos)
    if data.get("euler") is not None:
        _maybe_set(camera, "quat", data.get("euler"), transform=_convert_quat_from_euler)
    elif data.get("quat") is not None:
        _maybe_set(camera, "quat", _to_sequence(data.get("quat")))
    _set_enum(camera, "mode", mujoco.mjtCamLight, data.get("mode"))
    if data.get("targetbody") is not None:
        camera.targetbody = data["targetbody"]
    if data.get("orthographic") is not None:
        camera.orthographic = bool(data["orthographic"])
    _maybe_set(camera, "fovy", data.get("fovy"))
    _maybe_set(camera, "ipd", data.get("ipd"))
    _maybe_set(camera, "intrinsic", data.get("intrinsic"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(camera, "sensor_size", data.get("sensor_size"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(camera, "resolution", data.get("resolution"), transform=lambda v: _to_sequence(v, cast=int))
    _maybe_set(camera, "focal_length", data.get("focal_length"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(camera, "focal_pixel", data.get("focal_pixel"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(camera, "principal_length", data.get("principal_length"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(camera, "principal_pixel", data.get("principal_pixel"), transform=lambda v: _to_sequence(v, cast=float))
    if "userdata" in data:
        camera.userdata = _to_sequence(data.get("userdata")) or []


def _apply_light_properties(light: mujoco.MjsLight, data: Dict[str, Any]) -> None:
    if data.get("name"):
        light.name = data["name"]
    _maybe_set(light, "pos", data.get("pos"), transform=_convert_pos)
    _maybe_set(light, "dir", data.get("dir"), transform=_convert_pos)
    _set_enum(light, "mode", mujoco.mjtCamLight, data.get("mode"))
    if data.get("targetbody") is not None:
        light.targetbody = data["targetbody"]
    if data.get("active") is not None:
        light.active = bool(data["active"])
    _set_enum(light, "type", mujoco.mjtLightType, data.get("type"))
    if data.get("texture") is not None:
        light.texture = data["texture"]
    if data.get("castshadow") is not None:
        light.castshadow = bool(data["castshadow"])
    _maybe_set(light, "bulbradius", data.get("bulbradius"))
    _maybe_set(light, "intensity", data.get("intensity"))
    _maybe_set(light, "range", data.get("range"))
    _maybe_set(light, "attenuation", data.get("attenuation"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(light, "cutoff", data.get("cutoff"))
    _maybe_set(light, "exponent", data.get("exponent"))
    _maybe_set(light, "ambient", data.get("ambient"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(light, "diffuse", data.get("diffuse"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(light, "specular", data.get("specular"), transform=lambda v: _to_sequence(v, cast=float))


def _apply_mesh_properties(mesh: mujoco.MjsMesh, data: Dict[str, Any]) -> None:
    if data.get("name"):
        mesh.name = data["name"]
    if data.get("file") is not None:
        mesh.file = data["file"]
    _maybe_set(mesh, "refpos", data.get("refpos"), transform=_convert_pos)
    if data.get("refeuler") is not None:
        _maybe_set(mesh, "refquat", data.get("refeuler"), transform=_convert_quat_from_euler)
    elif data.get("refquat") is not None:
        _maybe_set(mesh, "refquat", _to_sequence(data.get("refquat")))
    _maybe_set(mesh, "scale", data.get("scale"), transform=lambda v: _to_sequence(v, cast=float))
    if data.get("inertiatype") is not None:
        _set_enum(mesh, "inertia", mujoco.mjtMeshInertia, data.get("inertiatype"))
    if data.get("smoothnormal") is not None:
        mesh.smoothnormal = bool(data["smoothnormal"])
    if data.get("needsdf") is not None:
        mesh.needsdf = bool(data["needsdf"])
    if data.get("maxhullvert") is not None:
        mesh.maxhullvert = int(data["maxhullvert"])
    _maybe_set(mesh, "userface", data.get("userface"), transform=lambda v: _to_numpy_array(v, dtype=int))
    _maybe_set(mesh, "userfacetexcoord", data.get("userfacetexcoord"), transform=lambda v: _to_numpy_array(v, dtype=int))
    _maybe_set(mesh, "usernormal", data.get("usernormal"), transform=lambda v: _to_numpy_array(v, dtype=float))
    _maybe_set(mesh, "usertexcoord", data.get("usertexcoord"), transform=lambda v: _to_numpy_array(v, dtype=float))
    _maybe_set(mesh, "uservert", data.get("uservert"), transform=lambda v: _to_numpy_array(v, dtype=float))


def _apply_texture_properties(texture: mujoco.MjsTexture, data: Dict[str, Any]) -> None:
    if data.get("name"):
        texture.name = data["name"]
    _set_enum(texture, "type", mujoco.mjtTexture, data.get("type"))
    _set_enum(texture, "colorspace", mujoco.mjtColorSpace, data.get("colorspace"))
    if data.get("builtin") is not None:
        texture.builtin = data["builtin"]
    if data.get("mark") is not None:
        texture.mark = data["mark"]
    _maybe_set(texture, "rgb1", data.get("rgb1"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(texture, "rgb2", data.get("rgb2"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(texture, "markrgb", data.get("markrgb"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(texture, "random", data.get("random"))
    if data.get("height") is not None:
        texture.height = int(data["height"])
    if data.get("width") is not None:
        texture.width = int(data["width"])
    if data.get("nchannel") is not None:
        texture.nchannel = int(data["nchannel"])
    if data.get("content_type") is not None:
        texture.content_type = data["content_type"]
    if data.get("file") is not None:
        texture.file = data["file"]
    _maybe_set(texture, "gridsize", data.get("gridsize"), transform=lambda v: _to_sequence(v, cast=int))
    if data.get("gridlayout") is not None:
        texture.gridlayout = _to_sequence(data.get("gridlayout")) or []
    cubefiles_raw = data.get("cubefiles")
    if cubefiles_raw:
        cubefiles_seq = _to_sequence(cubefiles_raw)
        cleaned_cubefiles = [
            str(entry)
            for entry in (cubefiles_seq or [])
            if entry is not None and str(entry).strip() != ""
        ]
        if cleaned_cubefiles:
            texture.cubefiles = cleaned_cubefiles
    if data.get("hflip") is not None:
        texture.hflip = bool(data["hflip"])
    if data.get("vflip") is not None:
        texture.vflip = bool(data["vflip"])


def _apply_material_properties(material: mujoco.MjsMaterial, data: Dict[str, Any]) -> None:
    if data.get("name"):
        material.name = data["name"]
    if data.get("texture") is not None:
        textures = _to_sequence(data.get("texture")) or []
        material.textures = textures
    if data.get("texuniform") is not None:
        material.texuniform = bool(data["texuniform"])
    _maybe_set(material, "texrepeat", data.get("texrepeat"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(material, "emission", data.get("emission"))
    _maybe_set(material, "specular", data.get("specular"))
    _maybe_set(material, "shininess", data.get("shininess"))
    _maybe_set(material, "reflectance", data.get("reflectance"))
    _maybe_set(material, "metallic", data.get("metallic"))
    _maybe_set(material, "roughness", data.get("roughness"))
    _maybe_set(material, "rgba", data.get("rgba"), transform=lambda v: _to_sequence(v, cast=float))


def _apply_pair_properties(pair: mujoco.MjsPair, data: Dict[str, Any]) -> None:
    if data.get("name"):
        pair.name = data["name"]
    if data.get("geomname1") is not None:
        pair.geomname1 = data["geomname1"]
    if data.get("geomname2") is not None:
        pair.geomname2 = data["geomname2"]
    if data.get("condim") is not None:
        pair.condim = data["condim"]
    _maybe_set(pair, "solref", data.get("solref"), transform=lambda v: _to_sequence(v, cast=float))
    if data.get("solreffriction") is not None:
        pair.solreffriction = _to_sequence(data.get("solreffriction"), cast=float) or []
    _maybe_set(pair, "solimp", data.get("solimp"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(pair, "margin", data.get("margin"))
    _maybe_set(pair, "gap", data.get("gap"))
    _maybe_set(pair, "friction", data.get("friction"), transform=lambda v: _to_sequence(v, cast=float))


def _apply_exclude_properties(exclude: mujoco.MjsExclude, data: Dict[str, Any]) -> None:
    if data.get("name"):
        exclude.name = data["name"]
    if data.get("bodyname1") is not None:
        exclude.bodyname1 = data["bodyname1"]
    if data.get("bodyname2") is not None:
        exclude.bodyname2 = data["bodyname2"]


def _build_equality_data_array(data: Dict[str, Any]) -> Optional[List[float]]:
    """Rebuild the MuJoCo equality.data array from the frontend field structure."""

    type_value = data.get("type")
    if type_value is None and data.get("data") is None:
        return None

    if isinstance(type_value, mujoco.mjtEq):
        eq_type = type_value
    else:
        try:
            eq_type = mujoco.mjtEq(int(type_value))
        except (TypeError, ValueError):
            eq_type = None

    if eq_type == mujoco.mjtEq.mjEQ_CONNECT:
        values: List[float] = [0.0] * 11
        anchor_values = _convert_pos(data.get("anchor"))
        if anchor_values is not None:
            for idx in range(min(3, len(anchor_values))):
                values[idx] = anchor_values[idx]
        torquescale = data.get("torquescale")
        if torquescale is not None:
            values[-1] = _safe_float(torquescale, 0.0)
        return values

    if eq_type == mujoco.mjtEq.mjEQ_WELD:
        values = [0.0] * 11
        anchor_values = _convert_pos(data.get("anchor"))
        if anchor_values is not None:
            for idx in range(min(3, len(anchor_values))):
                values[idx] = anchor_values[idx]
        relpos_values = _convert_pos(data.get("relpos"))
        if relpos_values is not None:
            for idx in range(min(3, len(relpos_values))):
                values[3 + idx] = relpos_values[idx]
        releuler_raw = data.get("releuler")
        releuler_values = _convert_numeric_sequence(
            releuler_raw,
            cast=float,
            default=0.0,
            attr_name="releuler",
        )
        if releuler_values is not None:
            while len(releuler_values) < 3:
                releuler_values.append(0.0)
            relquat = _convert_quat_from_euler(releuler_values[:3])
            if relquat is not None:
                for idx in range(min(4, len(relquat))):
                    values[6 + idx] = float(relquat[idx])
        torquescale = data.get("torquescale")
        if torquescale is not None:
            values[-1] = _safe_float(torquescale, 0.0)
        return values

    if eq_type in (mujoco.mjtEq.mjEQ_JOINT, mujoco.mjtEq.mjEQ_TENDON):
        values = [0.0] * 11
        polycoef_raw = data.get("polycoef")
        polycoef_values = _convert_numeric_sequence(
            polycoef_raw,
            cast=float,
            default=0.0,
            attr_name="polycoef",
        )
        if polycoef_values is None:
            return values
        for idx in range(min(5, len(polycoef_values))):
            values[idx] = polycoef_values[idx]
        return values

    existing_data = data.get("data")
    if existing_data is None:
        return None
    converted = _convert_numeric_sequence(existing_data, cast=float, default=0.0, attr_name="data")
    return converted


def _apply_equality_properties(equality: mujoco.MjsEquality, data: Dict[str, Any]) -> None:
    if data.get("name"):
        equality.name = data["name"]
    _set_enum(equality, "type", mujoco.mjtEq, data.get("type"))
    equality_data_values = _build_equality_data_array(data)
    if equality_data_values is not None:
        try:
            array_value = np.asarray(equality_data_values, dtype=float)
            if array_value.ndim == 1:
                array_value = array_value.reshape(-1, 1)
            setattr(equality, "data", array_value)
        except Exception:
            logger.warning(
                "Failed to reshape equality data for %s, falling back to generic assignment",
                _describe_spec_object(equality),
                exc_info=True,
            )
            _maybe_set(equality, "data", equality_data_values, transform=lambda v: _to_sequence(v, cast=float))
    else:
        _maybe_set(equality, "data", data.get("data"), transform=lambda v: _to_sequence(v, cast=float))
    if data.get("active") is not None:
        equality.active = bool(data["active"])
    if data.get("name1") is not None:
        equality.name1 = data["name1"]
    if data.get("name2") is not None:
        equality.name2 = data["name2"]
    _set_enum(equality, "objtype", mujoco.mjtObj, data.get("objtype"))
    _maybe_set(equality, "solref", data.get("solref"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(equality, "solimp", data.get("solimp"), transform=lambda v: _to_sequence(v, cast=float))


def _apply_tendon_properties(tendon: mujoco.MjsTendon, data: Dict[str, Any]) -> None:
    if data.get("name"):
        tendon.name = data["name"]
    _maybe_set(tendon, "stiffness", data.get("stiffness"))
    _maybe_set(tendon, "springlength", data.get("springlength"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(tendon, "damping", data.get("damping"))
    _maybe_set(tendon, "frictionloss", data.get("frictionloss"))
    _maybe_set(tendon, "solref_friction", data.get("solref_friction"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(tendon, "solimp_friction", data.get("solimp_friction"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(tendon, "armature", data.get("armature"))
    if "limited" in data:
        tendon.limited = data["limited"]
    if "actfrclimited" in data:
        tendon.actfrclimited = data["actfrclimited"]
    _maybe_set(tendon, "range", data.get("range"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(tendon, "actfrcrange", data.get("actfrcrange"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(tendon, "margin", data.get("margin"))
    _maybe_set(tendon, "solref_limit", data.get("solref_limit"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(tendon, "solimp_limit", data.get("solimp_limit"), transform=lambda v: _to_sequence(v, cast=float))
    if data.get("material") is not None:
        tendon.material = data["material"]
    _maybe_set(tendon, "width", data.get("width"))
    _maybe_set(tendon, "rgba", data.get("rgba"), transform=lambda v: _to_sequence(v, cast=float))
    if "group" in data:
        tendon.group = data["group"]
    if "userdata" in data:
        tendon.userdata = _to_sequence(data.get("userdata")) or []


def _apply_tendon_path(
    tendon: mujoco.MjsTendon,
    path_segments: Optional[List[Dict[str, Any]]],
) -> None:
    """Rebuild the tendon path from the provided path segments."""

    if not path_segments:
        tendon_name = getattr(tendon, "name", None) or ""
        raise ValueError(f"tendon '{tendon_name}' path cannot be empty")

    for index, segment in enumerate(path_segments):
        wrap_type = segment.get("wrap_type")
        kind = segment.get("kind")
        geom_type = segment.get("geom_type")
        objtype = segment.get("objtype")
        param_value = segment.get("param")
        if param_value is None:
            param_value = segment.get("coef")
        target = segment.get("target") or segment.get("joint") or segment.get("site") or segment.get("geom")

        if wrap_type is None and isinstance(kind, str):
            kind_lower = kind.lower()
            if kind_lower == "joint":
                wrap_type = mujoco.mjtWrap.mjWRAP_JOINT.value
            elif kind_lower == "pulley":
                wrap_type = mujoco.mjtWrap.mjWRAP_PULLEY.value
            elif kind_lower == "site":
                wrap_type = mujoco.mjtWrap.mjWRAP_SITE.value
            elif kind_lower == "geom":
                if isinstance(geom_type, str) and geom_type.lower() == "sphere":
                    wrap_type = mujoco.mjtWrap.mjWRAP_SPHERE.value
                elif isinstance(geom_type, str) and geom_type.lower() == "cylinder":
                    wrap_type = mujoco.mjtWrap.mjWRAP_CYLINDER.value

        try:
            if wrap_type == mujoco.mjtWrap.mjWRAP_JOINT.value:
                if not target:
                    raise ValueError("joint segment missing target name")
                coef = _safe_float(param_value, 1.0)
                tendon.wrap_joint(str(target), coef)
            elif wrap_type == mujoco.mjtWrap.mjWRAP_PULLEY.value:
                divisor = _safe_float(param_value, 1.0)
                tendon.wrap_pulley(divisor)
            elif wrap_type == mujoco.mjtWrap.mjWRAP_SITE.value or objtype == mujoco.mjtObj.mjOBJ_SITE:
                if not target:
                    raise ValueError("site segment missing target name")
                tendon.wrap_site(str(target))
            elif wrap_type == mujoco.mjtWrap.mjWRAP_SPHERE.value:
                if not target:
                    raise ValueError("sphere segment missing geom name")
                sidesite = str(param_value) if isinstance(param_value, str) else ""
                tendon.wrap_geom(str(target), sidesite)
            elif wrap_type == mujoco.mjtWrap.mjWRAP_CYLINDER.value:
                if not target:
                    raise ValueError("cylinder segment missing geom name")
                sidesite = str(param_value) if isinstance(param_value, str) else ""
                tendon.wrap_geom(str(target), sidesite)
            else:
                raise ValueError(f"unsupported tendon path segment: {segment}")
        except Exception as exc:
            tendon_name = getattr(tendon, "name", None) or ""
            logger.error(
                "Failed to rebuild tendon path segment %s (index %d) for tendon '%s'",
                segment,
                index,
                tendon_name,
                exc_info=True,
            )
            raise


def _apply_actuator_properties(actuator: mujoco.MjsActuator, data: Dict[str, Any]) -> None:
    _apply_actuator_shortcut(actuator, data)
    if data.get("name"):
        actuator.name = data["name"]
    _set_enum(actuator, "gaintype", mujoco.mjtGain, data.get("gaintype"))
    _maybe_set(actuator, "gainprm", data.get("gainprm"), transform=lambda v: _to_sequence(v, cast=float))
    _set_enum(actuator, "biastype", mujoco.mjtBias, data.get("biastype"))
    _maybe_set(actuator, "biasprm", data.get("biasprm"), transform=lambda v: _to_sequence(v, cast=float))
    _set_enum(actuator, "dyntype", mujoco.mjtDyn, data.get("dyntype"))
    _maybe_set(actuator, "dynprm", data.get("dynprm"), transform=lambda v: _to_sequence(v, cast=float))
    if data.get("actdim") is not None:
        actuator.actdim = int(data["actdim"])
    if data.get("actearly") is not None:
        actuator.actearly = int(data["actearly"])
    _set_enum(actuator, "trntype", mujoco.mjtTrn, data.get("trntype"))
    _maybe_set(actuator, "gear", data.get("gear"), transform=lambda v: _to_sequence(v, cast=float))
    if data.get("target") is not None:
        actuator.target = data["target"]
    if data.get("refsite") is not None:
        actuator.refsite = data["refsite"]
    if data.get("slidersite") is not None:
        actuator.slidersite = data["slidersite"]
    _maybe_set(actuator, "cranklength", data.get("cranklength"))
    _maybe_set(actuator, "lengthrange", data.get("lengthrange"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(actuator, "inheritrange", data.get("inheritrange"))
    if "ctrllimited" in data:
        actuator.ctrllimited = data["ctrllimited"]
    _maybe_set(actuator, "ctrlrange", data.get("ctrlrange"), transform=lambda v: _to_sequence(v, cast=float))
    if "forcelimited" in data:
        actuator.forcelimited = data["forcelimited"]
    _maybe_set(actuator, "forcerange", data.get("forcerange"), transform=lambda v: _to_sequence(v, cast=float))
    if "actlimited" in data:
        actuator.actlimited = data["actlimited"]
    _maybe_set(actuator, "actrange", data.get("actrange"), transform=lambda v: _to_sequence(v, cast=float))
    if "group" in data:
        actuator.group = data["group"]
    if "userdata" in data:
        actuator.userdata = _to_sequence(data.get("userdata")) or []


def _apply_sensor_properties(sensor: mujoco.MjsSensor, data: Dict[str, Any]) -> None:
    if data.get("name"):
        sensor.name = data["name"]
    _set_enum(sensor, "type", mujoco.mjtSensor, data.get("type"))
    _set_enum(sensor, "objtype", mujoco.mjtObj, data.get("objtype"))
    if data.get("objname") is not None or data.get("objname") != "":
        sensor.objname = data["objname"]
    _set_enum(sensor, "reftype", mujoco.mjtObj, data.get("reftype"))
    if data.get("refname") is not None or data.get("refname") != "":
        sensor.refname = data["refname"]
    _set_enum(sensor, "datatype", mujoco.mjtDataType, data.get("datatype"))
    _set_enum(sensor, "needstage", mujoco.mjtStage, data.get("needstage"))
    if data.get("dim") is not None:
        sensor.dim = int(data["dim"])
    _maybe_set(sensor, "cutoff", data.get("cutoff"))
    _maybe_set(sensor, "noise", data.get("noise"))
    if "userdata" in data:
        sensor.userdata = _to_sequence(data.get("userdata")) or []


def _apply_key_properties(key: mujoco.MjsKey, data: Dict[str, Any]) -> None:
    if data.get("time") is not None:
        key.time = float(data["time"])
    _maybe_set(key, "name", data.get("name"))
    _maybe_set(key, "qpos", data.get("qpos"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(key, "qvel", data.get("qvel"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(key, "act", data.get("act"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(key, "mpos", data.get("mpos"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(key, "mquat", data.get("mquat"), transform=lambda v: _to_sequence(v, cast=float))
    _maybe_set(key, "ctrl", data.get("ctrl"), transform=lambda v: _to_sequence(v, cast=float))


def _rebuild_assets(save_spec: mujoco.MjSpec, model_dict: Dict[str, Any]) -> None:
    for texture_data in model_dict.get("textures", []) or []:
        texture = save_spec.add_texture()
        _apply_texture_properties(texture, texture_data)

    for material_data in model_dict.get("materials", []) or []:
        material = save_spec.add_material()
        _apply_material_properties(material, material_data)

    for mesh_data in model_dict.get("meshes", []) or []:
        mesh = save_spec.add_mesh()
        _apply_mesh_properties(mesh, mesh_data)


def _rebuild_bodies_and_children(save_spec: mujoco.MjSpec, model_dict: Dict[str, Any]) -> Dict[int, mujoco.MjsBody]:
    bodies_data = model_dict.get("bodies", []) or []
    if not bodies_data:
        return {0: save_spec.worldbody}

    body_map: Dict[int, mujoco.MjsBody] = {}
    joints_by_body = _group_by(model_dict.get("joints"), "body_id")
    geoms_by_body = _group_by(model_dict.get("geoms"), "body_id")
    sites_by_body = _group_by(model_dict.get("sites"), "body_id")
    cameras_by_body = _group_by(model_dict.get("cameras"), "body_id")
    lights_by_body = _group_by(model_dict.get("lights"), "body_id")

    for body_data in sorted(bodies_data, key=lambda item: item.get("id", 0)):
        body_id = body_data.get("id")
        parent_id = body_data.get("parent_id", -1)

        if body_id == 0 or parent_id in (-1, None):
            body = save_spec.worldbody
        else:
            parent_body = body_map.get(parent_id)
            if parent_body is None:
                raise ValueError(f"Missing parent body {parent_id} for body {body_id}")
            body = parent_body.add_body()

        _apply_body_properties(body, body_data)
        if body_id is not None:
            body_map[body_id] = body

        for joint_data in joints_by_body.get(body_id, []):
            joint = body.add_joint()
            _apply_joint_properties(joint, joint_data)

        for geom_data in geoms_by_body.get(body_id, []):
            geom = body.add_geom()
            _apply_geom_properties(geom, geom_data)

        for site_data in sites_by_body.get(body_id, []):
            site = body.add_site()
            _apply_site_properties(site, site_data)

        for camera_data in cameras_by_body.get(body_id, []):
            camera = body.add_camera()
            _apply_camera_properties(camera, camera_data)

        for light_data in lights_by_body.get(body_id, []):
            light = body.add_light()
            _apply_light_properties(light, light_data)

    if 0 not in body_map:
        body_map[0] = save_spec.worldbody

    return body_map


def _rebuild_other_collections(
    save_spec: mujoco.MjSpec,
    model_dict: Dict[str, Any],
    *,
    tendon_path_fallback: Optional[Dict[int, List[Dict[str, Any]]]] = None,
) -> None:
    for pair_data in model_dict.get("pairs", []) or []:
        pair = save_spec.add_pair()
        _apply_pair_properties(pair, pair_data)

    for exclude_data in model_dict.get("excludes", []) or []:
        exclude = save_spec.add_exclude()
        _apply_exclude_properties(exclude, exclude_data)

    for equality_data in model_dict.get("equalities", []) or []:
        equality = save_spec.add_equality()
        _apply_equality_properties(equality, equality_data)

    for tendon_data in model_dict.get("tendons", []) or []:
        tendon = save_spec.add_tendon()
        _apply_tendon_properties(tendon, tendon_data)
        path_segments = tendon_data.get("path")
        if not path_segments and tendon_path_fallback:
            tendon_id_raw = tendon_data.get("id")
            if tendon_id_raw is not None:
                try:
                    tendon_id = int(tendon_id_raw)
                except (TypeError, ValueError):
                    tendon_id = None
                else:
                    path_segments = tendon_path_fallback.get(tendon_id)
        _apply_tendon_path(tendon, path_segments)

    for actuator_data in model_dict.get("actuators", []) or []:
        actuator = save_spec.add_actuator()
        _apply_actuator_properties(actuator, actuator_data)

    for sensor_data in model_dict.get("sensors", []) or []:
        sensor = save_spec.add_sensor()
        _apply_sensor_properties(sensor, sensor_data)

    for key_data in model_dict.get("keys", []) or []:
        key = save_spec.add_key()
        _apply_key_properties(key, key_data)

def save_model_data(spec: mujoco.MjSpec, model: mujoco.MjModel, model_data: dict, path: str) -> Dict[str, Any]:
    """Rebuild an MjSpec from the frontend-provided model_data and write it to an MJCF file."""

    target_path = Path(os.path.abspath(path))
    target_path.parent.mkdir(parents=True, exist_ok=True)

    save_spec = mujoco.MjSpec()
    if spec and spec.modelname:
        save_spec.modelname = spec.modelname

    compiler_settings_raw = model_data.get("compiler") or {}
    compiler_settings = dict(compiler_settings_raw)
    meshdir_override = compiler_settings.pop("meshdir", None)
    texturedir_override = compiler_settings.pop("texturedir", None)

    option_settings = model_data.get("option") or {}
    stat_settings = model_data.get("stat") or {}
    visual_settings = model_data.get("visual") or {}

    if compiler_settings:
        _apply_spec_settings(save_spec.compiler, compiler_settings)
    if option_settings:
        _apply_spec_settings(save_spec.option, option_settings)
    if stat_settings:
        _apply_spec_settings(save_spec.stat, stat_settings)
    if visual_settings:
        _apply_spec_settings(save_spec.visual, visual_settings)

    root_path = target_path.parent

    spec_meshdir = getattr(spec, "meshdir", None) if spec is not None else None
    spec_texturedir = getattr(spec, "texturedir", None) if spec is not None else None

    desired_meshdir = meshdir_override if meshdir_override is not None else spec_meshdir
    desired_texturedir = texturedir_override if texturedir_override is not None else spec_texturedir

    def _combine_with_root(root: Path, value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return None
        try:
            value_path = Path(value)
        except Exception:
            return str(root / Path(str(value)))
        if value_path.is_absolute():
            return str(value_path)
        return str(root / value_path)

    def _normalize_for_output(root: Path, value: Optional[str]) -> str:
        if value is None or value == "":
            return ""
        value_str = str(value)
        try:
            value_path = Path(value_str)
        except Exception:
            value_path = None

        if value_path is not None and value_path.is_absolute():
            try:
                relative_path = value_path.relative_to(root)
                return relative_path.as_posix()
            except ValueError:
                pass
        elif value_path is not None and not value_path.is_absolute():
            return value_path.as_posix()

        normalized_value = value_str.replace("\\", "/")
        normalized_root = str(root).replace("\\", "/")
        if not normalized_root.endswith("/"):
            normalized_root = f"{normalized_root}/"
        if normalized_value.lower().startswith(normalized_root.lower()):
            return normalized_value[len(normalized_root):]
        return normalized_value

    meshdir_for_compile = _combine_with_root(root_path, desired_meshdir)
    texturedir_for_compile = _combine_with_root(root_path, desired_texturedir)

    normalized_meshdir = ""
    normalized_texturedir = ""

    previous_meshdir = getattr(save_spec, "meshdir", None)
    previous_texturedir = getattr(save_spec, "texturedir", None)

    tendon_path_fallback: Dict[int, List[Dict[str, Any]]] = {}
    if spec is not None and model is not None:
        try:
            tendon_path_fallback = extract_tendon_path_segments(spec, model)
        except Exception:
            logger.warning("Failed to extract tendon path fallback from existing model", exc_info=True)
            tendon_path_fallback = {}

    try:
        if meshdir_for_compile is not None:
            save_spec.meshdir = meshdir_for_compile
        elif previous_meshdir:
            save_spec.meshdir = previous_meshdir

        if texturedir_for_compile is not None:
            save_spec.texturedir = texturedir_for_compile
        elif previous_texturedir:
            save_spec.texturedir = previous_texturedir

        _clear_spec(save_spec)
        _rebuild_assets(save_spec, model_data)
        _rebuild_bodies_and_children(save_spec, model_data)
        _rebuild_other_collections(
            save_spec,
            model_data,
            tendon_path_fallback=tendon_path_fallback,
        )
        
        current_comment = (getattr(save_spec, "comment", "") or "").strip()
        if COMPILE_COMMENT_TAG.lower() not in current_comment.lower():
            save_spec.comment = (
                f"{current_comment}\n{COMPILE_COMMENT_TAG}" if current_comment else COMPILE_COMMENT_TAG
            )
        save_spec.compile()
    finally:
        normalized_meshdir = _normalize_for_output(root_path, meshdir_for_compile if meshdir_for_compile is not None else desired_meshdir or previous_meshdir)
        normalized_texturedir = _normalize_for_output(root_path, texturedir_for_compile if texturedir_for_compile is not None else desired_texturedir or previous_texturedir)
        save_spec.meshdir = normalized_meshdir
        save_spec.texturedir = normalized_texturedir
    try:
        save_spec.to_file(str(target_path))
    except Exception as exc:
        logger.error("Failed to write MJCF model to %s", str(target_path), exc_info=True)
        raise RuntimeError(f"Failed to write MJCF model to {str(target_path)}") from exc
    try:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        tree = ET.parse(str(target_path), parser=parser)
        root = tree.getroot()
        compiler_elem = root.find("compiler")

        if compiler_elem is not None:
            def _set_or_clear(attr: str, value: str) -> None:
                if value:
                    normalized = value.replace("\\", "/")
                    if not normalized.endswith("/"):
                        normalized = f"{normalized}/"
                    compiler_elem.set(attr, normalized)
                else:
                    compiler_elem.attrib.pop(attr, None)

            _set_or_clear("meshdir", normalized_meshdir)
            _set_or_clear("texturedir", normalized_texturedir)

            tree.write(str(target_path), encoding="utf-8", xml_declaration=True)
    except Exception:
        logger.warning("Failed to normalize asset directories in output MJCF", exc_info=True)
    
    print("Finished compiling MJCF model to %s" % str(target_path))

    return {
        "target_file": str(target_path),
    }