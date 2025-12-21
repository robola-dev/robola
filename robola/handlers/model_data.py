"""
Model Data Packing - Pack MuJoCo model data into JSON format

Extract all element information from MjSpec/MjModel/MjData and convert it into
a dictionary format usable by the frontend.
"""
from __future__ import annotations

import json
import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

_FLOAT_TOL = 1e-6

ACTUATOR_SHORTCUTS: Tuple[str, ...] = (
    "general",
    "motor",
    "position",
    "velocity",
    "intvelocity",
    "damper",
    "cylinder",
    "muscle",
    "adhesion",
)

ACTUATOR_SHORTCUT_TAGS = set(ACTUATOR_SHORTCUTS)

ACTUATOR_SHORTCUT_TO_TAG: Dict[str, str] = {
    shortcut: shortcut for shortcut in ACTUATOR_SHORTCUTS
}

_ENUM_STRING_CACHE: Dict[type, Dict[str, Any]] = {}


def _get_enum_lookup(enum_cls: type) -> Dict[str, Any]:
    cached = _ENUM_STRING_CACHE.get(enum_cls)
    if cached is not None:
        return cached

    lookup: Dict[str, Any] = {}
    for member in enum_cls:
        name = member.name.lower()
        lookup[name] = member
        if name.startswith("mj"):
            lookup[name[2:]] = member
        lookup[name.split("_", 1)[-1]] = member
    _ENUM_STRING_CACHE[enum_cls] = lookup
    return lookup


def _coerce_enum_value(value: Any, enum_cls: type) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (int, np.integer)):
        try:
            return enum_cls(int(value))
        except ValueError:
            return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        lookup = _get_enum_lookup(enum_cls)
        return lookup.get(normalized)
    return None


def _to_float_list(values: Optional[Iterable[Any]], *, length: Optional[int] = None) -> List[float]:
    if values is None:
        result: List[float] = []
    elif isinstance(values, np.ndarray):
        result = [float(x) for x in values.tolist()]
    else:
        result = []
        for item in values:
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                result.append(0.0)
    if length is not None:
        if len(result) < length:
            result = result + [0.0] * (length - len(result))
        elif len(result) > length:
            result = result[:length]
    return result


def _to_int_flag(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_actuator_shortcut_from_params(
    *,
    gaintype: Any = None,
    biastype: Any = None,
    dyntype: Any = None,
    trntype: Any = None,
    biasprm: Optional[Iterable[Any]] = None,
    gainprm: Optional[Iterable[Any]] = None,
    dynprm: Optional[Iterable[Any]] = None,
    ctrllimited: Any = None,
    actlimited: Any = None,
    forcelimited: Any = None,
    target: Any = None,
) -> str:
    gain_enum = _coerce_enum_value(gaintype, mujoco.mjtGain)
    bias_enum = _coerce_enum_value(biastype, mujoco.mjtBias)
    dyn_enum = _coerce_enum_value(dyntype, mujoco.mjtDyn)
    trn_enum = _coerce_enum_value(trntype, mujoco.mjtTrn)

    bias_params = _to_float_list(biasprm, length=3)
    dyn_params = _to_float_list(dynprm, length=3)
    gain_params = _to_float_list(gainprm, length=3)

    # Muscle actuators
    if (
        gain_enum == mujoco.mjtGain.mjGAIN_MUSCLE
        or bias_enum == mujoco.mjtBias.mjBIAS_MUSCLE
        or dyn_enum == mujoco.mjtDyn.mjDYN_MUSCLE
    ):
        return "muscle"
    
    # Adhesion 
    if (
        gain_enum == mujoco.mjtGain.mjGAIN_FIXED
        and bias_enum == mujoco.mjtBias.mjBIAS_NONE
        and dyn_enum == mujoco.mjtDyn.mjDYN_NONE
        and trn_enum == mujoco.mjtTrn.mjTRN_BODY
    ):
        gainprm_list = gain_params[1:3]
        dynprm_list = dyn_params[:3]
        biasprm_list = bias_params[:3]

        if ctrllimited and gainprm_list == [0.0, 0.0] and dynprm_list == [1.0, 0.0, 0.0] and biasprm_list == [0.0, 0.0, 0.0]:
            return "adhesion"   
        
    # motor
    if gain_enum == mujoco.mjtGain.mjGAIN_FIXED and bias_enum == mujoco.mjtBias.mjBIAS_NONE and dyn_enum == mujoco.mjtDyn.mjDYN_NONE:
        gainprm_list = gain_params[:3]
        dynprm_list = dyn_params[:3]
        biasprm_list = bias_params[:3]
        if gainprm_list == [1.0, 0.0, 0.0] and dynprm_list == [1.0, 0.0, 0.0] and biasprm_list == [0.0, 0.0, 0.0]:
            return "motor"
        
    # Cylinder
    if gain_enum == mujoco.mjtGain.mjGAIN_FIXED and bias_enum == mujoco.mjtBias.mjBIAS_AFFINE and dyn_enum == mujoco.mjtDyn.mjDYN_FILTER:
        gainprm_list = gain_params[1:3]
        dynprm_list = dyn_params[1:3]
        if gainprm_list == [0.0, 0.0] and dynprm_list == [0.0, 0.0]:
            return "cylinder"

    # Damper
    if gain_enum == mujoco.mjtGain.mjGAIN_AFFINE and bias_enum == mujoco.mjtBias.mjBIAS_NONE and dyn_enum == mujoco.mjtDyn.mjDYN_NONE:
        gainprm_list = gain_params[:2]
        dynprm_list = dyn_params[:3]
        biasprm_list = bias_params[:3]
        if ctrllimited and gainprm_list == [0.0, 0.0] and dynprm_list == [1.0, 0.0, 0.0] and biasprm_list == [0.0, 0.0, 0.0]:
            return "damper" 

    # intvelocity
    if gain_enum == mujoco.mjtGain.mjGAIN_FIXED and bias_enum == mujoco.mjtBias.mjBIAS_AFFINE and dyn_enum == mujoco.mjtDyn.mjDYN_INTEGRATOR:
        gainprm_list = gain_params[1:3]
        dynprm_list = dyn_params[:3]
        biasprm_list = bias_params[:1]
        if actlimited and gainprm_list == [ 0.0, 0.0] and dynprm_list == [1.0, 0.0, 0.0] and biasprm_list == [0.0] and gain_params[0] == -bias_params[1]:
            return "intvelocity"

    # velocity
    if gain_enum == mujoco.mjtGain.mjGAIN_FIXED and bias_enum == mujoco.mjtBias.mjBIAS_AFFINE and dyn_enum == mujoco.mjtDyn.mjDYN_NONE:
        gainprm_list = gain_params[1:3]
        dynprm_list = dyn_params[:3]
        biasprm_list = bias_params[:2]
        if gainprm_list == [0.0, 0.0] and dynprm_list == [1.0, 0.0, 0.0] and biasprm_list == [0.0, 0.0] and gain_params[0] == -bias_params[2]:
            return "velocity"
        
    # position
    if gain_enum == mujoco.mjtGain.mjGAIN_FIXED and bias_enum == mujoco.mjtBias.mjBIAS_AFFINE and (dyn_enum == mujoco.mjtDyn.mjDYN_NONE or dyn_enum == mujoco.mjtDyn.mjDYN_FILTEREXACT):
        gainprm_list = gain_params[1:3]
        dynprm_list = dyn_params[1:3]
        biasprm_list = bias_params[:1]
        if gainprm_list == [0.0, 0.0] and dynprm_list == [0.0, 0.0] and biasprm_list == [0.0] and gain_params[0] == -bias_params[1]:
            return "position"    

    return "general"


def infer_actuator_shortcut_from_spec(actuator: mujoco.MjsActuator) -> str:
    return infer_actuator_shortcut_from_params(
        gaintype=actuator.gaintype,
        biastype=actuator.biastype,
        dyntype=actuator.dyntype,
        trntype=actuator.trntype,
        biasprm=actuator.biasprm,
        gainprm=actuator.gainprm,
        dynprm=actuator.dynprm,
        ctrllimited=actuator.ctrllimited,
        actlimited=actuator.actlimited,
        forcelimited=actuator.forcelimited,
        target=getattr(actuator, "target", None),
    )


def infer_actuator_shortcut_from_dict(data: Dict[str, Any]) -> str:
    return infer_actuator_shortcut_from_params(
        gaintype=data.get("gaintype"),
        biastype=data.get("biastype"),
        dyntype=data.get("dyntype"),
        trntype=data.get("trntype"),
        biasprm=data.get("biasprm"),
        gainprm=data.get("gainprm"),
        dynprm=data.get("dynprm"),
        ctrllimited=data.get("ctrllimited"),
        actlimited=data.get("actlimited"),
        forcelimited=data.get("forcelimited"),
        target=data.get("target"),
    )


def _char_vec_to_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        length = len(value)
    except TypeError:
        return str(value)
    chars = []
    for idx in range(length):
        char = value[idx]
        if char == "\x00":
            break
        chars.append(char)
    return "".join(chars)


def _serialize_mj_value(value: Any, *, _visited: set[int] | None = None) -> Any:
    if _visited is None:
        _visited = set()

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, str)):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, np.generic):
        scalar = value.item()
        return _serialize_mj_value(scalar, _visited=_visited)

    if isinstance(value, np.ndarray):
        return _serialize_mj_value(value.tolist(), _visited=_visited)

    if isinstance(value, (list, tuple)):
        return [_serialize_mj_value(item, _visited=_visited) for item in value]

    value_type = type(value)
    if value_type.__module__.startswith("mujoco."):
        if value_type.__name__ == "MjCharVec":
            return _char_vec_to_str(value)

        obj_id = id(value)
        if obj_id in _visited:
            return None
        _visited.add(obj_id)

        result: dict[str, Any] = {}
        for attr in dir(value):
            if attr.startswith("_"):
                continue
            try:
                attr_value = getattr(value, attr)
            except Exception:
                continue
            if callable(attr_value):
                continue
            result[attr] = _serialize_mj_value(attr_value, _visited=_visited)
        return result

    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            return _serialize_mj_value(converted, _visited=_visited)
        except Exception:
            pass

    return value


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    try:
        return list(value)
    except TypeError:
        return [value]


def quat2BabylonEuler(quat, degrees=True, invert=True):
    """MuJoCo quat [w,x,y,z] -> Euler (roll,pitch,yaw), using scipy, xyz order."""
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,):
        raise ValueError("quat must be length-4")
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely determine all angles.",
        )
        rot = R.from_quat(q_xyzw)
        euler = rot.as_euler('xyz', degrees=degrees)
    if invert:
        euler = -euler
        euler[1], euler[2] = euler[2], euler[1]
    if degrees:
        euler = (euler + 180) % 360 - 180
    return euler.tolist()


def orientation2Quat(quatRaw, orientation):
    """将 mujoco._specs.MjsOrientation 中的方向转换为四元数。"""
    otype = orientation.type
    quat = np.zeros(4, dtype=np.float64)
    
    if otype == mujoco.mjtOrientation.mjORIENTATION_QUAT:
        quat[:] = quatRaw
    elif otype == mujoco.mjtOrientation.mjORIENTATION_AXISANGLE:
        axis = orientation.axisangle[:3]
        angle = orientation.axisangle[3]
        mujoco.mju_axisAngle2Quat(quat, axis, angle)
    elif otype == mujoco.mjtOrientation.mjORIENTATION_EULER:
        mujoco.mju_euler2Quat(quat, orientation.euler, "xyz")
    elif otype == mujoco.mjtOrientation.mjORIENTATION_XYAXES:
        vec_x = np.array(orientation.xyaxes[:3], dtype=np.float64)
        mujoco.mju_normalize3(vec_x)
        vec_y = np.array(orientation.xyaxes[3:], dtype=np.float64)
        dot_prod = np.dot(vec_x, vec_y)
        vec_y = vec_y - dot_prod * vec_x
        mujoco.mju_normalize3(vec_y)
        vec_z = np.zeros(3)
        mujoco.mju_cross(vec_z, vec_x, vec_y)
        mat = np.zeros(9)
        mat[0] = vec_x[0]; mat[1] = vec_y[0]; mat[2] = vec_z[0]
        mat[3] = vec_x[1]; mat[4] = vec_y[1]; mat[5] = vec_z[1]
        mat[6] = vec_x[2]; mat[7] = vec_y[2]; mat[8] = vec_z[2]
        mujoco.mju_mat2Quat(quat, mat)
    elif otype == mujoco.mjtOrientation.mjORIENTATION_ZAXIS:
        target_z = np.array(orientation.zaxis, dtype=np.float64)
        mujoco.mju_normalize3(target_z)
        ref_z = np.array([0.0, 0.0, 1.0])
        axis = np.zeros(3)
        mujoco.mju_cross(axis, ref_z, target_z)
        dot = np.dot(ref_z, target_z)
        if dot < -0.999999:
            axis = np.array([1.0, 0.0, 0.0])
            angle = np.pi
        else:
            angle = np.arccos(dot)
        mujoco.mju_axisAngle2Quat(quat, axis, angle)

    return quat.tolist()


def pack_model_data(spec: mujoco.MjSpec, model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """将Mujoco模型和数据打包为字典格式"""
    model_dict = {
        "bodies": pack_bodies_data(spec, model, data),
        "joints": pack_joints_data(spec, model, data),
        "geoms": pack_geoms_data(spec, model, data),
        "sites": pack_site_data(spec, model, data),
        "cameras": pack_camera_data(spec, model, data),
        "lights": pack_lights_data(spec, model, data),
        "meshes": pack_mesh_data(spec),
        "textures": pack_textures_data(spec),
        "materials": pack_materials_data(spec),
        "pairs": pack_pairs_data(spec),
        "excludes": pack_exclude_data(spec),
        "equalities": pack_equalities_data(spec),
        "tendons": pack_tendons_data(spec, model),
        "actuators": pack_actuators_data(spec),
        "sensors": pack_sensors_data(spec),
        "keys": pack_keys_data(spec),
        "compiler": _serialize_mj_value(spec.compiler),
        "option": _serialize_mj_value(spec.option),
        "stat": _serialize_mj_value(spec.stat),
        "visual": _serialize_mj_value(getattr(spec, "visual", None)),
        
    }
    return model_dict


def mujoco2WebPos(position):
    """Convert a mujoco position to a web-compatible format."""
    return [position[0], position[2], position[1]]


def mujoco2WebQuat(rotation):
    """Convert MuJoCo [w,x,y,z] quaternion to BabylonJS [x,y,z,w] (left-handed)."""
    quat = np.asarray(rotation, dtype=float)
    if quat.shape != (4,):
        raise ValueError("rotation must be a length-4 sequence [w, x, y, z]")

    rh_quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
    rh_rot = R.from_quat(rh_quat_xyzw)

    # Swap MuJoCo Y/Z axes to mirror into Babylon's left-handed basis.
    axis_swap = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    lh_matrix = axis_swap @ rh_rot.as_matrix() @ axis_swap.T
    lh_quat_xyzw = R.from_matrix(lh_matrix).as_quat()
    return lh_quat_xyzw.tolist()


def standardize_name(itemname, orig_name, id):
    """Standardize element name"""
    if not orig_name:
        return f"{itemname}_{id}_"
    elif orig_name.startswith(f"{itemname}_") and orig_name.endswith("_"):
        return f"{itemname}_{id}_"
    else:
        return orig_name


def pack_bodies_data(spec: mujoco.MjSpec, model: mujoco.MjModel, data: mujoco.MjData) -> list:
    bodies = []
    for i in range(len(spec.bodies)):
        body_dict = {
            "id": i,
            "parent_id": int(model.body_parentid[i]),
            "name": standardize_name("body", spec.bodies[i].name, i),
            "pos": mujoco2WebPos(spec.bodies[i].pos.tolist()),
            "euler": quat2BabylonEuler(orientation2Quat(spec.bodies[i].quat, spec.bodies[i].alt)),
            "mass": float(model.body_mass[i]),
            "ipos": mujoco2WebPos(model.body_ipos[i].tolist()),
            "ieuler": quat2BabylonEuler(orientation2Quat(spec.bodies[i].iquat, spec.bodies[i].ialt)),
            "inertia": mujoco2WebPos(spec.bodies[i].inertia.tolist()),
            "mocap": bool(spec.bodies[i].mocap),
            "gravcomp": float(spec.bodies[i].gravcomp),
            "userdata": list(spec.bodies[i].userdata),
            "explicitinertial": bool(spec.bodies[i].explicitinertial),
        }
        bodies.append(body_dict)
    return bodies


def pack_joints_data(spec: mujoco.MjSpec, model: mujoco.MjModel, data: mujoco.MjData) -> list:
    joints = []
    for i in range(len(spec.joints)):
        joint_dict = {
            "id": i,
            "name": standardize_name("joint", spec.joints[i].name, i),
            "body_id": int(model.jnt_bodyid[i]),
            "type": spec.joints[i].type.value,
            "pos": mujoco2WebPos(spec.joints[i].pos.tolist()),
            "axis": mujoco2WebPos(spec.joints[i].axis.tolist()),
            "ref": spec.joints[i].ref,
            "stiffness": spec.joints[i].stiffness,
            "springref": spec.joints[i].springref,
            "springdamper": spec.joints[i].springdamper.tolist(),
            "limited": spec.joints[i].limited,
            "range": spec.joints[i].range.tolist(),
            "margin": spec.joints[i].margin,
            "solref_limit": spec.joints[i].solref_limit.tolist(),
            "solimp_limit": spec.joints[i].solimp_limit.tolist(),
            "actfrclimited": spec.joints[i].actfrclimited,
            "actfrcrange": spec.joints[i].actfrcrange.tolist(),
            "armature": spec.joints[i].armature,
            "damping": spec.joints[i].damping,
            "frictionloss": spec.joints[i].frictionloss,
            "solref_friction": spec.joints[i].solref_friction.tolist(),
            "solimp_friction": spec.joints[i].solimp_friction.tolist(),
            "group": spec.joints[i].group,
            "actgravcomp": bool(spec.joints[i].actgravcomp),
            "userdata": list(spec.joints[i].userdata),
        }
        joints.append(joint_dict)
    return joints


def pack_geoms_data(spec: mujoco.MjSpec, model: mujoco.MjModel, data: mujoco.MjData) -> list:
    geoms = []
    for i in range(len(spec.geoms)):
        type = spec.geoms[i].type.value
        size = spec.geoms[i].size.tolist()
        if type == 4 or type == 6:
            size = [size[0], size[2], size[1]]

        geom_dict = {
            "id": i,
            "name": standardize_name("geom", spec.geoms[i].name, i),
            "body_id": int(model.geom_bodyid[i]),
            "type": spec.geoms[i].type.value,
            "pos": mujoco2WebPos(spec.geoms[i].pos.tolist()),
            "euler": quat2BabylonEuler(orientation2Quat(spec.geoms[i].quat, spec.geoms[i].alt)),
            "enableFromTo": False if spec.geoms[i].fromto is not None and np.isnan(spec.geoms[i].fromto[0]) else bool(spec.geoms[i].fromto is not None and spec.geoms[i].fromto[0] is not None),
            "fromto": (
                (
                    lambda arr: [arr[0], arr[2], arr[1], arr[3], arr[5], arr[4]]
                    if len(arr) == 6 else arr
                )([None if np.isnan(x) else x for x in spec.geoms[i].fromto.tolist()])
                if spec.geoms[i].fromto is not None and (
                    False if np.isnan(spec.geoms[i].fromto[0]) else bool(spec.geoms[i].fromto[0] is not None)
                )
                else ([None if np.isnan(x) else x for x in spec.geoms[i].fromto.tolist()] if spec.geoms[i].fromto is not None else [])
            ),
            "size": size,
            "contype": spec.geoms[i].contype,
            "conaffinity": spec.geoms[i].conaffinity,
            "condim": spec.geoms[i].condim,
            "priority": spec.geoms[i].priority,
            "friction": spec.geoms[i].friction.tolist() if spec.geoms[i].friction is not None else [1, 0.005, 0.0001],
            "solmix": float(spec.geoms[i].solmix),
            "solref": spec.geoms[i].solref.tolist(),
            "solimp": spec.geoms[i].solimp.tolist(),
            "margin": float(spec.geoms[i].margin),
            "gap": float(spec.geoms[i].gap),
            "mass": spec.geoms[i].mass if np.isnan(spec.geoms[i].mass) == False else None,
            "density": spec.geoms[i].density,
            "typeinertia": spec.geoms[i].typeinertia.value,
            "fluid_ellipsoid": bool(spec.geoms[i].fluid_ellipsoid),
            "fluid_coefs": spec.geoms[i].fluid_coefs.tolist(),
            "material": spec.geoms[i].material,
            "rgba": spec.geoms[i].rgba.tolist(),
            "group": spec.geoms[i].group,
            "hfieldname": spec.geoms[i].hfieldname if spec.geoms[i].hfieldname else "",
            "meshname": spec.geoms[i].meshname if spec.geoms[i].meshname else "",
            "fitscale": spec.geoms[i].fitscale,
            "userdata": list(spec.geoms[i].userdata),
        }
        geoms.append(geom_dict)
    return geoms


def pack_site_data(spec: mujoco.MjSpec, model: mujoco.MjModel, data: mujoco.MjData) -> list:
    sites = []
    for i in range(len(spec.sites)):
        type = spec.sites[i].type.value
        size = spec.sites[i].size.tolist()
        if type == 4 or type == 6:
            size = [size[0], size[2], size[1]]
        site_dict = {
            "id": i,
            "name": standardize_name("site", spec.sites[i].name, i),
            "body_id": int(model.site_bodyid[i]),
            "pos": mujoco2WebPos(spec.sites[i].pos.tolist()),
            "euler": quat2BabylonEuler(orientation2Quat(spec.sites[i].quat, spec.sites[i].alt)),
            "enableFromTo": False if spec.sites[i].fromto is not None and np.isnan(spec.sites[i].fromto[0]) else bool(spec.sites[i].fromto is not None and spec.sites[i].fromto[0] is not None),
            "fromto": (
                (
                    lambda arr: [arr[0], arr[2], arr[1], arr[3], arr[5], arr[4]]
                    if len(arr) == 6 else arr
                )([None if np.isnan(x) else x for x in spec.sites[i].fromto.tolist()])
                if spec.sites[i].fromto is not None and (
                    False if np.isnan(spec.sites[i].fromto[0]) else bool(spec.sites[i].fromto is not None and spec.sites[i].fromto[0] is not None)
                )
                else ([None if np.isnan(x) else x for x in spec.sites[i].fromto.tolist()] if spec.sites[i].fromto is not None else [])
            ),
            "size": size,
            "type": spec.sites[i].type.value,
            "material": spec.sites[i].material,
            "group": spec.sites[i].group,
            "rgba": spec.sites[i].rgba.tolist(),
            "userdata": list(spec.sites[i].userdata),
        }
        sites.append(site_dict)
    return sites


def pack_camera_data(spec: mujoco.MjSpec, model: mujoco.MjModel, data: mujoco.MjData) -> list:
    cameras = []
    for i in range(len(spec.cameras)):
        camera_dict = {
            "id": i,
            "name": standardize_name("camera", spec.cameras[i].name, i),
            "body_id": int(model.cam_bodyid[i]),
            "pos": mujoco2WebPos(spec.cameras[i].pos.tolist()),
            "euler": quat2BabylonEuler(orientation2Quat(spec.cameras[i].quat, spec.cameras[i].alt)),
            "mode": spec.cameras[i].mode.value,
            "targetbody": spec.cameras[i].targetbody,
            "orthographic": spec.cameras[i].orthographic,
            "fovy": spec.cameras[i].fovy,
            "ipd": spec.cameras[i].ipd,
            "intrinsic": spec.cameras[i].intrinsic.tolist(),
            "sensor_size": spec.cameras[i].sensor_size.tolist(),
            "resolution": spec.cameras[i].resolution.tolist(),
            "focal_length": spec.cameras[i].focal_length.tolist(),
            "focal_pixel": spec.cameras[i].focal_pixel.tolist(),
            "principal_length": spec.cameras[i].principal_length.tolist(),
            "principal_pixel": spec.cameras[i].principal_pixel.tolist(),
            "userdata": list(spec.cameras[i].userdata),
        }
        cameras.append(camera_dict)
    return cameras


def pack_lights_data(spec: mujoco.MjSpec, model: mujoco.MjModel, data: mujoco.MjData) -> list:
    lights = []
    for i in range(len(spec.lights)):
        light_dict = {
            "id": i,
            "name": standardize_name("light", spec.lights[i].name, i),
            "body_id": int(model.light_bodyid[i]),
            "pos": mujoco2WebPos(spec.lights[i].pos.tolist()),
            "dir": mujoco2WebPos(spec.lights[i].dir.tolist()),
            "mode": spec.lights[i].mode.value,
            "targetbody": spec.lights[i].targetbody,
            "active": bool(spec.lights[i].active),
            "type": spec.lights[i].type.value,
            "texture": spec.lights[i].texture,
            "castshadow": bool(spec.lights[i].castshadow),
            "bulbradius": spec.lights[i].bulbradius,
            "intensity": spec.lights[i].intensity,
            "range": spec.lights[i].range,
            "attenuation": spec.lights[i].attenuation.tolist(),
            "cutoff": spec.lights[i].cutoff,
            "exponent": spec.lights[i].exponent,
            "ambient": spec.lights[i].ambient.tolist(),
            "diffuse": spec.lights[i].diffuse.tolist(),
            "specular": spec.lights[i].specular.tolist(),
        }
        lights.append(light_dict)
    return lights


def pack_mesh_data(spec: mujoco.MjSpec) -> list:
    meshes = []
    for i in range(len(spec.meshes)):
        mesh_dict = {
            "id": i,
            "name": standardize_name("mesh", spec.meshes[i].name, i),
            "file": spec.meshes[i].file,
            "refpos": mujoco2WebPos(spec.meshes[i].refpos.tolist()),
            "refeuler": quat2BabylonEuler(spec.meshes[i].refquat.tolist()),
            "scale": spec.meshes[i].scale.tolist(),
            "inertiatype": spec.meshes[i].inertia.value,
            "smoothnormal": bool(spec.meshes[i].smoothnormal),
            "needsdf": bool(spec.meshes[i].needsdf),
            "maxhullvert": spec.meshes[i].maxhullvert,
            "userface": _to_list(spec.meshes[i].userface),
            "userfacetexcoord": _to_list(spec.meshes[i].userfacetexcoord),
            "usernormal": _to_list(spec.meshes[i].usernormal),
            "usertexcoord": _to_list(spec.meshes[i].usertexcoord),
            "uservert": _to_list(spec.meshes[i].uservert),
        }
        meshes.append(mesh_dict)
    return meshes


def pack_textures_data(spec: mujoco.MjSpec) -> list:
    textures = []
    for i in range(len(spec.textures)):
        texture_dict = {
            "id": i,
            "name": standardize_name("texture", spec.textures[i].name, i),
            "type": spec.textures[i].type.value,
            "colorspace": spec.textures[i].colorspace.value,
            "builtin": spec.textures[i].builtin,
            "mark": spec.textures[i].mark,
            "rgb1": spec.textures[i].rgb1.tolist(),
            "rgb2": spec.textures[i].rgb2.tolist(),
            "markrgb": spec.textures[i].markrgb.tolist(),
            "random": spec.textures[i].random,
            "height": spec.textures[i].height,
            "width": spec.textures[i].width,
            "nchannel": spec.textures[i].nchannel,
            "content_type": spec.textures[i].content_type,
            "file": spec.textures[i].file,
            "gridsize": spec.textures[i].gridsize.tolist(),
            "gridlayout": list(spec.textures[i].gridlayout)[:-1],
            "cubefiles": list(spec.textures[i].cubefiles),
            "hflip": bool(spec.textures[i].hflip),
            "vflip": bool(spec.textures[i].vflip),
        }
        textures.append(texture_dict)
    return textures


def pack_materials_data(spec: mujoco.MjSpec) -> list:
    materials = []
    for i in range(len(spec.materials)):
        material_dict = {
            "id": i,
            "name": standardize_name("material", spec.materials[i].name, i),
            "texture": list(spec.materials[i].textures),
            "texuniform": bool(spec.materials[i].texuniform),
            "texrepeat": spec.materials[i].texrepeat.tolist(),
            "emission": spec.materials[i].emission,
            "specular": spec.materials[i].specular,
            "shininess": spec.materials[i].shininess,
            "reflectance": spec.materials[i].reflectance,
            "metallic": spec.materials[i].metallic,
            "roughness": spec.materials[i].roughness,
            "rgba": spec.materials[i].rgba.tolist(),
        }
        materials.append(material_dict)
    return materials


def pack_pairs_data(spec: mujoco.MjSpec) -> list:
    pairs = []
    for i in range(len(spec.pairs)):
        pair_dict = {
            "id": i,
            "name": standardize_name("pair", spec.pairs[i].name, i),
            "geomname1": spec.pairs[i].geomname1,
            "geomname2": spec.pairs[i].geomname2,
            "condim": spec.pairs[i].condim,
            "solref": spec.pairs[i].solref.tolist(),
            "solreffriction": spec.pairs[i].solreffriction.tolist(),
            "solimp": spec.pairs[i].solimp.tolist(),
            "margin": spec.pairs[i].margin,
            "gap": spec.pairs[i].gap,
            "friction": spec.pairs[i].friction.tolist(),
        }
        pairs.append(pair_dict)
    return pairs


def pack_exclude_data(spec: mujoco.MjSpec) -> list:
    excludes = []
    for i in range(len(spec.excludes)):
        exclude_dict = {
            "id": i,
            "name": standardize_name("exclude", spec.excludes[i].name, i),
            "bodyname1": spec.excludes[i].bodyname1,
            "bodyname2": spec.excludes[i].bodyname2,
        }
        excludes.append(exclude_dict)
    return excludes


def pack_equalities_data(spec: mujoco.MjSpec) -> list:
    equalities = []
    for i in range(len(spec.equalities)):
        type_val = spec.equalities[i].type.value
        anchor = None
        relpos = None
        releuler = None
        torquescale = None
        polycoef = None
        
        if type_val == mujoco.mjtEq.mjEQ_CONNECT.value:
            anchor = mujoco2WebPos(spec.equalities[i].data.tolist()[0:3])
            torquescale = spec.equalities[i].data.tolist()[-1]
        if type_val == mujoco.mjtEq.mjEQ_WELD.value:
            anchor = mujoco2WebPos(spec.equalities[i].data.tolist()[0:3])
            if not all(abs(x) < 1e-8 for x in spec.equalities[i].data.tolist()[6:10]):
                releuler = quat2BabylonEuler(spec.equalities[i].data.tolist()[6:10])
                relpos = mujoco2WebPos(spec.equalities[i].data.tolist()[3:6])
            torquescale = spec.equalities[i].data.tolist()[-1]
        if type_val == mujoco.mjtEq.mjEQ_JOINT.value or type_val == mujoco.mjtEq.mjEQ_TENDON.value:
            polycoef = spec.equalities[i].data.tolist()[0:5]
            
        equality_dict = {
            "id": i,
            "name": standardize_name("equality", spec.equalities[i].name, i),
            "type": type_val,
            "anchor": anchor,
            "relpos": relpos,
            "releuler": releuler,
            "torquescale": torquescale,
            "polycoef": polycoef,
            "active": bool(spec.equalities[i].active),
            "name1": spec.equalities[i].name1,
            "name2": spec.equalities[i].name2,
            "objtype": spec.equalities[i].objtype.value,
            "solref": spec.equalities[i].solref.tolist(),
            "solimp": spec.equalities[i].solimp.tolist(),
        }
        equalities.append(equality_dict)
    return equalities


_WRAP_TYPE_KIND = {
    mujoco.mjtWrap.mjWRAP_JOINT.value: "joint",
    mujoco.mjtWrap.mjWRAP_PULLEY.value: "pulley",
    mujoco.mjtWrap.mjWRAP_SITE.value: "site",
    mujoco.mjtWrap.mjWRAP_SPHERE.value: "geom",
    mujoco.mjtWrap.mjWRAP_CYLINDER.value: "geom",
}

_WRAP_TYPE_TO_OBJTYPE = {
    mujoco.mjtWrap.mjWRAP_JOINT.value: mujoco.mjtObj.mjOBJ_JOINT,
    mujoco.mjtWrap.mjWRAP_SITE.value: mujoco.mjtObj.mjOBJ_SITE,
    mujoco.mjtWrap.mjWRAP_SPHERE.value: mujoco.mjtObj.mjOBJ_GEOM,
    mujoco.mjtWrap.mjWRAP_CYLINDER.value: mujoco.mjtObj.mjOBJ_GEOM,
}


def extract_tendon_path_segments(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel | None,
) -> dict[int, list[dict[str, Any]]]:
    """Collect the path segments for each tendon."""
    if spec is None or model is None:
        return {}

    tendon_count = min(len(spec.tendons), int(model.ntendon))
    if tendon_count == 0:
        return {}

    wrap_type_arr = np.asarray(model.wrap_type, dtype=int)
    wrap_objid_arr = np.asarray(model.wrap_objid, dtype=int)
    wrap_prm_arr = np.asarray(model.wrap_prm, dtype=float)
    wrap_len = wrap_type_arr.shape[0]

    paths: dict[int, list[dict[str, Any]]] = {}
    for tendon_idx in range(tendon_count):
        adr = int(model.tendon_adr[tendon_idx])
        count = int(model.tendon_num[tendon_idx])
        if count <= 0:
            paths[tendon_idx] = []
            continue

        segments: list[dict[str, Any]] = []
        for offset in range(count):
            wrap_index = adr + offset
            if wrap_index < 0 or wrap_index >= wrap_len:
                continue

            wrap_type_val = int(wrap_type_arr[wrap_index])
            param_val = float(wrap_prm_arr[wrap_index])
            target_id = int(wrap_objid_arr[wrap_index])

            objtype_enum = _WRAP_TYPE_TO_OBJTYPE.get(wrap_type_val)
            if objtype_enum == mujoco.mjtObj.mjOBJ_GEOM:
                if param_val == -1.0:
                    param_val = ''
                else:
                    param_val = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, int(param_val))
                    
            target_name = None
            if objtype_enum is not None and target_id >= 0:
                try:
                    target_name = mujoco.mj_id2name(model, objtype_enum, target_id)
                except Exception:
                    target_name = None

            segment = {
                "wrap_type": wrap_type_val,
                "target": target_name,
                "objtype": int(objtype_enum) if objtype_enum is not None else None,
                "param": param_val,
            }
            segments.append(segment)

        paths[tendon_idx] = segments

    return paths


def pack_tendons_data(spec: mujoco.MjSpec, model: mujoco.MjModel | None) -> list:
    tendon_paths = extract_tendon_path_segments(spec, model)
    tendons = []
    for i in range(len(spec.tendons)):
        tendon_dict = {
            "id": i,
            "name": standardize_name("tendon", spec.tendons[i].name, i),
            "stiffness": spec.tendons[i].stiffness,
            "springlength": spec.tendons[i].springlength.tolist(),
            "damping": spec.tendons[i].damping,
            "frictionloss": spec.tendons[i].frictionloss,
            "solref_friction": spec.tendons[i].solref_friction.tolist(),
            "solimp_friction": spec.tendons[i].solimp_friction.tolist(),
            "armature": spec.tendons[i].armature,
            "limited": spec.tendons[i].limited,
            "actfrclimited": spec.tendons[i].actfrclimited,
            "range": spec.tendons[i].range.tolist(),
            "actfrcrange": spec.tendons[i].actfrcrange.tolist(),
            "margin": spec.tendons[i].margin,
            "solref_limit": spec.tendons[i].solref_limit.tolist(),
            "solimp_limit": spec.tendons[i].solimp_limit.tolist(),
            "material": spec.tendons[i].material,
            "width": spec.tendons[i].width,
            "rgba": spec.tendons[i].rgba.tolist(),
            "group": spec.tendons[i].group,
            "userdata": list(spec.tendons[i].userdata),
            "path": tendon_paths.get(i, []),
        }
        tendons.append(tendon_dict)
    return tendons


def pack_actuators_data(spec: mujoco.MjSpec) -> list:
    actuators = []
    for i in range(len(spec.actuators)):
        shortcut = infer_actuator_shortcut_from_spec(spec.actuators[i])
        bias_params = _to_float_list(spec.actuators[i].biasprm, length=3)
        dyn_params = _to_float_list(spec.actuators[i].dynprm, length=3)
        gain_params = _to_float_list(spec.actuators[i].gainprm, length=3)
        
        shortcut_position = None
        shortcut_velocity = None
        shortcut_intvelocity = None
        shortcut_damper = None
        shortcut_cylinder = None
        shortcut_muscle = None
        shortcut_adhesion = None

        if shortcut == "position":
            kp = gain_params[0]
            inheritrange = spec.actuators[i].inheritrange
            if spec.actuators[i].dyntype == mujoco.mjtDyn.mjDYN_FILTEREXACT:
                timeconst = dyn_params[0] 
            else:
                timeconst = 0.0
            if bias_params[2] <= 0:
                kv = -bias_params[2]
                dampratio = None
            else:
                dampratio = bias_params[2]
                kv = None
            shortcut_position = {
                "kp": kp,
                "kv": kv,
                "dampratio": dampratio,
                "timeconst": timeconst,
                "inheritrange": inheritrange,
            }
        if shortcut == "velocity":
            kv = gain_params[0]
            shortcut_velocity = {"kv": kv}
        if shortcut == "intvelocity":
            kp = gain_params[0]
            inheritrange = spec.actuators[i].inheritrange
            if bias_params[2] <= 0:
                kv = -bias_params[2]
                dampratio = None
            else:
                dampratio = bias_params[2]
                kv = None
            shortcut_intvelocity = {
                "kp": kp,
                "kv": kv,
                "dampratio": dampratio,
                "inheritrange": inheritrange,
            }
        if shortcut == "damper":
            kv = -gain_params[2]
            shortcut_damper = {"kv": kv}
        if shortcut == "cylinder":
            timeconst = dyn_params[0]
            area = gain_params[0]
            bias = bias_params[:3]
            shortcut_cylinder = {"timeconst": timeconst, "area": area, "bias": bias}
        if shortcut == "muscle":
            gain_params = _to_float_list(spec.actuators[i].gainprm, length=9)
            timeconst = dyn_params[:2]
            tausmooth = dyn_params[2]
            length_range = gain_params[:2]
            force, scale, lmin, lmax, vmax, fpmax, fvmax = gain_params[2:9]
            shortcut_muscle = {
                "timeconst": timeconst,
                "tausmooth": tausmooth,
                "range": length_range,
                "force": force,
                "scale": scale,
                "lmin": lmin,
                "lmax": lmax,
                "vmax": vmax,
                "fpmax": fpmax,
                "fvmax": fvmax,
            }
        if shortcut == "adhesion":
            gain = gain_params[0]
            shortcut_adhesion = {"gain": gain}

        actuator_dict = {
            "id": i,
            "name": standardize_name("actuator", spec.actuators[i].name, i),
            "gaintype": spec.actuators[i].gaintype.value,
            "gainprm": spec.actuators[i].gainprm.tolist(),
            "biastype": spec.actuators[i].biastype.value,
            "biasprm": spec.actuators[i].biasprm.tolist(),
            "dyntype": spec.actuators[i].dyntype.value,
            "dynprm": spec.actuators[i].dynprm.tolist(),
            "actdim": spec.actuators[i].actdim,
            "actearly": spec.actuators[i].actearly,
            "trntype": spec.actuators[i].trntype.value,
            "gear": spec.actuators[i].gear.tolist(),
            "target": spec.actuators[i].target,
            "refsite": spec.actuators[i].refsite,
            "slidersite": spec.actuators[i].slidersite,
            "cranklength": spec.actuators[i].cranklength,
            "lengthrange": spec.actuators[i].lengthrange.tolist(),
            "ctrllimited": spec.actuators[i].ctrllimited,
            "ctrlrange": spec.actuators[i].ctrlrange.tolist(),
            "forcelimited": spec.actuators[i].forcelimited,
            "forcerange": spec.actuators[i].forcerange.tolist(),
            "actlimited": spec.actuators[i].actlimited,
            "actrange": spec.actuators[i].actrange.tolist(),
            "group": spec.actuators[i].group,
            "shortcut": shortcut,
            "shortcut_position": shortcut_position,
            "shortcut_velocity": shortcut_velocity,
            "shortcut_intvelocity": shortcut_intvelocity,
            "shortcut_damper": shortcut_damper,
            "shortcut_cylinder": shortcut_cylinder,
            "shortcut_muscle": shortcut_muscle,
            "shortcut_adhesion": shortcut_adhesion,
            "userdata": list(spec.actuators[i].userdata),
        }
        actuators.append(actuator_dict)
    return actuators


def pack_sensors_data(spec: mujoco.MjSpec) -> list:
    sensors = []
    for i in range(len(spec.sensors)):
        sensor_dict = {
            "id": i,
            "name": standardize_name("sensor", spec.sensors[i].name, i),
            "type": spec.sensors[i].type.value,
            "objtype": spec.sensors[i].objtype.value,
            "objname": spec.sensors[i].objname,
            "reftype": spec.sensors[i].reftype.value,
            "refname": spec.sensors[i].refname,
            "datatype": spec.sensors[i].datatype.value,
            "needstage": spec.sensors[i].needstage.value,
            "dim": spec.sensors[i].dim,
            "cutoff": spec.sensors[i].cutoff,
            "noise": spec.sensors[i].noise,
            "userdata": list(spec.sensors[i].userdata),
        }
        sensors.append(sensor_dict)
    return sensors


def pack_keys_data(spec: mujoco.MjSpec) -> list:
    keys = []
    for i in range(len(spec.keys)):
        key_dict = {
            "id": i,
            "name": standardize_name("key", spec.keys[i].name, i),
            "time": spec.keys[i].time,
            "qpos": list(spec.keys[i].qpos),
            "qvel": list(spec.keys[i].qvel),
            "act": list(spec.keys[i].act),
            "mpos": list(spec.keys[i].mpos),
            "mquat": list(spec.keys[i].mquat),
            "ctrl": list(spec.keys[i].ctrl),
        }
        keys.append(key_dict)
    return keys
