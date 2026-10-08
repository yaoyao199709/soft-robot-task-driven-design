"""Optimize helical-actuator designs for a target spatial shape.

Uses surrogate meta-models, PyBullet and CMA-ES.
"""

import pybullet as p
import time
import pybullet_data
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import cma
import time
import pandas as pd



original_path = os.path.dirname(__file__)
os.chdir(original_path)

def import_tensorflow():
    # Filter tensorflow version warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    return tf

tf = import_tensorflow()

from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.losses import MeanSquaredError  # type: ignore
from tensorflow.keras.saving import register_keras_serializable  # type: ignore

from joblib import load
import tensorflow.lite as tflite
import pandas as pd

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path

# Paths (match your folder tree)
THIS_DIR = Path(__file__).resolve().parent                  # .../Helical/Shape-matching_opt/optimization
SM_DIR   = THIS_DIR.parent                                  # .../Helical/Shape-matching_opt
HELICAL_DIR = SM_DIR.parent                                 # .../Helical
MESH_DIR = HELICAL_DIR / "mesh"                             # .../Helical/mesh

def _load_axis_model(axis: str):
    """
    Expected layout (per your screenshot):
      .../Helical/Shape-matching_opt/optimization/models_{axis}/behavior_model_{axis}.tflite
      .../Helical/Shape-matching_opt/optimization/models_{axis}/x_mean_{axis}.npy, x_std_{axis}.npy, y_mean_{axis}.npy, y_std_{axis}.npy
    """
    model_dir = THIS_DIR / f"models_{axis}"
    tflite_path = model_dir / f"behavior_model_{axis}.tflite"

    needed = [
        tflite_path,
        model_dir / f"x_mean_{axis}.npy",
        model_dir / f"x_std_{axis}.npy",
        model_dir / f"y_mean_{axis}.npy",
        model_dir / f"y_std_{axis}.npy",
    ]
    for fp in needed:
        if not fp.exists():
            raise FileNotFoundError(f"Missing file: {fp}")

    interpreter = tflite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    x_mean = np.load(str(model_dir / f"x_mean_{axis}.npy"))
    x_std  = np.load(str(model_dir / f"x_std_{axis}.npy"))
    y_mean = np.load(str(model_dir / f"y_mean_{axis}.npy"))
    y_std  = np.load(str(model_dir / f"y_std_{axis}.npy"))

    return interpreter, input_details, output_details, x_mean, x_std, y_mean, y_std

# Load models for y, z, x
y_interpreter, y_input_details, y_output_details, y_x_mean, y_x_std, y_y_mean, y_y_std = _load_axis_model("y")
z_interpreter, z_input_details, z_output_details, z_x_mean, z_x_std, z_y_mean, z_y_std = _load_axis_model("z")
x_interpreter, x_input_details, x_output_details, x_x_mean, x_x_std, x_y_mean, x_y_std = _load_axis_model("x")

# Mesh path (in Helical/mesh)
mesh_path = MESH_DIR / "module.STL"
if not mesh_path.exists():
    mesh_path = MESH_DIR / "module.stl"
if not mesh_path.exists():
    raise FileNotFoundError(f"Missing mesh file: {mesh_path}")
mesh_path_str = str(mesh_path)




X = 1.0  # Scale factor


# Quaternion helper functions
def quaternion_difference(q1, q2):
    """Calculate the difference between two quaternions."""
    return p.getDifferenceQuaternion(q1, q2)


def quaternion_to_euler(q):
    """Convert quaternion to Euler angles."""
    return p.getEulerFromQuaternion(q)


def wrap_angle(angle):
    """Force angle into [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def quaternion_rotation_batch(q, axes):
    """
    Vectorized computation of rotation angle around given axes for a batch of quaternions.
    q:     shape (N, 4)
    axes:  shape (N, 3)
    Returns: theta_axis, shape (N,)
    """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    axes = np.array(axes)
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    theta = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    sin_half_theta = np.sqrt(np.clip(1 - w**2, 0.0, None))
    sin_half_theta = np.clip(sin_half_theta, 1e-6, None)
    rotation_axis = np.stack([x, y, z], axis=1) / sin_half_theta[:, None]
    dot_product = np.einsum('ij,ij->i', rotation_axis, axes)
    theta_axis = theta * dot_product
    return wrap_angle(theta_axis)

def compute_module_mass(inner_radius, average_radius, module_length, wall_thickness, density):
    """
    input unit: m
    output unit: kg
    Corresponds to the formula you provided, but unified to SI units, no longer using 1e-9.
    """
    flank = 2 * average_radius - 2 * inner_radius - module_length / 2

    V_bellow_1 = 2 * math.pi * wall_thickness * module_length / 4 * (
        (inner_radius + module_length / 4) * math.pi / 2 - module_length / 4
    )

    V_bellow_2 = 2 * math.pi * wall_thickness * (
        (inner_radius + module_length / 4) * flank + flank * flank / 2
    )

    V_bellow_3 = 2 * math.pi * wall_thickness * module_length / 4 * (
        (inner_radius + module_length / 4 + flank) * math.pi / 2 + module_length / 4
    )

    V_bellow = 2 * (V_bellow_1 + V_bellow_2 + V_bellow_3)

    V_constraint = math.pi / 4 * module_length * (
        flank / 2 + module_length / 4 + inner_radius - wall_thickness
    ) ** 2

    V_total = V_bellow + V_constraint  # m^3
    mass = density * V_total  # kg
    return mass

def compute_segment_phis(segments):
    """Compute cumulative Phi list based on each segment's delta_phi."""
    phis = []
    current = 0.0
    for seg in segments:
        phis.append(current)
        current += seg["delta_phi"]
    return phis

# build_soft_robot(design, ...)：create PyBullet multibody
def build_soft_robot(design, physicsClient, X=1.0):
    inner_radius = design["inner_radius"] * X
    wall_thickness = design["wall_thickness"] * X
    density = design["material_density"]
    segments = design["segments"]
    end_cylinder = design["end_cylinder"]
    base_z = design["base_z"]

    N = len(segments)
    phi_list = compute_segment_phis(segments)

    # === base ===
    base_length = segments[0]["module_length"] * X
    baseMass = 0  # static
    baseCollisionShapeIndex = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=15 / 2 / 1000 * X,
        height=base_length,
        physicsClientId=physicsClient,
    )
    baseVisualShapeIndex = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=15 / 2 / 1000 * X,
        length=base_length,
        rgbaColor=[0.0, 0.5, 1.0, 1.0],
        physicsClientId=physicsClient,
    )

    # === end mass ===

    # use inner radius + wall thickness/2 as end radius
    end_radius = inner_radius + wall_thickness / 2
    end_length = end_cylinder["length"] * X
    end_density = end_cylinder["density"]
    end_mass = math.pi * end_radius**2 * end_length * end_density

    endCollisionShapeIndex = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=end_radius,
        height=end_length,
        physicsClientId=physicsClient,
    )
    endVisualShapeIndex = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=end_radius,
        length=end_length,
        rgbaColor=[0.0, 0.5, 1.0, 1.0],
        physicsClientId=physicsClient,
    )

    # === for each module ===
    module_numbers = []
    module_lengths_seg = []
    module_radii_seg = []
    module_masses = []

    for seg in segments:
        avg_radius = seg["avg_radius"] * X
        module_length = seg["module_length"] * X
        module_number = seg["module_number"]

        module_radius = avg_radius * 2 - inner_radius

        mass = compute_module_mass(
            inner_radius=inner_radius,
            average_radius=avg_radius,
            module_length=module_length,
            wall_thickness=wall_thickness,
            density=density,
        )

        module_numbers.append(module_number)
        module_lengths_seg.append(module_length)
        module_radii_seg.append(module_radius)
        module_masses.append(mass)

    num_links = sum(module_numbers)
    

    # === 生成 linkMasses ===
    linkMasses = []
    for i, n_mod in enumerate(module_numbers):
        linkMasses += [module_masses[i]] * n_mod
    linkMasses.append(end_mass)

    # === mesh scaling ===
    S_out_radius = (10 * 2 - 5) * 1e-3 * X
    S_module_length = 10 * 1e-3 * X

    linkCollisionShapeIndices = []
    linkVisualShapeIndices = []

    link_to_segment = []
    for seg_idx, seg in enumerate(segments):
        link_to_segment += [seg_idx] * seg["module_number"]
        
        R_seg = module_radii_seg[seg_idx]
        L_seg = module_lengths_seg[seg_idx]
        for _ in range(seg["module_number"]):
            col_id = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=mesh_path_str,
                meshScale=[
                    X * 1e-3 * R_seg / S_out_radius,
                    X * 1e-3 * R_seg / S_out_radius,
                    X * 1e-3 * L_seg / S_module_length,
                ],
                physicsClientId=physicsClient,
            )
            vis_id = p.createVisualShape(
                p.GEOM_MESH,
                fileName=mesh_path_str,
                meshScale=[
                    X * 1e-3 * R_seg / S_out_radius,
                    X * 1e-3 * R_seg / S_out_radius,
                    X * 1e-3 * L_seg / S_module_length,
                ],
                rgbaColor=[1.0, 1.0, 0.941, 1.0],
                physicsClientId=physicsClient,
            )
            linkCollisionShapeIndices.append(col_id)
            linkVisualShapeIndices.append(vis_id)

    linkCollisionShapeIndices.append(endCollisionShapeIndex)
    linkVisualShapeIndices.append(endVisualShapeIndex)


    Δy = inner_radius
    # linkPositions length = num_links + 1
    linkPositions = [[0, -Δy, module_lengths_seg[0]]]
    for i in range(len(segments)):
        if i == 0:
            linkPositions += [
                [0, 0, module_lengths_seg[i]] for _ in range(1, module_numbers[i])
            ]
        else:
            linkPositions.append(
                [math.sin(segments[i]["delta_phi"]) * Δy, -math.cos(segments[i]["delta_phi"]) * Δy + Δy, module_lengths_seg[i]]
            )
            linkPositions += [
                [0, 0, module_lengths_seg[i]] for _ in range(1, module_numbers[i])
            ]
    # end link
    linkPositions.append([0, Δy, (module_lengths_seg[-1] + end_length) / 2])

    # Orientations
    linkOrientations = []
    for i in range(len(segments)):
        if i > 0:
            eulerangle = [0, 0, segments[i]["delta_phi"]]
        else:
            eulerangle = [0, 0, segments[i]["delta_phi"]]
        q0 = p.getQuaternionFromEuler(eulerangle)
        quats = [q0] + [p.getQuaternionFromEuler([0, 0, 0]) for _ in range(module_numbers[i] - 1)]
        linkOrientations.extend(quats)

    linkOrientations.append(p.getQuaternionFromEuler([0, 0, 0]))
    linkOrientations[0] = p.getQuaternionFromEuler([0, 0, 0])

    linkInertialFramePositions = [[0, 0, 0] for _ in range(num_links + 1)]
    linkInertialFrameOrientations = [[0, 0, 0, 1] for _ in range(num_links + 1)]

    linkParentIndices = list(range(num_links + 1))
    linkJointTypes = [p.JOINT_SPHERICAL for _ in range(num_links)] + [p.JOINT_FIXED]
    linkJointAxis = [[1, 0, 0] for _ in range(num_links + 1)]

    baseStartPos = [0, 0, base_z]
    baseStartOri = p.getQuaternionFromEuler(
        [0, np.pi / 2 + np.radians(90 - 62), np.pi / 2]
    )

    robotUniqueId = p.createMultiBody(
        baseMass,
        baseCollisionShapeIndex,
        baseVisualShapeIndex,
        basePosition=baseStartPos,
        baseOrientation=baseStartOri,
        baseInertialFramePosition=[0, 0, 0],
        linkMasses=linkMasses,
        linkCollisionShapeIndices=linkCollisionShapeIndices,
        linkVisualShapeIndices=linkVisualShapeIndices,
        linkPositions=linkPositions,
        linkOrientations=linkOrientations,
        linkInertialFramePositions=linkInertialFramePositions,
        linkInertialFrameOrientations=linkInertialFrameOrientations,
        linkParentIndices=linkParentIndices,
        linkJointTypes=linkJointTypes,
        linkJointAxis=linkJointAxis,
        physicsClientId=physicsClient,
    )

    meta = {
        "num_links": num_links,
        "link_to_segment": link_to_segment,
        "base_pos": baseStartPos,
        "base_ori": baseStartOri,
        "Δy": Δy,
        "phi_list": phi_list,
        "module_numbers": module_numbers,
    }

    return robotUniqueId, meta

def torque_model(inner_radius, average_radius, module_length, pressure, half_joint_angle):
    
    # scale back the design parameters
    inner_radius = inner_radius * 1e3 / X  # m to mm
    average_radius = average_radius * 1e3 / X  # m to mm
    module_length = module_length * 1e3 / X  # m to mm

    
    y_X_input = np.array([[average_radius, module_length, pressure, half_joint_angle[0]]])
    y_X_scaled = (y_X_input - y_x_mean) / y_x_std
    y_interpreter.set_tensor(y_input_details[0]['index'], y_X_scaled.astype(np.float32))
    y_interpreter.invoke()
    y_Y_scaled = y_interpreter.get_tensor(y_output_details[0]['index'])
    y_Y = - (y_Y_scaled * y_y_std + y_y_mean)
        
    
    z_X_input = np.array([[average_radius, module_length, pressure, half_joint_angle[1]]])
    z_X_scaled = (z_X_input - z_x_mean) / z_x_std
    z_interpreter.set_tensor(z_input_details[0]['index'], z_X_scaled.astype(np.float32))
    z_interpreter.invoke()
    z_Y_scaled = z_interpreter.get_tensor(z_output_details[0]['index'])
    z_Y = - (z_Y_scaled * z_y_std + z_y_mean)

    
    x_X_input = np.array([[average_radius, module_length, pressure, half_joint_angle[2]]])
    x_X_scaled = (x_X_input - x_x_mean) / x_x_std
    x_interpreter.set_tensor(x_input_details[0]['index'], x_X_scaled.astype(np.float32))
    x_interpreter.invoke()
    x_Y_scaled = x_interpreter.get_tensor(x_output_details[0]['index'])
    x_Y = - (x_Y_scaled * x_y_std + x_y_mean)
    
    return np.array([y_Y.item(), z_Y.item(), x_Y.item()])

# run_simulation
def run_simulation(design, X=1.0, use_gui=True, maximum_pressure = 15.0):
    physicsClient = p.connect(p.GUI if use_gui else p.DIRECT)
    p.setGravity(0, 0, -9.8 * X)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(0)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.2,
        cameraYaw=0,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 1.9],
    )

    robotUniqueId, meta = build_soft_robot(design, physicsClient, X)
    num_links = meta["num_links"]
    Δy = meta["Δy"]
    base_pos = np.array(meta["base_pos"])

    # damping
    damping_force = [1e-4 * X * X] * 3
    p.setJointMotorControlMultiDofArray(
        bodyUniqueId=robotUniqueId,
        jointIndices=list(range(num_links)),
        controlMode=p.POSITION_CONTROL,
        targetPositions=[[0, 0, 0, 1] for _ in range(num_links)],
        targetVelocities=[[0, 0, 0] for _ in range(num_links)],
        forces=[damping_force for _ in range(num_links)],
        positionGains=[0 for _ in range(num_links)],
        velocityGains=[1 for _ in range(num_links)],
        physicsClientId=physicsClient,
    )

    time_step = 1.0 / 240.0 / 10.0
    p.setTimeStep(time_step)

    omega = 1.0 * math.pi * 2

    pressure_fns = [
        lambda t: 0.0,
        lambda t: maximum_pressure * (0.5 * np.sin(omega * t - math.pi / 2) + 0.5),
        lambda t: maximum_pressure,
    ]

    n_steps = int(2 * math.pi / omega / time_step * 2)
    sim_time = 0.0

    cycle_time = 2 * math.pi / omega
    settle_time = 0.5 * cycle_time
    actuation_time = cycle_time

    
    joint_angle = np.zeros((num_links, 3, n_steps))
    
    joint_torque = np.zeros((num_links, 3, n_steps))

    time_plot = np.zeros((n_steps,))
    actuator_position = np.zeros((num_links + 1, 3))
    actuator_position[0] = base_pos
    arclengths = np.zeros((num_links,))

    
    for _ in range(10):
        p.stepSimulation()

    for k in range(n_steps):
        # 1. Get joint quaternions
        spherical_states = p.getJointStatesMultiDof(robotUniqueId, list(range(num_links)))
        quats = np.array([state[0] for state in spherical_states])

        main_axes = np.tile(np.array([1, 0, 0]), (num_links, 1))
        second_axes = np.tile(np.array([0, 1, 0]), (num_links, 1))
        twist_axes = np.tile(np.array([0, 0, 1]), (num_links, 1))

        
        joint_angle[:, 0, k] = quaternion_rotation_batch(quats, main_axes)
        joint_angle[:, 1, k] = quaternion_rotation_batch(quats, second_axes)
        joint_angle[:, 2, k] = quaternion_rotation_batch(quats, twist_axes)

        # 2. Pressure profile
        if sim_time <= settle_time:
            pressure = pressure_fns[0](sim_time)
        elif sim_time <= actuation_time:
            pressure = pressure_fns[1](sim_time - settle_time)
        else:
            pressure = pressure_fns[2](sim_time)

        # 3. Torques from NN surrogate
        
        for i in range(num_links):
            seg_idx = meta["link_to_segment"][i]
            
            τ = torque_model(
                inner_radius=design["inner_radius"] * X,
                average_radius=design["segments"][seg_idx]["avg_radius"] * X,
                module_length=design["segments"][seg_idx]["module_length"] * X,
                pressure=pressure,
                half_joint_angle=joint_angle[i, :, k] / 2
            )
            
            joint_torque[i, :, k] = τ

        p.setJointMotorControlMultiDofArray(
            bodyUniqueId=robotUniqueId,
            jointIndices=list(range(num_links)),
            controlMode=p.TORQUE_CONTROL,
            forces=joint_torque[:, :, k].tolist(),
        )

        # 4. Integrate physics
        p.stepSimulation()

        # 5. Read link positions & apply Δy correction
        actuator_position[0] = base_pos
        for i in range(num_links):
            link_pos, link_ori = p.getLinkState(robotUniqueId, i)[:2]
            rot_matrix = np.array(p.getMatrixFromQuaternion(link_ori)).reshape(3, 3)
            local_y = rot_matrix[:, 1]
            corrected_pos = np.array(link_pos) + local_y * Δy
            actuator_position[i + 1] = corrected_pos
            
            seg_idx = meta["link_to_segment"][i]
            l0 = design["segments"][seg_idx]["module_length"] * X
            theta_total = np.linalg.norm(joint_angle[i, :2, k], axis=0)
            arclengths[i] = l0 + theta_total * Δy
            
        s = np.concatenate([[0.0], np.cumsum(arclengths)]) #arc parameter

        time_plot[k] = sim_time
        sim_time += time_step
    p.disconnect()
    return actuator_position, arclengths, s

def sample_target_helix_by_arclength(a, h, s_sim):
    """
    Helix:
        x = -a cos t
        y = -a sin t
        z = h * t / (2π)
        t ∈ [0, 2π]
    a, h: meters
    s_sim: (num_links+1,) simulated true arclength
    """

    
    k = np.sqrt(a**2 + (h / (2*np.pi))**2)
    S_target = k * 2 * np.pi

    
    s_sim_norm = s_sim / s_sim[-1]
    s_target = s_sim_norm * S_target

    
    t = s_target / k   

    x = -a * np.cos(t)
    y = -a * np.sin(t)
    z =  h * t / (2 * np.pi)

    return np.stack([x, y, z], axis=1)

# Shape matching loss
def shape_matching_loss_true_arclength(
    design,
    pressure_kPa,
    a=0.018,   # 18 mm
    h=0.060    # 60 mm
):
    actuator_position, arclengths, s_sim = run_simulation(
        design,
        X=X,
        use_gui=False,
        maximum_pressure=pressure_kPa
    )

    target_points = sample_target_helix_by_arclength(a, h, s_sim)
    target_points = target_points[::-1]  # reverse order if needed

    # offset both curves to start from the same base position
    actuator_position -= actuator_position[0]
    target_points -= target_points[0]

    diffs = actuator_position - target_points
    sq_err = np.sum(diffs**2, axis=1)

    mse = np.mean(sq_err)
    max_err = np.max(np.sqrt(sq_err))

    return actuator_position, target_points, mse, max_err

# Design encoding/decoding
BASE_DESIGN_TEMPLATE = {
    "wall_thickness": 1.5e-3,
    "material_density": 1150.0,
    "base_z": 2.0,
    "end_cylinder": {
        "length": 6e-3,
        "density": 1150.0,
        # radius is derived as inner_radius + wall_thickness/2
    }
}



def denorm(x, lo, hi):
    return lo + x * (hi - lo)

def decode_theta_with_pressure_normalized(theta_norm, module_numbers):
    
    N = len(module_numbers)

    
    r_in = 3e-3   

    
    p_n = theta_norm[0]
    pressure = denorm(p_n, 10.0, 15.0)

    idx = 1
    segments = []

    for i in range(N):
        R_n = theta_norm[idx]
        L_n = theta_norm[idx + 1]
        phi_n = theta_norm[idx + 2]
        idx += 3

        
        R = denorm(R_n, 5e-3, 7e-3)   # ✅ 5–7 mm
        L = denorm(L_n, 4e-3, 6e-3)   # ✅ 4–6 mm

        if i == 0:
            phi = 0.0                
        else:
            phi = denorm(phi_n, 0.0, np.pi/2)

        
        if R - r_in < L / 4:
            return None, None

        segments.append({
            "avg_radius": R,
            "module_length": L,
            "module_number": module_numbers[i],
            "delta_phi": phi,
        })

    design = {
        "inner_radius": r_in,  
        "segments": segments,
        **BASE_DESIGN_TEMPLATE
    }

    return design, pressure




def init_log(log_path):
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write(
                "iter,pressure_kPa,module_numbers,"
                "inner_radius_mm,avg_radius_mm_list,module_length_mm_list,delta_phi_deg_list,"
                "rmse_mm,max_err_mm,sim_time_s,total_time_s\n"
            )
def log_attempt(
    log_path,
    iter_id,
    pressure,
    module_numbers,
    design,
    mse,
    max_err,
    sim_time,
    total_time,
):
    avg_radius = [seg["avg_radius"] * 1e3 / X for seg in design["segments"]]
    module_len = [seg["module_length"] * 1e3 / X for seg in design["segments"]]
    phi_list = [seg["delta_phi"] * 180 / np.pi for seg in design["segments"]]

    with open(log_path, "a") as f:
        f.write(
            f"{iter_id},"
            f"{pressure:.4f},"
            f"\"{module_numbers}\","
            f"{design['inner_radius'] * 1e3 / X:.4f},"
            f"\"{avg_radius}\","
            f"\"{module_len}\","
            f"\"{phi_list}\","
            f"{np.sqrt(mse) * 1000:.4f},"
            f"{max_err * 1000:.4f},"
            f"{sim_time:.4f},"
            f"{total_time:.4f}\n"
        )

# Target length & structure candidates
def compute_target_helix_length(a=0.018, h=0.060):
    k = np.sqrt(a**2 + (h / (2*np.pi))**2)
    return 2 * np.pi * k


def infer_total_module_range(S_target,
                             L_min=4e-3,
                             L_max=6e-3,
                             dy_min=2e-3,
                             dy_max=4e-3,
                             theta_min = np.pi/8,
                             theta_max=np.pi/4):
    """
    use physical limits to estimate reasonable range of total module number.
    """
    s_min = L_min + theta_min * dy_min
    s_max = L_max + theta_max * dy_max

    N_min = int(np.floor(S_target / s_max))
    N_max = int(np.ceil(S_target / s_min))
    print(f"Inferred total module number range: [{N_min}, {N_max}]")
    return max(N_min, 4), max(N_max, 4)

def partition_total_into_4(total, max_diff=2):
    """
    separate total into 4 parts, with max-min <= max_diff.
    """
    candidates = []
    for a in range(1, total-2):
        for b in range(1, total-a-1):
            for c in range(1, total-a-b):
                d = total - a - b - c
                if d < 1:
                    continue
                if max(a, b, c, d) - min(a, b, c, d) <= max_diff:
                    candidates.append([a, b, c, d])
    return candidates



def generate_module_number_structures(
    a=0.018, 
    h=0.060,
    num_totals=5,      
    sigma_ratio=0.25  
):
    S_target = compute_target_helix_length(a=a, h=h)
    N_min, N_max = infer_total_module_range(S_target)


    N_center = int(np.ceil(0.5 * (N_min + N_max)))
    

    totals = [
        N_center - 2,
        N_center - 1,
        N_center,
        N_center + 1,
        N_center + 2,
    ]


    totals = [
        int(T) for T in totals
        if T >= N_min and T <= N_max
    ]


    

    print(f"[Structure Sampling]")
    print(f"  N_min={N_min}, N_max={N_max}, N_center={N_center}")
    print(f"  Sampled totals = {totals}")


    all_structs = []
    for T in totals:
        structs_T = partition_total_into_4(T, max_diff=2)
        all_structs.extend(structs_T)


    unique_structs = []
    seen = set()
    for s in all_structs:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_structs.append(s)

    print(f"  Total candidate structures = {len(unique_structs)}")

    return unique_structs




# CMA-ES Optimization
def optimize_geometry_cmaes(
    module_numbers,
    max_evals=60,
    max_allow_error=5e-3,   # 5 mm
    pressure_min=10.0,
    pressure_max=15.0,
    a=0.018,
    h=0.060
):
    N = len(module_numbers)


    eval_counter = {"i": 0}


    def fitness(theta_norm):
        t0 = time.time()

        design, pressure = decode_theta_with_pressure_normalized(
            theta_norm, module_numbers
        )

        if design is None:
            return 1e6


        if pressure < 10 or pressure > 15:
            return 1e6
        
        actuator_position, target_points, mse, max_err = shape_matching_loss_true_arclength(
            design,
            pressure_kPa=pressure,
            a=a,
            h=h
        )
        # 3D plot of final shape
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot target helix
        ax.plot(
            target_points[:, 0] * 1000 / X,
            target_points[:, 1] * 1000 / X,
            target_points[:, 2] * 1000 / X,
            marker='o',
            linestyle='-',
            label='Target Helix'
        )
        # Plot simulated helix
        ax.plot(
            actuator_position[:, 0] * 1000 / X,
            actuator_position[:, 1] * 1000 / X,
            actuator_position[:, 2] * 1000 / X,
            marker='o',
            linestyle='--',
            label='Simulated Helix'
        )
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        # add the design parameters and pressure info and error metrics to the title
        avg_radius = []
        module_len = []
        phi_list = []
        for seg in design["segments"]:
            avg_radius.append(seg["avg_radius"]*1e3/X)
            module_len.append(seg["module_length"]*1e3/X)
            phi_list.append(seg["delta_phi"]*180/np.pi)
            
        plt.title(f'Helix Shape Matching\nPressure: {pressure:.2f} kPa, \n'
                  f'Inner radius: {design["inner_radius"]*1e3/X:.2f} mm, \n'
                  f'Avg Radii: {[f"{r:.2f}" for r in avg_radius]} mm, \n'
                  f'Module Lengths: {[f"{l:.2f}" for l in module_len]} mm, \n'
                  f'Delta Phis: {[f"{phi:.1f}" for phi in phi_list]} deg\n'
                  f'RMSE: {np.sqrt(mse)*1000:.2f} mm, Max Error: {max_err*1000:.2f} mm')
        # make the title won't be cut off
        plt.tight_layout()
        plt.grid()
        plt.axis('equal')
        plt.legend()


        
        plt.ion()   

        plt.show()
        plt.pause(2.0)   
        plt.close(fig)


        rmse = np.sqrt(mse)

        


        penalty = 0.0
        if rmse > max_allow_error:
            penalty += ((rmse - max_allow_error) / max_allow_error) ** 2
        if max_err > max_allow_error:
            penalty += ((max_err - max_allow_error) / max_allow_error) ** 2

        fitness_value = mse + penalty


        
        sim_time = time.time() - t0
        total_time = sim_time   

        eval_counter["i"] += 1

        log_attempt(
            log_path=str(THIS_DIR / "optimization_log.csv"),
            iter_id=eval_counter["i"],
            pressure=pressure,
            module_numbers=module_numbers,
            design=design,
            mse=mse,
            max_err=max_err,
            sim_time=sim_time,
            total_time=total_time,
        )


        return fitness_value


    


    dim = 1 + 3 * N

    x0 = np.full(dim, 0.5)
    lower = np.zeros(dim)
    upper = np.ones(dim)


    es = cma.CMAEvolutionStrategy(
        x0,
        0.3,   
        {
            "bounds": [lower, upper],
            "maxfevals": max_evals,
            "verb_disp": 0,
        }
    )


    es.optimize(fitness)
    best_theta = es.result.xbest
    # best_design, best_pressure = decode_theta_with_pressure(best_theta, module_numbers)
    best_design, best_pressure = decode_theta_with_pressure_normalized(
        best_theta, module_numbers
    )

    _, _, best_mse, best_max_err = shape_matching_loss_true_arclength(
        best_design,
        pressure_kPa=best_pressure,
        a=a,
        h=h
    )

    return best_design, best_pressure, best_mse, best_max_err





def full_design_optimization(
    a=0.018,
    h=0.060,
    max_evals_per_cma=60
):
    init_log("optimization_log2.csv")
    structures = generate_module_number_structures(a=a, h=h)

    global_best = {
        "design": None,
        "pressure": None,
        "module_numbers": None,
        "mse": np.inf,
        "max_err": np.inf,
    }

    print(f"Candidate structures (module_numbers): {structures}")
    

    for module_numbers in structures:
        print(f"\n=== Structure {module_numbers} ===")

        design_s, p_s, mse_s, max_err_s = optimize_geometry_cmaes(
            module_numbers=module_numbers,
            max_evals=max_evals_per_cma,
            max_allow_error=5e-3,
            a=a,
            h=h
        )
        
        print(f"  RMSE ≈ {np.sqrt(mse_s)*1000:.2f} mm, "
              f"max ≈ {max_err_s*1000:.2f} mm @ p={p_s:.2f} kPa")

        if mse_s < global_best["mse"]:
            global_best.update({
                "design": design_s,
                "pressure": p_s,
                "module_numbers": module_numbers,
                "mse": mse_s,
                "max_err": max_err_s,
            })

    print("\n========== GLOBAL BEST ==========")
    if global_best["design"] is None:
        print("❌ No feasible design found.")
    else:
        print(f"Best module_numbers: {global_best['module_numbers']}")
        print(f"Best pressure: {global_best['pressure']} kPa")
        print(f"RMSE ≈ {np.sqrt(global_best['mse'])*1000:.2f} mm")
        print(f"Max error ≈ {global_best['max_err']*1000:.2f} mm")

    return global_best


# main
if __name__ == "__main__":
    result = full_design_optimization(
        a=0.018,
        h=0.060,
        max_evals_per_cma=30
    )

    best_design = result["design"]
    best_pressure = result["pressure"]
    best_structure = result["module_numbers"]

    print("Best design:", best_design)
    print("Best pressure:", best_pressure, "kPa")
    print("Best structure (module_numbers):", best_structure)