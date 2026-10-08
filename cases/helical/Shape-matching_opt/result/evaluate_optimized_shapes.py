"""Evaluate optimized helical-actuator shapes against the target.

Simulates reconstructed models and visualizes shape matching.
"""

import pybullet as p
import time
import pybullet_data
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from pathlib import Path
from functools import lru_cache

THIS_DIR = Path(__file__).resolve().parent                # .../Shape-matching_opt/result
SHAPE_MATCH_DIR = THIS_DIR.parent                         # .../Shape-matching_opt
HELICAL_DIR = SHAPE_MATCH_DIR.parent                      # .../Helical

MESH_DIR   = HELICAL_DIR / "mesh"                         # .../Helical/mesh
MODEL_DIR  = THIS_DIR / "opt_models"                      # .../result/opt_models
SCALER_DIR = THIS_DIR / "opt_scalers"                     # .../result/opt_scalers



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


# Design parameters (optimized)
X = 1.0  # Scale factor
best_module_numbers = [6, 4, 6, 6]
best_pressure = 11.81267466010964  # kPa
# predicted by behavior model
# RMSE ≈ 4.51 mm
# Max error ≈ 8.46 mm
best_design = {'inner_radius': 0.003, 'segments': [{'avg_radius': 0.006648806983725312, 'module_length': 0.004370341289439773, 'module_number': 6, 'delta_phi': 0.0}, {'avg_radius': 0.00567600797921504, 'module_length': 0.004824045651245557, 'module_number': 4, 'delta_phi': 0.3150601441322955}, {'avg_radius': 0.0062232346178090715, 'module_length': 0.005202891296021214, 'module_number': 6, 'delta_phi': 0.7066654666818416}, {'avg_radius': 0.00586713347965703, 'module_length': 0.004495281143807268, 'module_number': 6, 'delta_phi': 1.1389483943166965}], 'wall_thickness': 0.0015, 'material_density': 1150.0, 'base_z': 2.0, 'end_cylinder': {'length': 0.006, 'density': 1150.0}}
# predicted by its own small model:
# Shape matching RMSE (mm):  5.0516468352134005
# Shape matching Max Error (mm):  8.187711184229078

design = {
    "inner_radius": best_design["inner_radius"],         # m
    "wall_thickness": 1.5e-3,        # m
    "material_density": 1150.0,      # kg/m^3 
    "base_z": 2.0,                   # base
    "segments": [
        {
            "avg_radius": best_design["segments"][0]["avg_radius"],
            "module_length": best_design["segments"][0]["module_length"],
            "module_number": best_design["segments"][0]["module_number"],
            "delta_phi": 0.0,       
        },
        {
            "avg_radius": best_design["segments"][1]["avg_radius"],
            "module_length": best_design["segments"][1]["module_length"],
            "module_number": best_design["segments"][1]["module_number"],
            "delta_phi": best_design["segments"][1]["delta_phi"],
        },
        {
            "avg_radius": best_design["segments"][2]["avg_radius"],
            "module_length": best_design["segments"][2]["module_length"],
            "module_number": best_design["segments"][2]["module_number"],
            "delta_phi": best_design["segments"][2]["delta_phi"],
        },
        {
            "avg_radius": best_design["segments"][3]["avg_radius"],
            "module_length": best_design["segments"][3]["module_length"],
            "module_number": best_design["segments"][3]["module_number"],
            "delta_phi": best_design["segments"][3]["delta_phi"],
        },
    ],
    "end_cylinder": {
        "radius": best_design["inner_radius"],  # same as the inner radius
        "length": 6e-3,
        "density": 1150.0,
    },
}

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
    end_radius = end_cylinder["radius"] * X
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
    

    # === linkMasses ===
    linkMasses = []
    for i, n_mod in enumerate(module_numbers):
        linkMasses += [module_masses[i]] * n_mod
    linkMasses.append(end_mass)

    # === mesh scaling ===
    mesh_path = str(MESH_DIR / "module.STL") 

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
                fileName=mesh_path,
                meshScale=[
                    X * 1e-3 * R_seg / S_out_radius,
                    X * 1e-3 * R_seg / S_out_radius,
                    X * 1e-3 * L_seg / S_module_length,
                ],
                physicsClientId=physicsClient,
            )
            vis_id = p.createVisualShape(
                p.GEOM_MESH,
                fileName=mesh_path,
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
        flags = p.URDF_USE_SELF_COLLISION,
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

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, ".."))


@lru_cache(maxsize=1)
def _load_scalers():
    scalers = {
        "y": (
            np.load(SCALER_DIR / "x_mean_y.npy"),
            np.load(SCALER_DIR / "x_std_y.npy"),
            np.load(SCALER_DIR / "y_mean_y.npy"),
            np.load(SCALER_DIR / "y_std_y.npy"),
        ),
        "z": (
            np.load(SCALER_DIR / "x_mean_z.npy"),
            np.load(SCALER_DIR / "x_std_z.npy"),
            np.load(SCALER_DIR / "y_mean_z.npy"),
            np.load(SCALER_DIR / "y_std_z.npy"),
        ),
        "x": (
            np.load(SCALER_DIR / "x_mean_x.npy"),
            np.load(SCALER_DIR / "x_std_x.npy"),
            np.load(SCALER_DIR / "y_mean_x.npy"),
            np.load(SCALER_DIR / "y_std_x.npy"),
        ),
    }
    return scalers

@lru_cache(maxsize=256)
def _get_interpreter(axis: str, tag: str):
    model_path = MODEL_DIR / f"small_model_{axis}_{tag}.tflite"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing TFLite model: {model_path}")
    interp = tflite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()
    out_det = interp.get_output_details()
    return interp, in_det, out_det

def torque_model(inner_radius, average_radius, module_length, pressure, half_joint_angle):
    R = average_radius * 1e3 / X
    L = module_length * 1e3 / X
    tag = f"R{R:.1f}_l{L:.1f}"

    scalers = _load_scalers()

    def infer_axis(axis: str, u: float):
        x_mean, x_std, y_mean, y_std = scalers[axis]
        interp, in_det, out_det = _get_interpreter(axis, tag)

        X_in = np.array([[pressure, u]], dtype=np.float32)
        X_scaled = (X_in - x_mean) / x_std
        interp.set_tensor(in_det[0]["index"], X_scaled.astype(np.float32))
        interp.invoke()
        y_scaled = interp.get_tensor(out_det[0]["index"])
        y = -(y_scaled * y_std + y_mean)   
        return float(y.item())

    ty = infer_axis("y", half_joint_angle[0])
    tz = infer_axis("z", half_joint_angle[1])
    tx = infer_axis("x", half_joint_angle[2])
    return np.array([ty, tz, tx])


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

    return actuator_position, arclengths, s

def sample_target_helix_by_arclength(a, h, s_sim):
    """
    a, h: helix parameters (meters)
    s_sim: simulated arclength parameters (num_links+1,), arbitrary total length
    """
    k = np.sqrt(a**2 + h**2)
    S_target = k * 2 * np.pi

    s_sim_norm = s_sim / s_sim[-1]
    s_target = s_sim_norm * S_target

    t = s_target / k
    x = -a * np.cos(t)
    y = -a * np.sin(t)
    z = h * t/(2 * np.pi)

    return np.stack([x, y, z], axis=1)  # (num_links+1, 3)

def shape_matching_loss_true_arclength(
    design,
    a=0.018,
    h=0.060
):
    actuator_position, arclengths, s_sim = run_simulation(design, X=X, use_gui=False, maximum_pressure=best_pressure)
    
    target_points = sample_target_helix_by_arclength(a, h, s_sim)
    # inverse the target point order
    target_points = target_points[::-1]
    # offset the actuator position to start from the same base position
    actuator_position -= actuator_position[0]
    target_points -= target_points[0]

    diffs = actuator_position - target_points
    sq_err = np.sum(diffs**2, axis=1)
    loss = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))
    
    rmse = np.sqrt(np.mean(sq_err))
    max_err = np.max(np.sqrt(sq_err))
    return actuator_position, target_points, rmse, max_err

# main： run simulation and plot final shape
if __name__ == "__main__":
    actuator_position, target_points, rmse, max_err = shape_matching_loss_true_arclength(
        design,
        a=0.018,
        h=0.060
    )
    print("Shape matching RMSE (mm): ", rmse*1000)
    print("Shape matching Max Error (mm): ", max_err*1000)
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
        color='black',
        label='Target Helix'
    )
    segment_colors = ['#F86D45', '#5BB1D5', '#45A085', '#83BB51']

    # first point use the same color as target, so start from index 1
    start_idx = 1
    for i, n_mod in enumerate(best_module_numbers):
        end_idx = start_idx + n_mod
        ax.plot(
            actuator_position[start_idx:end_idx, 0] * 1000 / X,
            actuator_position[start_idx:end_idx, 1] * 1000 / X,
            actuator_position[start_idx:end_idx, 2] * 1000 / X,
            marker='o',
            linestyle='--',
            color=segment_colors[i],
            label=f'Simulated Seg {i+1}'
        )
        start_idx = end_idx
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_xticks(np.arange(-40, 41, 20))
    ax.set_yticks(np.arange(-40, 41, 20))
    ax.set_zticks(np.arange(-60, 1, 20))
    ax.view_init(elev=11, azim=-50)
    plt.title('Comparison of Target and Simulated Shapes Using CMA-ES Optimized Design')
    plt.grid()
    plt.axis('equal')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.5, 1)) # position the legend very left outside the plot
    plt.show()

    p.disconnect()
