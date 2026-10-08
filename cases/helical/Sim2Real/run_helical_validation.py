"""Simulate the helical actuator under gravity and specified tip loads.

Uses axis-wise surrogate models for task-level PyBullet validation.
"""

import pybullet as p
import time
import pybullet_data
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent          # Helical/Sim2Real
ROOT_DIR = THIS_DIR.parent                          # Helical
MESH_DIR = ROOT_DIR / "mesh"                        # Helical/mesh
SUR_DIR  = THIS_DIR / "surrogate_models"            # Helical/Sim2Real/surrogate_models


def import_tensorflow():
    # Filter tensorflow version warnings
    import os
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    return tf

tf = import_tensorflow()

from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from tensorflow.keras.saving import register_keras_serializable # type: ignore

from joblib import load
import tensorflow.lite as tflite
import pandas as pd




# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'

def quaternion_difference(q1, q2):
    """Calculate the difference between two quaternions."""
    q_diff = p.getDifferenceQuaternion(q1, q2)
    return p.getDifferenceQuaternion(q1, q2)

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles."""
    return p.getEulerFromQuaternion(q)

def quaternion_rotation_angle_around_axis(q, axis):
    """
    Computes the rotation angle of quaternion q around a given axis.
    
    Parameters:
    q : list or numpy array (x, y, z, w) - Quaternion
    axis : list or numpy array (x, y, z) - Desired rotation axis
    
    Returns:
    theta_axis : float - Rotation angle around the given axis (radians)
    """
    x, y, z, w = q
    
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)  # Ensure the axis is a unit vector

    theta = 2 * np.arccos(np.clip(w, -1.0, 1.0)) # This avoids domain errors when w ≈ 1.0 + ε

    sin_half_theta = np.sqrt(1 - w**2)

    if sin_half_theta < 1e-6:  # Avoid division by zero (quaternion is near identity)
        return 0.0

    # Compute rotation axis
    rotation_axis = np.array([x, y, z]) / sin_half_theta
    
    # Project onto desired axis
    dot_product = np.dot(rotation_axis, axis)
    
    # Compute the angle component around the desired axis
    theta_axis = theta * dot_product
    return theta_axis

# Force the angle to wrap into [-π, π]
def wrap_angle(angle):
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
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)  # Ensure the axes are unit vectors
    theta = 2 * np.arccos(w)  # Total rotation angle
    sin_half_theta = np.sqrt(1 - w**2)
    sin_half_theta = np.clip(sin_half_theta, 1e-6, None)  # Avoid division by zero
    # Compute rotation axis
    rotation_axis = np.stack([x, y, z], axis=1) / sin_half_theta[:, None]
    # Compute projection
    dot_product = np.einsum('ij,ij->i', rotation_axis, axes)
    theta_axis = theta * dot_product
    return wrap_angle(theta_axis)


X = 1  # X is the factor to scale up the system
# Setup simulation
physicsClient = p.connect(p.GUI)


p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(
        0
    )

p.resetDebugVisualizerCamera(
        cameraDistance=0.2, #26
        cameraYaw=0,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 1.9],
    )



# Load the TensorFlow Lite model for y1_M: Half
y_tflite_model_path_M_1 = str(SUR_DIR / "helix_y_stiffness_surrogate_model_M_1.tflite")
y_x_scaler_M_1 = load(str(SUR_DIR / "helix_y_stiffness_small_x_scaler_M_1.joblib"))
y_y_scaler_M_1 = load(str(SUR_DIR / "helix_y_stiffness_small_y_scaler_M_1.joblib"))
y_interpreter_M_1 = tflite.Interpreter(model_path=y_tflite_model_path_M_1)
y_interpreter_M_1.allocate_tensors()
# Get input and output details
y_input_details_M_1 = y_interpreter_M_1.get_input_details()
y_output_details_M_1 = y_interpreter_M_1.get_output_details()
# Load scalers for input and output



# Load the TensorFlow Lite model for y2_M: Half
y_tflite_model_path_M_2 = str(SUR_DIR / "helix_y_stiffness_surrogate_model_M_2.tflite")
y_interpreter_M_2 = tflite.Interpreter(model_path=y_tflite_model_path_M_2)
y_interpreter_M_2.allocate_tensors()
# Get input and output details
y_input_details_M_2 = y_interpreter_M_2.get_input_details()
y_output_details_M_2 = y_interpreter_M_2.get_output_details()
# Load scalers for input and output
y_x_scaler_M_2 = load(str(SUR_DIR / "helix_y_stiffness_small_x_scaler_M_2.joblib"))
y_y_scaler_M_2 = load(str(SUR_DIR / "helix_y_stiffness_small_y_scaler_M_2.joblib"))


# Load the TensorFlow Lite model for z1_M: Half unit
z_tflite_model_path_M_1 = str(SUR_DIR / "helix_z_stiffness_surrogate_model_M_1.tflite")
z_interpreter_M_1 = tflite.Interpreter(model_path=z_tflite_model_path_M_1)
z_interpreter_M_1.allocate_tensors()
# Get input and output details
z_input_details_M_1 = z_interpreter_M_1.get_input_details()
z_output_details_M_1 = z_interpreter_M_1.get_output_details()
# Load scalers for input and output
z_x_scaler_M_1 = load(str(SUR_DIR / "helix_z_stiffness_small_x_scaler_M_1.joblib"))
z_y_scaler_M_1 = load(str(SUR_DIR / "helix_z_stiffness_small_y_scaler_M_1.joblib"))

# Load the TensorFlow Lite model for z2_M: Half unit
z_tflite_model_path_M_2 = str(SUR_DIR / "helix_z_stiffness_surrogate_model_M_2.tflite")
z_interpreter_M_2 = tflite.Interpreter(model_path=z_tflite_model_path_M_2)
z_interpreter_M_2.allocate_tensors()
# Get input and output details
z_input_details_M_2 = z_interpreter_M_2.get_input_details()
z_output_details_M_2 = z_interpreter_M_2.get_output_details()
# Load scalers for input and output
z_x_scaler_M_2 = load(str(SUR_DIR / "helix_z_stiffness_small_x_scaler_M_2.joblib"))
z_y_scaler_M_2 = load(str(SUR_DIR / "helix_z_stiffness_small_y_scaler_M_2.joblib"))


# Load the TensorFlow Lite model for x1_M: Half unit
x_tflite_model_path_M_1 = str(SUR_DIR / "helix_x_stiffness_surrogate_model_M_1.tflite")
x_interpreter_M_1 = tflite.Interpreter(model_path=x_tflite_model_path_M_1)
x_interpreter_M_1.allocate_tensors()
# Get input and output details
x_input_details_M_1 = x_interpreter_M_1.get_input_details()
x_output_details_M_1 = x_interpreter_M_1.get_output_details()
# Load scalers for input and output
x_x_scaler_M_1 = load(str(SUR_DIR / "helix_x_stiffness_small_x_scaler_M_1.joblib"))
x_y_scaler_M_1 = load(str(SUR_DIR / "helix_x_stiffness_small_y_scaler_M_1.joblib"))

# Load the TensorFlow Lite model for x2_M: Half unit
x_tflite_model_path_M_2 = str(SUR_DIR / "helix_x_stiffness_surrogate_model_M_2.tflite")
x_interpreter_M_2 = tflite.Interpreter(model_path=x_tflite_model_path_M_2)
x_interpreter_M_2.allocate_tensors()
# Get input and output details
x_input_details_M_2 = x_interpreter_M_2.get_input_details()
x_output_details_M_2 = x_interpreter_M_2.get_output_details()
# Load scalers for input and output
x_x_scaler_M_2 = load(str(SUR_DIR / "helix_x_stiffness_small_x_scaler_M_2.joblib"))
x_y_scaler_M_2 = load(str(SUR_DIR / "helix_x_stiffness_small_y_scaler_M_2.joblib"))


inner_radius = 3.09/1000*X # m
wall_thickness = 1.51/1000*X # m
average_radius1 = 5.66/1000*X # m
average_radius2 = 5.71/1000*X # m
module_radius1 = average_radius1*2 - inner_radius
module_radius2 = average_radius2*2 - inner_radius
module_length1 = 4.72/1000*X # m
module_length2 = 4.77/1000*X # m
module_mass1 = 0.000867880000000000*X # kg
module_mass2 = 0.000891920000000000*X # kg
end_mass = np.pi*(4/1000*X)*(4/1000*X)*(6/1000*X)*1150 # kg, 4mm radius, 6mm length, density 1150kg/m^3


N = 4
phi = math.pi/2
Phi = [0, 0.85, 0.85+0.59, 0.85+0.59+1.01]
module_number = [5, 5, 6, 6]
MainJointAxes = []
SecondJointAxes = []
for n in range(N):
    axis1 = [math.cos(Phi[n]), math.sin(Phi[n]), 0]
    axis2 = [math.cos(Phi[n]+phi), math.sin(Phi[n]+phi), 0]
    MainJointAxis = [axis1 for _ in range(module_number[n])]
    SecondJointAxis = [axis2 for _ in range(module_number[n])]
    MainJointAxes.extend(MainJointAxis)
    SecondJointAxes.extend(SecondJointAxis)

# Parameters for the base (fixed in space)
baseMass = 0  # Set mass to 0 to make it static
baseCollisionShapeIndex = p.createCollisionShape(p.GEOM_SPHERE, radius = module_radius1)
baseVisualShapeIndex = p.createVisualShape(p.GEOM_SPHERE, radius=module_radius1, rgbaColor=[0.0, 0.5, 1.0, 1.0])
baseCollisionShapeIndex = p.createCollisionShape(p.GEOM_CYLINDER, radius=15/2/1000*X, height=module_length1)  # 6mm length
baseVisualShapeIndex = p.createVisualShape(p.GEOM_CYLINDER, radius=15/2/1000*X, length=module_length1, rgbaColor=[0.0, 0.5, 1.0, 1.0])


endCollisionShapeIndex = p.createCollisionShape(p.GEOM_CYLINDER, radius=4/1000* X, height=6/1000*X)  # 6mm length
endVisualShapeIndex = p.createVisualShape(p.GEOM_CYLINDER, radius=4/1000* X, length=6/1000*X, rgbaColor=[0.0, 0.5, 1.0, 1.0])

# Parameters for links
num_links1 = 10
num_links2 = 12

num_links = num_links1 + num_links2
baseStartPos = [0, 0, 2]  # Adjusted to center the links
print("baseStartPos: ", baseStartPos)

baseStartOri = p.getQuaternionFromEuler([0, np.pi/2 + np.radians(90-62), np.pi/2])



linkPositions = []  # Joint positions (offset by wall_thickness)
linkInertialFramePositions = []  # Correcting offset for links (opposite offset)
module_lengths = []
for i in range(num_links):
    # Module-specific parameters
    if i < num_links1:
        module_length = module_length1
    elif i == num_links1:
        module_length = module_length1/2 + module_length2/2  # Half of the first link and half of the second link
    elif i < num_links-1:
        module_length = module_length2
    else:
        module_length = module_length2
    module_lengths.append(module_length)
    # second joint axis for offset direction
    second_axis = SecondJointAxes[i]

    # Compute lateral offset for the joint pivot
    joint_lateral_offset = [component * wall_thickness for component in second_axis]

    # Set joint pivot position relative to parent link (laterally offset)
    linkPositions.append([
        -joint_lateral_offset[0],
        -joint_lateral_offset[1],
        module_length  # still along the Z-axis
    ])


    


linkMasses = [module_mass1] * num_links1 + [module_mass2] * (num_links2) + [end_mass]
mesh_path1 = str(MESH_DIR / "Seg1.STL")
mesh_path2 = str(MESH_DIR / "Seg2.STL")



linkCollisionShapeIndices = [
    p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path1, meshScale=[X * 1e-3]*3, physicsClientId=physicsClient) for _ in range(num_links1)] + [
    p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path2, meshScale=[X * 1e-3]*3, physicsClientId=physicsClient) for _ in range(num_links2)
] + [endCollisionShapeIndex]
linkVisualShapeIndices = [
    p.createVisualShape(p.GEOM_MESH, fileName=mesh_path1, meshScale=[X * 1e-3]*3, rgbaColor=[1.0, 1.0, 0.941, 1.0], physicsClientId=physicsClient) for _ in range(num_links1)] + [
    p.createVisualShape(p.GEOM_MESH, fileName=mesh_path2, meshScale=[X * 1e-3]*3, rgbaColor=[1.0, 1.0, 0.941, 1.0], physicsClientId=physicsClient) for _ in range(num_links2)
] + [endCollisionShapeIndex]
linkPositions = [[0, 0, module_lengths[i]] for i in range(num_links)] + [[0, 0, (module_length2 + 6/1000*X)/2]]
linkOrientations = []
for n in range(N):
    if n > 0:
        eulerangle = np.array([0, 0, Phi[n] - Phi[n-1]])
    else:
        eulerangle = np.array([0, 0, Phi[n]])
         
    quaternion = [p.getQuaternionFromEuler(eulerangle.tolist())] + [p.getQuaternionFromEuler([0, 0, 0]) for _ in range(module_number[n]-1)]
    linkOrientations.extend(quaternion)
# Add the end link with a fixed orientation
linkOrientations.append(p.getQuaternionFromEuler([0, 0, 0]))  # End link has no rotation

linkOrientations[0] = p.getQuaternionFromEuler([0, 0, 0])
linkInertialFramePositions = [[0, 0, 0] for _ in range(num_links+1)]
linkInertialFrameOrientations = [[0, 0, 0, 1] for _ in range(num_links+1)]
linkParentIndices = list(range(num_links+1))  # Each link is parented to the one before it
linkJointTypes = [p.JOINT_SPHERICAL for _ in range(num_links)] + [p.JOINT_FIXED]  # Spherical joints for all links, fixed joint for the end link

linkJointAxis = [[1, 0, 0] for _ in range(num_links+1)]  # Not used for spherical joints, but included for completeness


# Create multi-body with the base and links
robotUniqueId = p.createMultiBody(baseMass,
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
                                  linkJointAxis=linkJointAxis)



print("first link height [mm]: ", p.getLinkState(robotUniqueId, 0)[0][2]*1000/X)
print("last link height [mm]: ", p.getLinkState(robotUniqueId, num_links-1)[0][2]*1000/X)
initial_orientations = []
for linkIndex in range(num_links):
    # Assuming the base is index 0 and links start from 1
    linkState = p.getLinkState(robotUniqueId, linkIndex)
    orientation = linkState[1]
    initial_orientations.append(orientation)

# Track the height of the end link
end_link_heights = []

damping_force = [1e-4*X*X]*3
p.setJointMotorControlMultiDofArray(
    bodyUniqueId=robotUniqueId,
    jointIndices=list(range(num_links)),
    controlMode=p.POSITION_CONTROL,
    targetPositions=[[0, 0, 0, 1] for _ in range(num_links)],
    targetVelocities=[[0, 0, 0] for _ in range(num_links)],
    forces=[damping_force for _ in range(num_links)],
    positionGains=[0 for _ in range(num_links)],
    velocityGains=[1 for _ in range(num_links)],
    physicsClientId=physicsClient
)
    




time_step = 1. / 240./10 #0.0005
p.setTimeStep(time_step)

# torque applied function
omega = 1.0*math.pi*2


pressure_fns = [
        lambda t: 0,
        lambda t: 15*(1/2 * np.sin(omega * (t)  - np.pi/2) + 1/2),
        lambda t: 15
    ]

external_load_fns = [
        lambda t: 0,
        lambda t: -50/1000*9.81*(1/2 * np.sin(omega * (t) + np.pi/2) + 1/2),
        lambda t: -50/1000*9.81
]

n_steps = int(2*math.pi/omega/time_step*2) 
# the UnderG finish at 1 cycles but will add 1 cycles to obtain the last half cycle average and std
# so to obtain the average and std for UnderG, 2 cycles are simulated and comment out the external load part plz
# the UnderW finish at 2 cycles but will add 1 cycles to obtain the last half cycle average and std
# so to obtain the average and std for UnderW, 3 cycles are simulated 
sim_time = 0.0

cycle_time = 2*math.pi/omega # one cycle
settle_time = 1/2*cycle_time # half cycle
actuation_time = cycle_time # one
external_load_time = 1.5*cycle_time # one and a half
constant_load_time = 2*cycle_time # two cycles


joint_angle1 = np.zeros((num_links, n_steps))
joint_angle2 = np.zeros((num_links, n_steps))
joint_angle3 = np.zeros((num_links, n_steps))
joint_torque1 = np.zeros((num_links, n_steps))
joint_torque2 = np.zeros((num_links, n_steps))
joint_torque3 = np.zeros((num_links, n_steps))
time_plot = np.zeros((n_steps,))
end_position = np.zeros((num_links+1, 3)) # x, y, z
end_position[0] = np.array(baseStartPos)
helix_position = np.zeros((num_links+1, 3, n_steps))  # Store positions of each link over time
ext_load = np.zeros((n_steps,))

# Advance a few frames to let constraints settle
for _ in range(10):
    p.stepSimulation()
# Main simulation loop
for n_step in range(n_steps):
    
    
    # batch processing joint control

    # Batch get states
    start_time = time.time()
    spherical_states = p.getJointStatesMultiDof(robotUniqueId, [i for i in range(num_links)], physicsClient)

    # Extract all quaternions into a NumPy array of shape (N, 4)
    quats = np.array([state[0] for state in spherical_states])  # shape: (num_links, 4)
    
    MainJointAxes = [[1, 0, 0] for _ in range(num_links)]
    SecondJointAxes = [[0, 1, 0] for _ in range(num_links)]
    # Normalize axes
    main_axes = np.array(MainJointAxes)
    second_axes = np.array(SecondJointAxes)
    twist_axes = np.tile(np.array([0, 0, 1]), (num_links, 1))
    
    # Vectorized angle computation
    joint_angle1[:, n_step] = quaternion_rotation_batch(quats, main_axes)
    joint_angle2[:, n_step] = quaternion_rotation_batch(quats, second_axes) 
    joint_angle3[:, n_step] = quaternion_rotation_batch(quats, twist_axes)
        
    if sim_time <= settle_time:  # half cycle
        pressure = pressure_fns[0](sim_time)
    elif sim_time <= actuation_time: # half to full cycle
        pressure = pressure_fns[1](sim_time - settle_time)
    else:
        pressure = pressure_fns[2](sim_time)
    
    # Prepare main torques from TFLite model
    helix_position[0,:, n_step] = np.array(baseStartPos)
    for i in range(num_links):
        
        y_input_data = np.array([[pressure, joint_angle1[i][n_step]/2]])
        z_input_data = np.array([[pressure, joint_angle2[i][n_step]/2]]) 
        x_input_data = np.array([[pressure, joint_angle3[i][n_step]/2]])
        # first num_links1 uses y1_M, z1_M, x1_M model and rest uses y2_M, z2_M, x2_M model
        start_time = time.perf_counter()  # Start timing for this iteration
        if i < num_links1:    
            y_input_scaled = y_x_scaler_M_1.transform(y_input_data).astype(np.float32)
            y_interpreter_M_1.set_tensor(y_input_details_M_1[0]['index'], y_input_scaled)
            y_interpreter_M_1.invoke()
            y_output_scaled = y_interpreter_M_1.get_tensor(y_output_details_M_1[0]['index'])
            y_output = y_y_scaler_M_1.inverse_transform(y_output_scaled)
            joint_torque1[i][n_step] = y_output[0][0]
            model_time = time.perf_counter() - start_time  # End timing for this iteration
            
            z_input_scaled = z_x_scaler_M_1.transform(z_input_data).astype(np.float32)
            z_interpreter_M_1.set_tensor(z_input_details_M_1[0]['index'], z_input_scaled)
            z_interpreter_M_1.invoke()
            z_output_scaled = z_interpreter_M_1.get_tensor(z_output_details_M_1[0]['index'])
            z_output = z_y_scaler_M_1.inverse_transform(z_output_scaled)
            joint_torque2[i][n_step] = -z_output[0][0]
            
            x_input_scaled = x_x_scaler_M_1.transform(x_input_data).astype(np.float32)
            x_interpreter_M_1.set_tensor(x_input_details_M_1[0]['index'], x_input_scaled)
            x_interpreter_M_1.invoke()
            x_output_scaled = x_interpreter_M_1.get_tensor(x_output_details_M_1[0]['index'])
            x_output = x_y_scaler_M_1.inverse_transform(x_output_scaled)
            joint_torque3[i][n_step] = -x_output[0][0]  # Single value
                
        else:
            y_input_scaled = y_x_scaler_M_2.transform(y_input_data).astype(np.float32)
            y_interpreter_M_2.set_tensor(y_input_details_M_2[0]['index'], y_input_scaled)
            y_interpreter_M_2.invoke()
            y_output_scaled = y_interpreter_M_2.get_tensor(y_output_details_M_2[0]['index'])
            y_output = y_y_scaler_M_2.inverse_transform(y_output_scaled)
            joint_torque1[i][n_step] = y_output[0][0]
            
            model_time = time.perf_counter() - start_time  # End timing for this iteration
            
            z_input_scaled = z_x_scaler_M_2.transform(z_input_data).astype(np.float32)
            z_interpreter_M_2.set_tensor(z_input_details_M_2[0]['index'], z_input_scaled)
            z_interpreter_M_2.invoke()
            z_output_scaled = z_interpreter_M_2.get_tensor(z_output_details_M_2[0]['index'])
            z_output = z_y_scaler_M_2.inverse_transform(z_output_scaled)
            joint_torque2[i][n_step] = -z_output[0][0]  # Single value
            
            x_input_scaled = x_x_scaler_M_2.transform(x_input_data).astype(np.float32)
            x_interpreter_M_2.set_tensor(x_input_details_M_2[0]['index'], x_input_scaled)
            x_interpreter_M_2.invoke()
            x_output_scaled = x_interpreter_M_2.get_tensor(x_output_details_M_2[0]['index'])
            x_output = x_y_scaler_M_2.inverse_transform(x_output_scaled)
            joint_torque3[i][n_step] = -x_output[0][0]  # Single value

    

    applied_torques = (
        joint_torque1[:, n_step][:, np.newaxis] * np.array(MainJointAxes) +
        joint_torque2[:, n_step][:, np.newaxis] * np.array(SecondJointAxes) +
        joint_torque3[:, n_step][:, np.newaxis] * np.array([[0, 0, 1]] * num_links)
    )
    
    p.setJointMotorControlMultiDofArray(
        bodyUniqueId=robotUniqueId,
        jointIndices=[i for i in range(num_links)],
        controlMode=p.TORQUE_CONTROL,
        forces=applied_torques.tolist()
    )
        
    end_position[1:] = [p.getLinkState(robotUniqueId, linkIndex)[0] for linkIndex in range(num_links)]
    helix_position[1:,:, n_step] = end_position[1:]  # Store positions of each link over time
    
    
    
    
    # comment this if simulating UnderG
    # # apply external load after 1.5 cycles at the end link 
    # if sim_time > external_load_time and sim_time <= constant_load_time:
    # elif sim_time > constant_load_time:
    # else:
        
   
    
    # )
    # ext_load[n_step] = external_load
    
    
    # Track the height of the end link
    end_link_state = p.getLinkState(robotUniqueId, num_links-1)
    end_link_height = end_link_state[0][2]
    end_link_heights.append(end_link_height)
    
    first_link_orientation = p.getLinkState(robotUniqueId, 0)[5]
    first_link_normal_vector = p.getMatrixFromQuaternion(first_link_orientation)
    first_link_normal_vector = np.array(first_link_normal_vector).reshape(3, 3)[:, 2]
    
    last_link_orientation = p.getLinkState(robotUniqueId, num_links-1)[5]
    last_link_normal_vector = p.getMatrixFromQuaternion(last_link_orientation)
    last_link_normal_vector = np.array(last_link_normal_vector).reshape(3, 3)[:, 2]
        
        
    time_plot[n_step] = sim_time
    sim_time += time_step
    end_time = time.time()
    p.stepSimulation()
time.sleep(1)
    
first_position = end_position[0] - 1.5/1000*X*first_link_normal_vector
print("last link normal vector: ", last_link_normal_vector)


end_position_df = pd.DataFrame(end_position, columns=['x', 'y', 'z'])
end_position_df.to_csv('helical_end_pos.csv', index=False)

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'cyan', 'purple', 'pink', 'indigo', 'brown', 'lime', 'magenta', 'teal', 'lavender', 'turquoise', 'maroon', 'gold', 'silver', 'olive', 'coral', 'salmon', 'tan', 'plum', 'slate', 'charcoal']




plt.plot(time_plot, helix_position[0,:, :].T[:, 2]*1000/X, label='Ring1')
plt.plot(time_plot, helix_position[6,:, :].T[:, 2]*1000/X, label='Ring2')
plt.plot(time_plot, helix_position[12,:, :].T[:, 2]*1000/X, label='Ring3')
plt.plot(time_plot, helix_position[18,:, :].T[:, 2]*1000/X, label='Ring4')
plt.plot(time_plot, helix_position[-1,:, :].T[:, 2]*1000/X, label='Ring5')
plt.xlabel('time (s)')
plt.ylabel('helix ring height [mm]')
plt.title('Helix Ring Height')
plt.legend()
plt.grid()
plt.show()

plt.plot(time_plot, ext_load)
plt.xlabel('time (s)')
plt.ylabel('external load [N]')
plt.title('External Load over Time')
plt.grid()
plt.show()


# reshape helix_position to a DataFrame and save as a .csv file
# Reshape to (n_steps, (num_links+1) * 3)
num_links_plus_1, _, n_steps = helix_position.shape
reshaped = helix_position.transpose(2, 0, 1).reshape(n_steps, -1)
# Create column names
columns = [f'link{i}_{axis}' for i in range(num_links_plus_1) for axis in ['x', 'y', 'z']]
# Save to CSV
df = pd.DataFrame(reshaped, columns=columns)
df.to_csv("helix_position.csv", index=False)


p.disconnect()