"""Simulate a tendon-driven actuator under selected tip loads.

Uses the pretrained joint surrogate in PyBullet.
"""

import time
from zipfile import Path  # for waiting
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import ctypes
import warnings

import math




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

import tensorflow.lite as tflite
from joblib import load

from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent

tflite_model_path = str(THIS_DIR / "tendon_driven_surrogate_model.tflite")
x_scaler_path = str(THIS_DIR / "x_scaler.joblib")
y_scaler_path = str(THIS_DIR / "y_scaler.joblib")

top_mesh_path    = str(THIS_DIR / "link" / "link1.stl")
module_mesh_path = str(THIS_DIR / "link" / "link23.stl")
end_mesh_path    = str(THIS_DIR / "link" / "link4.stl")

interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

x_scaler = load(x_scaler_path)
y_scaler = load(y_scaler_path)


class SuppressOutputFD:
    def __enter__(self):
        # Flush Python-level buffers
        sys.stdout.flush()
        sys.stderr.flush()

        # Remember the original file descriptors
        self.orig_stdout_fd = os.dup(sys.stdout.fileno())
        self.orig_stderr_fd = os.dup(sys.stderr.fileno())

        # Open a null file
        self.devnull = os.open(os.devnull, os.O_WRONLY)

        # Duplicate the null file descriptor over stdout/stderr
        os.dup2(self.devnull, sys.stdout.fileno())
        os.dup2(self.devnull, sys.stderr.fileno())

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original file descriptors to stdout/stderr
        os.dup2(self.orig_stdout_fd, sys.stdout.fileno())
        os.dup2(self.orig_stderr_fd, sys.stderr.fileno())
        
        # Close the duplicated and null file descriptors
        os.close(self.orig_stdout_fd)
        os.close(self.orig_stderr_fd)
        os.close(self.devnull)
            
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc



def tendon_driven_simulation():
    

    # with SuppressOutputFD():
    X = 1  # world scaling factor
    
    physicsClient = p.connect(
        p.GUI
    )  # p.GUI for graphical, or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )  # defines the path used by p.loadURDF
    p.setGravity(0, 0, -9.8*X)
    p.setPhysicsEngineParameter(enableConeFriction=1)
    p.setRealTimeSimulation(
        0
    )  # only if this is set to 0 and with explicit steps will the torque control work correctly
    p.resetDebugVisualizerCamera(
        cameraDistance=0.3*X,
        cameraYaw=0.0,
        cameraPitch=-30.0,
        cameraTargetPosition=[0, 0, 0],
    )
    
    

    # load the ground plane
    planeId = p.loadURDF("plane.urdf")
    p.changeDynamics(planeId, -1, lateralFriction=1.0, rollingFriction=1.0, spinningFriction=1.0)
    
    module_number = 3
    # calculate the mass and inertia of the gripper
    rigid_density = 1240  # kg/m^3
    soft_density = 1120  # kg/m^3
    
    
    # link 2 & 3
    rigid_volume2 = 6967.283023e-9
    soft_volume2 = 2055.561247e-9
    
    link2_mass = rigid_volume2*rigid_density + soft_volume2*soft_density  # kg, this is the mass of one module, we can set it to a fixed value for simplicity
    module_mass = [link2_mass] * (module_number - 1)  # mass of each module
    # link 4
    rigid_volume3 = 8166.460645e-9
    soft_volume3 = 1703.826106e-9
    
    end_mass = rigid_volume3*rigid_density + soft_volume3*soft_density  # kg, this is the mass of the end
    module_mass.append(end_mass)  # mass of the end
    
    linkheight = 20.2*1e-3*X  # height of the link in meters
    baseLength = 31.987038*1e-3*X  # base length in meters
    linkLength = 28*1e-3*X  # length of the link in meters
    endLength = 28*1e-3*X  # length of the end in meters
    fingerLength = baseLength + linkLength*(module_number-1) + endLength  # total length of the finger in meters
    
    rgbaColor = [0.2, 0.2, 0.7, 1.0]

    baseMass = 0  # Set mass to 0 to make it static
    baseCollisionShape = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=top_mesh_path,
        meshScale=[X * 1e-3] * 3,
        physicsClientId=physicsClient,
    )
    baseVisualShape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=top_mesh_path,
        meshScale=[X * 1e-3] * 3,
        rgbaColor=rgbaColor,
        physicsClientId=physicsClient,
    )
    
    
    PosX = 0
    PosY = 0
    PosZ = (baseLength + linkLength*2 + endLength)
    baseStartPos = [PosX, PosY, PosZ]  # base position of the gripper
    baseStartOrn = p.getQuaternionFromEuler([0, 0, 0])  # XYZ Euler Angles

    
    
    linkCollisionShapeIndices = [
        p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=module_mesh_path, meshScale=[X * 1e-3] * 3, physicsClientId=physicsClient)
        for _ in range(module_number-1)] + [
        p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=end_mesh_path, meshScale=[X * 1e-3] * 3, physicsClientId=physicsClient)
    ]
    linkVisualShapeIndices = [
        p.createVisualShape(shapeType=p.GEOM_MESH, fileName=module_mesh_path, meshScale=[X * 1e-3] * 3, rgbaColor=rgbaColor, physicsClientId=physicsClient)
        for _ in range(module_number-1)] + [
        p.createVisualShape(shapeType=p.GEOM_MESH, fileName=end_mesh_path, meshScale=[X * 1e-3] * 3, rgbaColor=rgbaColor, physicsClientId=physicsClient)
    ]
    
    

    linkPositions = []
    for i in range(module_number):
        if i == 0:
            linkPosition = [baseLength, 0, 0]
        else:
            linkPosition = [linkLength, 0, 0]
        linkPositions.append(linkPosition)
    

            
    linkOrientations = [[0, 0, 0, 1] for _ in range(module_number)]  # No rotation for links
    linkInertialFramesPos = [[0, 0, 0] for _ in range(module_number)]  # No offset for inertial frames
    linkInertialFramesOrn = [[0, 0, 0, 1] for _ in range(module_number)]
    linkParentIds = list(range(module_number))  # Each link is a child of the previous one
    linkJointTypes = [p.JOINT_REVOLUTE] * module_number  # All joints are revolute
    linkJointAxis = [[0, 1, 0]] * module_number
    
    finger = p.createMultiBody(
        baseMass=baseMass,
        baseCollisionShapeIndex=baseCollisionShape,
        baseVisualShapeIndex=baseVisualShape,
        basePosition=baseStartPos,
        baseOrientation=baseStartOrn,
        baseInertialFramePosition=[0, 0, 0],
        linkMasses = module_mass,
        linkCollisionShapeIndices=linkCollisionShapeIndices,
        linkVisualShapeIndices=linkVisualShapeIndices,
        linkPositions=linkPositions,
        linkOrientations=linkOrientations,
        linkInertialFramePositions=linkInertialFramesPos,
        linkInertialFrameOrientations=linkInertialFramesOrn,
        linkParentIndices=linkParentIds,
        linkJointTypes=linkJointTypes,
        linkJointAxis=linkJointAxis,
        physicsClientId=physicsClient
    )
    inertia = p.getDynamicsInfo(finger, 0, physicsClientId=physicsClient)[2]
        
    for jointId in range(p.getNumJoints(finger, physicsClient)):
        limitforce = 1*1e-4*X*X
        p.setJointMotorControl2(
            bodyIndex=finger,
            jointIndex=jointId,
            controlMode=p.VELOCITY_CONTROL,
            force=limitforce,
            physicsClientId=physicsClient,
        )

    
    # simulation setup
    time_step = 1. / 240./10 #0.0005
    p.setTimeStep(time_step)
    omega = math.pi
    cycle = 2*math.pi/omega
    total_time = 3*cycle
    n_steps = int(total_time/time_step)
    sim_time = 0.0
    
    load_mass = 0 #grams
    if load_mass == 0:
        maximum_tension1 = 8.07
    elif load_mass == 20:
        maximum_tension1 = 9.185
    else:
        maximum_tension1 = 10.77 
        
    maximum_tension2 = maximum_tension1
    # tension applied function
    tension_fns = [
        lambda t: 0,
        lambda t: maximum_tension1*(1/2 * np.sin(omega * t - np.pi/2) + 1/2),
        lambda t: maximum_tension1,
        lambda t: (maximum_tension2 - maximum_tension1) * (1/2 * np.sin(omega * t - np.pi/2) + 1/2) + maximum_tension1,
        lambda t: maximum_tension2
    ]
    
    if load_mass == 0:
        maximum_load = 0
    elif load_mass == 20:
        maximum_load = 20*1e-3*X*9.81  # maximum load in N
    else:
        maximum_load = 50*1e-3*X*9.81  # maximum load in N

    external_fncs = [
        lambda t: 0,
        lambda t: maximum_load*(1/2 * np.sin(omega * t - np.pi/2) + 1/2),  # external load in N
        lambda t: maximum_load
    ]
    
    real_time = time.time()
    time_plot = np.zeros((n_steps,))
    
    
    applied_torque = np.zeros((module_number, n_steps))
    joint_angle = np.zeros((module_number, n_steps))
    overall_angle = np.zeros((n_steps,))
    joint_distance = np.zeros((module_number, n_steps))
    tendon_length = np.zeros((n_steps,))
    applied_load = np.zeros((n_steps, 3))  # external load vector
    applied_load_1 = np.zeros((n_steps, 3))  # external load vector
    applied_load_2 = np.zeros((n_steps, 3))  # external load vector
    external_load = np.zeros((n_steps,))  # external load scalar
    end_link_position = np.zeros((n_steps, 3))  # end link position
    load_vector_all_1 = np.zeros((n_steps, 3))  # load vector in the world frame
    load_vector_all_2 = np.zeros((n_steps, 3))  # load vector in the world frame
    
        
    for i in range(n_steps):
        
        for jointId in range(module_number):
            
            
            jointState = p.getJointState(finger, jointId, physicsClient)
            jointPos = -jointState[0]
            
            # relationship between joint angle and distance
            
            p0 = -1.26976032902333*1e-3*X
            p1 = -2.49327048903771*1e-3*X
            p2 = 27.2972620871167*1e-3*X*X
            
            dist = p0*jointPos**2 + p1*jointPos + p2  # distance in meters
            joint_distance[jointId, i] = dist
            
            
            if sim_time <= 1/2*cycle:  # half cycle, T = 0
                T = tension_fns[0](sim_time)
            elif sim_time >= 1/2*cycle and sim_time < 1*cycle:  # second half cycle, T -> Tmax1
                T = tension_fns[1](sim_time - 1/2*cycle)
            elif sim_time >= 1*cycle and sim_time < 2*cycle:  # third cycle, T = Tmax1
                T = tension_fns[2](sim_time - 1*cycle)
            elif sim_time >= 2*cycle and sim_time < 5/2*cycle:  # fourth cycle, T -> Tmax2
                T = tension_fns[3](sim_time - 2*cycle)
            else:  # fifth cycle, T = Tmax2
                T = tension_fns[4](sim_time - 2*cycle)

            input_data = np.array([[T, jointPos]])                
            input_scaled = x_scaler.transform(input_data).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_scaled)
            interpreter.invoke()
            output_scaled = interpreter.get_tensor(output_details[0]['index'])
            output = y_scaler.inverse_transform(output_scaled)
            torque_net = output[0][0]
            

            
            p.setJointMotorControl2(
                bodyIndex=finger, 
                jointIndex=jointId,
                controlMode=p.TORQUE_CONTROL,
                force = -X*X*torque_net, 
                physicsClientId=physicsClient,
            )
            applied_torque[jointId, i] = -torque_net
            joint_angle[jointId, i] = jointPos
            
        overall_angle[i] = sum(joint_angle[:, i])  # overall angle of the finger
        tendon_length[i] = sum(joint_distance[:, i])  # overall distance of the finger
        
        if sim_time >= 0*cycle and sim_time < 1/2*cycle:
            external_load[i] = external_fncs[1](sim_time - cycle)
        elif sim_time >= 1/2*cycle:
            external_load[i] = external_fncs[2](sim_time)
        else:
            external_load[i] = external_fncs[0](sim_time)

            
        tip_position = p.getLinkState(finger, module_number-1, computeForwardKinematics=1, physicsClientId=physicsClient)[0]
        Rot = [np.cos(overall_angle[i]), 0, -np.sin(overall_angle[i]),
               0, 1, 0,
               np.sin(overall_angle[i]), 0, np.cos(overall_angle[i])]
        
        # [(25.486914 + 23.247312)/2*1e-3*X, 0, 6.7*1e-3*X]
        # [8*1e-3*X, 0, 6.7*1e-3*X]
        
        applied_vector = [endLength/2, 0, 6.7*1e-3*X] # last used
         
        applied_vector_1 = [8*1e-3*X, 0, 6.7*1e-3*X] # fixed point of tendon on the finger
        applied_vector_2 = [-3.59*1e-3*X, 0, -linkheight/2]  # point of where tendon is leaving the finger
        
        applied_position = np.array(tip_position) + np.dot(np.array(Rot).reshape(3, 3), applied_vector)
        applied_position_1 = np.array(tip_position) + np.dot(np.array(Rot).reshape(3, 3), applied_vector_1)
        applied_position_2 = np.array(tip_position) + np.dot(np.array(Rot).reshape(3, 3), applied_vector_2)
        

        trans_vector = [endLength, 0, 0] # last used
        
        end_position = np.array(tip_position) + np.dot(np.array(Rot).reshape(3, 3), trans_vector)
        end_position_1 = np.array(tip_position) + np.dot(np.array(Rot).reshape(3, 3), applied_vector_1)
        end_position_2 = np.array(tip_position) + np.dot(np.array(Rot).reshape(3, 3), applied_vector_2)
        
        load_vector = np.array([fingerLength - end_position[0], 0, -(end_position[2] + linkheight/2 - PosZ)])  # load vector in the world frame
        load_vector_1 = np.array([fingerLength + linkheight/2 - end_position_1[0], 0, -(end_position_1[2] + linkheight/2 - PosZ)])  # load vector in the world frame
        load_vector_2 = np.array([fingerLength + linkheight/2 - end_position_2[0], 0, -(end_position_2[2] + linkheight/2 - PosZ)])  # load vector in the world frame
        
        
        load_norm = np.linalg.norm(load_vector)
        load_vector_all_1[i] = load_vector/ load_norm
        
        load_vector2 = np.dot(np.array(Rot).reshape(3, 3), [-1, 0, 0])
        load_norm2 = np.linalg.norm(load_vector2)
        load_vector_all_2[i] = load_vector2/ load_norm2
        
        

        
        

        if load_norm > 0:
            applied_load[i] = (external_load[i] * load_vector / load_norm)
            applied_load_1[i] = (external_load[i] * load_vector_1/ np.linalg.norm(load_vector_1))
            applied_load_2[i] = (external_load[i] * load_vector_2 / np.linalg.norm(load_vector_2))
        else:
            applied_load[i] = np.array([0.0, 0.0, 0.0])
            applied_load_1[i] = np.array([0.0, 0.0, 0.0])
            applied_load_2[i] = np.array([0.0, 0.0, 0.0])
        
        p.applyExternalForce(
            objectUniqueId=finger,
            linkIndex=module_number-1,  # Apply torque to the last link
            forceObj = applied_load_2[i],
            posObj = applied_position_2,
            flags = p.WORLD_FRAME
        )
        
        dist_end_center = 8.516179*1e-3*X  # distance from the end center to the last link center

        initial_x = baseLength + linkLength*(module_number-1) + dist_end_center
        initial_z = 0
        
        end_link_position[i] = p.getLinkState(finger, module_number-1, computeForwardKinematics=1, physicsClientId=physicsClient)[0]
        end_link_position[i] = [initial_x - (end_link_position[i][0] + dist_end_center*np.cos(overall_angle[i])),
                                end_link_position[i][1],
                                end_link_position[i][2] + dist_end_center*np.sin(overall_angle[i]) - PosZ]
        
        

        p.stepSimulation()

        time_plot[i] = sim_time
        sim_time += time_step

        # time it took to simulate
        delta = time.time() - real_time
        real_time = time.time()
    
    time.sleep(1)
    
    p.disconnect()
    
    # SuppressOutputFD disabled
    print(linkPositions)
    
    
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'cyan', 'purple', 'pink', 'indigo', 'brown', 'lime', 'magenta', 'teal', 'lavender', 'turquoise', 'maroon', 'gold', 'silver', 'olive', 'coral', 'salmon', 'tan', 'plum', 'slate', 'charcoal']

    for i in range(module_number):
        plt.plot(time_plot, joint_angle[i]*180/np.pi, color = colors[i], label=f'joint {i+1}')
        print(f'joint {i+1} mean angle last half cycle: {np.mean(joint_angle[i, int(n_steps/2):-1])*180/np.pi:.2f} degree')
    plt.legend(loc='lower right')
    plt.xlabel('time (s)')
    plt.ylabel('joint angle [degree]')
    
    plt.show()
    
    plt.plot(time_plot, overall_angle*180/np.pi, color='black', label='overall angle')
    plt.legend(loc='lower right')
    plt.xlabel('time (s)')
    plt.ylabel('overall angle [degree]')
    plt.show()
    
    # print average of the last overall angle over last half cycle
    print(f'overall angle mean last half cycle: {np.mean(overall_angle[int(n_steps/2):-1])*180/np.pi:.2f} degree')
    # print the std of the last overall angle over last half cycle
    print(f'overall angle std last half cycle: {np.std(overall_angle[int(n_steps/2):-1])*180/np.pi:.2f} degree')

    # for i in range(module_number):
    
    # for i in range(module_number):
    print(f'tendon length mean last half cycle: {np.mean(tendon_length[int(n_steps/2):-1])/X*1000:.2f} mm')
    

    
    
    plt.plot(time_plot, load_vector_all_1[:, 0], color='black', label='load vector 1 x')
    plt.plot(time_plot, load_vector_all_1[:, 2], color='blue',label='load vector 1 z')
    plt.plot(time_plot, load_vector_all_2[:, 0], color='orange', label='load vector 2 x')
    plt.plot(time_plot, load_vector_all_2[:, 2], color='red',label='load vector 2 z')
    plt.legend(loc='lower right')
    plt.xlabel('time (s)')
    plt.ylabel('load vector [N]')
    plt.title('Load Vector on the Finger')
    plt.show()


    
   


tendon_driven_simulation()
 

