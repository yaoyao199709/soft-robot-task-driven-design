"""Surrogate-driven PyBullet grasp simulation used for RL co-design.

Constructs the gripper, evaluates actuator response using learned models,
and returns task outcomes for the reinforcement-learning environment."""

import time  # for waiting
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import ctypes
import warnings

import math
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent          # .../Gripper/Sim2Real
ROOT_DIR = THIS_DIR.parent                          # .../Gripper
LINK_DIR = ROOT_DIR / "link"
OBJ_DIR  = ROOT_DIR / "objects"
URDF_DIR = ROOT_DIR / "urdf"

holder_urdf_path = str(URDF_DIR / "holder.urdf")
cylinder_urdf_path = str(URDF_DIR / "cylinder.urdf")
egg_urdf_path = str(URDF_DIR / "egg.urdf")


sys.path.insert(0, str(URDF_DIR))
from modify_cylinder_urdf import modify_cylinder_urdf
from modify_egg_urdf import modify_egg_urdf


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


from functools import lru_cache

@lru_cache(maxsize=1)
def _load_torque_net():
    print("⚙️ [Model Load] Loading torque_net_NN2.h5 for the first time ...")

    model_path = THIS_DIR / "poly_meta-model.h5"
    xsc_path   = THIS_DIR / "poly_meta-model_x_scaler.joblib"
    ysc_path   = THIS_DIR / "poly_meta-model_y_scaler.joblib"

    for p in (model_path, xsc_path, ysc_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    model = load_model(str(model_path),
                       custom_objects={'mse': MeanSquaredError()},
                       compile=False)
    model.compile(optimizer='adam', loss=MeanSquaredError())

    xsc = load(str(xsc_path))
    ysc = load(str(ysc_path))

    print("✅ [Model Load] Model and scalers loaded successfully.")
    return model, xsc, ysc

def gripper_simulator(init_grasp_height, maximum_pressure, grasp_radius_distance, inclination_angle, inner_radius, wall_thickness, average_radius, module_length, module_number, actuator_number, object_type, extra_weight, weight_num, add_noise,
                      render: bool=True,
                      plot: bool=False,
                      seed: int|None = None,
                      return_traces: bool=False,
                      verbose: int=0):
    
    if seed is not None:
        rng = np.random.default_rng(seed)
        rand = lambda *size: rng.random(size)
    else:
        rand = np.random.rand
        
    
    torque_net_NN, torque_net_x_scaler, torque_net_y_scaler = _load_torque_net()
    if _load_torque_net.cache_info().hits > 0:
        print(f"[Cache Hit] ⚡ torque_net_NN.h5 reused ({_load_torque_net.cache_info().hits} hits so far)")

    
    design_para_scaled = torque_net_x_scaler.transform(np.array([[inner_radius, average_radius, module_length]]))
    torque_net_para_scaled = torque_net_NN.predict(design_para_scaled)
    torque_net_para = torque_net_y_scaler.inverse_transform(torque_net_para_scaled)
    
    m_act = torque_net_para[0][0]
    b_act = torque_net_para[0][1]
    m_ela = torque_net_para[0][2]
    b_ela = torque_net_para[0][3]
    k_torque_theta0 = torque_net_para[0][4]


    with SuppressOutputFD():
        X = 10  # world scaling factor
        physicsClient = p.connect(p.GUI if render else p.DIRECT)
        
        if render:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.3*X,
                cameraYaw=0.0,
                cameraPitch=-30.0,
                cameraTargetPosition=[0, 0, 0],
            )

        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        p.setGravity(0, 0, -9.8*X)
        p.setPhysicsEngineParameter(enableConeFriction=1)
        p.setRealTimeSimulation(0) 
        
        p.setAdditionalSearchPath(str(ROOT_DIR))
        p.setAdditionalSearchPath(str(OBJ_DIR))
        p.setAdditionalSearchPath(str(URDF_DIR))
        
        
        limitForce = 1e-3
        lateralFriction = 0.8
        # simulation setup
        time_step = 1. / 240./10 #0.0005
        
        p.setTimeStep(time_step)
        omega = math.pi
        cycle = 2*math.pi/omega
        total_time = 3*cycle 
        n_steps = int(total_time/time_step)
        sim_time = 0.0
        
        if add_noise == True:
            maximum_pressure = maximum_pressure + 1*(rand() - 0.5)  # add some noise to the maximum pressure

        
        # torque applied function
        pressure_fns = [
            lambda t: 0,
            lambda t: maximum_pressure*(1/2 * np.sin(omega * t - np.pi/2) + 1/2),
            lambda t: maximum_pressure
        ]
        
        # move the base
        # the maximum speed of the end effector is 2m/s and it was using 10% of it
        # so we can set the vertical velocity to 0.2m/s*X
        vertical_velocity =  0.03*X* time_step  # in m/step
        
        # as the first half cycle is the downward movement, we need to reset the initial position of the gripper
        movement = vertical_velocity * cycle / time_step  # downward movement in the first half cycle
        
        # load the ground plane
        planeId = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
        p.changeDynamics(planeId, -1, lateralFriction=1.0, rollingFriction=1.0, spinningFriction=1.0)
        
        # load the static holder
        holderId = p.loadURDF(holder_urdf_path, basePosition=[0, 0, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
        p.changeDynamics(holderId, -1, lateralFriction=1.0, rollingFriction=1.0, spinningFriction=1.0)
        holder_height = 10*1e-3*X  # height of the holder in meters
        
        # load the object
        script_dir = os.path.dirname(os.path.abspath(__file__))  # directory of your script
        
        if object_type == "cylinder":
            cylinder_urdf_path = os.path.join(script_dir, "cylinder.urdf")
            if extra_weight == True:
                cylinder_urdf_path = str(URDF_DIR / "cylinder.urdf")
                modify_cylinder_urdf(cylinder_urdf_path, cylinder_urdf_path, weight_num, X)
            else:
                cylinder_urdf_path = str(URDF_DIR / "cylinder.urdf")
                modify_cylinder_urdf(cylinder_urdf_path, cylinder_urdf_path, 0, X)
            object_radius = 32/2*1e-3*X  # radius of the cylinder in meters
            object_height = 35*1e-3*X  # height of the cylinder in meters
            # add random noise to the object position: within 5% of the object_radius
            if add_noise == True:
                noise = object_radius * 0.1 * (rand(2) - 0.5)  # random noise in x and y direction
            else:
                noise = [0, 0]
            object_base_Pos = [noise[0], noise[1], holder_height]  # base position of the cylinder
            object_base_Or = p.getQuaternionFromEuler([0, 0, 0])
            objectId = p.loadURDF(cylinder_urdf_path, basePosition = object_base_Pos, baseOrientation = object_base_Or)
        elif object_type == "egg":
            egg_urdf_path = os.path.join(script_dir, "egg.urdf")
            if extra_weight == True:
                egg_urdf_path = str(URDF_DIR / "egg.urdf")
                modify_egg_urdf(egg_urdf_path, egg_urdf_path, weight_num, X)
            else:
                egg_urdf_path = str(URDF_DIR / "egg.urdf")
                modify_egg_urdf(egg_urdf_path, egg_urdf_path, 0, X)
                
            object_radius = 30*1e-3*X
            object_height = 65*1e-3*X
            hollow_bottom = 25*1e-3*X
            if add_noise == True:
                noise = object_radius * 0.1 * (rand(2) - 0.5)  # random noise in x and y direction
            else:
                noise = [0, 0]
            object_base_Pos = [noise[0], noise[1], holder_height]  # base position of the egg
            object_base_Or = p.getQuaternionFromEuler([0, 0, 0])
            objectId = p.loadURDF(egg_urdf_path, basePosition = object_base_Pos, baseOrientation = object_base_Or)


        p.changeDynamics(objectId, -1, lateralFriction=1)


        # calculate the mass and inertia of the gripper
        density = 1150 # kg/m^3
        flank = 2*average_radius - 2*inner_radius - module_length/2
        V_bellow_1 = 2*math.pi*wall_thickness*module_length/4*((inner_radius + module_length/4)*math.pi/2 - module_length/4) # mm^3
        V_bellow_2 = 2*math.pi*wall_thickness*((inner_radius + module_length/4)*flank + flank*flank/2) # mm^3
        V_bellow_3 = 2*math.pi*wall_thickness*module_length/4*((inner_radius + module_length/4 + flank)*math.pi/2 + module_length/4) # mm^3
        V_bellow = 2*(V_bellow_1 + V_bellow_2 + V_bellow_3)
        V_bottom = 2*wall_thickness* 2*average_radius*module_length # mm^3
        module_mass = density*(V_bellow +V_bottom)*1e-9
        
        outer_radius = average_radius*2 - inner_radius
        Ixx = 1/12*module_mass*(6*outer_radius*outer_radius + module_length*module_length)*1e-6
        Iyy = 1/12*module_mass*(6*outer_radius*outer_radius + module_length*module_length)*1e-6
        Izz = 1/2*module_mass*outer_radius*outer_radius*1e-6
        module_length = module_length*1e-3*X  # convert to meters
        outer_radius = outer_radius*1e-3*X  # convert to meters
        inner_radius = inner_radius*1e-3*X  # convert to meters
        wall_thickness = wall_thickness*1e-3*X  # convert to meters
        
        gripper = [None] * actuator_number  # list to hold gripper objects
        for fingerId in range(actuator_number):
            top_mesh_path = str(LINK_DIR / "top.STL")
            baseMass = 0  # Set mass to 0 to make it static
            baseLength = 6.5*1e-3*X  # base length in meters
            default_base_Y = (2 + 10 + 1.5*3)*1e-3*X  # default base Y in meters, X and Z will be the same as the default
            baseCollisionShape = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=top_mesh_path,
                meshScale=[
                    X * 1e-3,
                    X * 1e-3 * (inner_radius + outer_radius + 3*wall_thickness) / default_base_Y,
                    X * 1e-3
                    ],
                physicsClientId=physicsClient,
            )
            baseVisualShape = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=top_mesh_path,
                meshScale=[
                    X * 1e-3,
                    X * 1e-3 * (inner_radius + outer_radius + 3*wall_thickness) / default_base_Y,
                    X * 1e-3
                    ],
                rgbaColor=[0.0, 0.5, 1.0, 1.0],
                physicsClientId=physicsClient,
            )
            radiusPos = object_radius + grasp_radius_distance*1e-3*X  # radius position in meters
            fingerZ = np.cos(inclination_angle)*(baseLength + module_length*module_number + wall_thickness)
            PosX = radiusPos*np.cos(np.pi*2/actuator_number*(fingerId))
            PosY = radiusPos*np.sin(np.pi*2/actuator_number*(fingerId))
            PosZ = holder_height + init_grasp_height*1e-3*X + max(fingerZ, object_height)
            
            baseStartPos = [PosX, PosY, PosZ + movement]  # base position of the gripper
            baseStartOrn = p.getQuaternionFromEuler([-inclination_angle, np.pi, -np.pi/2 + np.pi*2/actuator_number*(fingerId)])  # XYZ Euler Angles
            
            module_mesh_path = str(LINK_DIR / "module.STL")
            end_mesh_path = str(LINK_DIR / "end.STL")
            end_length = module_length + 2*wall_thickness  # end length in meters
            
            default_X = 23*1e-3*X  # default outer radius in meters
            default_Y = 24.5*1e-3*X  # default Y in meters
            default_Z = 11*1e-3*X  # default Z in meters
            default_end_Z = 14*1e-3*X  # default end Z in meters
            
            linkCollisionShapeIndices = [
                p.createCollisionShape(
                    shapeType=p.GEOM_MESH, 
                    fileName=module_mesh_path, 
                    meshScale=[
                        X * 1e-3 * (outer_radius*2 + wall_thickness*2) / default_X,
                        X * 1e-3 * (outer_radius*2 + wall_thickness*3) / default_Y,
                        X * 1e-3 * module_length / default_Z,
                        ], 
                    physicsClientId=physicsClient
                )
                for _ in range(module_number-1)] + [
                p.createCollisionShape(
                    shapeType=p.GEOM_MESH, 
                    fileName=end_mesh_path, 
                    meshScale=[
                        X * 1e-3 * (outer_radius*2 + wall_thickness*2) / default_X,
                        X * 1e-3 * (outer_radius*2 + wall_thickness*3) / default_Y,
                        X * 1e-3 * end_length / default_end_Z,
                        ], 
                    physicsClientId=physicsClient
                )
            ]
            linkVisualShapeIndices = [
                p.createVisualShape(
                    shapeType=p.GEOM_MESH, 
                    fileName=module_mesh_path, 
                    meshScale=[
                        X * 1e-3 * (outer_radius*2 + wall_thickness*2) / default_X,
                        X * 1e-3 * (outer_radius*2 + wall_thickness*3) / default_Y,
                        X * 1e-3 * module_length / default_Z,
                        ], 
                    rgbaColor=[1.0, 1.0, 0.941, 1.0], 
                    physicsClientId=physicsClient
                )
                for _ in range(module_number-1)] + [
                p.createVisualShape(
                    shapeType=p.GEOM_MESH, 
                    fileName=end_mesh_path, 
                    meshScale=[
                        X * 1e-3 * (outer_radius*2 + wall_thickness*2) / default_X,
                        X * 1e-3 * (outer_radius*2 + wall_thickness*3) / default_Y,
                        X * 1e-3 * end_length / default_end_Z,
                        ], 
                    rgbaColor=[1.0, 1.0, 0.941, 1.0], 
                    physicsClientId=physicsClient
                )
            ]
            
            linkPositions = []
            for i in range(module_number):
                if i == 0:
                    linkPosition = [0, 0, (baseLength + module_length)/2]
                elif i == module_number - 1:
                    linkPosition = [0, 0, (module_length + end_length)/2]
                else:
                    linkPosition = [0, 0, module_length]
                linkPositions.append(linkPosition)

                    
            linkOrientations = [[0, 0, 0, 1] for _ in range(module_number)]  # No rotation for links
            linkInertialFramesPos = [[0, 0, 0] for _ in range(module_number)]  # No offset for inertial frames
            linkInertialFramesOrn = [[0, 0, 0, 1] for _ in range(module_number)]
            linkParentIds = list(range(module_number))  # Each link is a child of the previous one
            linkJointTypes = [p.JOINT_REVOLUTE] * module_number  # All joints are revolute
            linkJointAxis = [[1, 0, 0]] * module_number
            
            gripper[fingerId] = p.createMultiBody(
                baseMass=baseMass,
                baseCollisionShapeIndex=baseCollisionShape,
                baseVisualShapeIndex=baseVisualShape,
                basePosition=baseStartPos,
                baseOrientation=baseStartOrn,
                baseInertialFramePosition=[0, 0, 0],
                linkMasses = [module_mass] * module_number,
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
            inertia = p.getDynamicsInfo(gripper[fingerId], 0, physicsClientId=physicsClient)[2]
            
            
            if add_noise == True:
                frictionNoise = lateralFriction * 0.1 * (rand() - 0.5)  # random noise in friction
            else:
                frictionNoise = 0

            for jointId in range(p.getNumJoints(gripper[fingerId], physicsClient)):
                p.changeDynamics(
                    gripper[fingerId],
                    jointId,
                    lateralFriction=lateralFriction + frictionNoise,  # add some noise to the friction
                    frictionAnchor=True,
                    physicsClientId=physicsClient,
                )
                
                p.setJointMotorControl2(
                    bodyIndex=gripper[fingerId],
                    jointIndex=jointId,
                    controlMode=p.VELOCITY_CONTROL,
                    force=limitForce*X*X,
                    physicsClientId=physicsClient,
                )


        real_time = time.time()
        time_plot = np.zeros((n_steps,))
        
        
        applied_torque = np.zeros((actuator_number, module_number, n_steps))
        joint_angle = np.zeros((actuator_number, module_number, n_steps))

        
        contact_normal = np.zeros((actuator_number, module_number, n_steps))
        contact_friction = np.zeros((actuator_number, module_number, n_steps))
        
        
        object_position = np.zeros((n_steps,))
        gripper_position = np.zeros((actuator_number, n_steps))  # fingerId, [z], step
        
        
        actuation_starttime = cycle
        actuation_endtime = 1.5*cycle
        downward_endtime = cycle
        upward_starttime = 2*cycle
        upward_endtime = 3*cycle
        
        sim_starttime = time.time()
        for i in range(n_steps):
            
            for jointId in range(module_number):
                
                for fingerId in range(actuator_number): 
                    jointState = p.getJointState(gripper[fingerId], jointId, physicsClient)
                    jointPos = jointState[0]
                    
                    if sim_time <= cycle:  # 0kPa for the first cycle
                        pressure = pressure_fns[0](sim_time)
                    elif sim_time <= actuation_endtime: # actuation phase
                        pressure = pressure_fns[1](sim_time - actuation_starttime)
                    else: # maximum pressure for the rest
                        pressure = pressure_fns[2](sim_time)
                    
                    torque_net = (m_act*pressure + b_act)*jointPos - (0.5*m_ela*jointPos*jointPos*np.sign(jointPos) + b_ela*jointPos) + k_torque_theta0*pressure

                    
                    p.setJointMotorControl2(
                        bodyIndex=gripper[fingerId], 
                        jointIndex=jointId,
                        controlMode=p.TORQUE_CONTROL,
                        force = X*X*torque_net, 
                        physicsClientId=physicsClient,
                    )
                    applied_torque[fingerId, jointId, i] = torque_net
                    joint_angle[fingerId, jointId, i] = jointPos
                    
                    # check for contacts on each link
                    contacts = p.getContactPoints(gripper[fingerId], objectId, jointId)
                    # sum the force for each link
                    contact_normal[fingerId, jointId, i] = np.sum([contact[9] for contact in contacts])/X
                    contact_friction[fingerId, jointId, i] = abs(np.sum([contact[12] for contact in contacts]))/X
                    frictiondir =  contacts[0][13] if contacts else [0, 0, 0]  # get the friction direction from the first contact point if exists

                    gripper_position[fingerId, i] = p.getBasePositionAndOrientation(gripper[fingerId])[0][2]/X*1e3 - PosZ*1e3/X  # get the z position of the gripper finger in mm
                
            # move the gripper downward
            if sim_time >= 0 and sim_time < downward_endtime:
                for fingerId in range(actuator_number):
                    pos, ori = p.getBasePositionAndOrientation(gripper[fingerId])
                    new_pos = [pos[0], pos[1], pos[2] - vertical_velocity]
                    p.resetBasePositionAndOrientation(gripper[fingerId], new_pos, ori)
                
            # move the gripper upward
            
            if sim_time >= upward_starttime and sim_time < upward_endtime: 
                for fingerId in range(actuator_number):
                    pos, ori = p.getBasePositionAndOrientation(gripper[fingerId])
                    new_pos = [pos[0], pos[1], pos[2] + vertical_velocity]
                    p.resetBasePositionAndOrientation(gripper[fingerId], new_pos, ori)


            object_pos, object_ori = p.getBasePositionAndOrientation(objectId)
            # get the z position of the object in mm
            object_position[i] = (object_pos[2] - holder_height)/X*1e3 
            p.stepSimulation()

            time_plot[i] = sim_time
            sim_time += time_step

            # time it took to simulate
            delta = time.time() - real_time
            real_time = time.time()
        
        time.sleep(1)
        
        p.disconnect()
    
    sim_endtime = time.time()
    
    
    if plot: 
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'cyan', 'purple', 'pink', 'indigo', 'brown', 'lime', 'magenta', 'teal', 'lavender', 'turquoise', 'maroon', 'gold', 'silver', 'olive', 'coral', 'salmon', 'tan', 'plum', 'slate', 'charcoal']
        
        
        plt.plot(time_plot, object_position, label='object position', color='black')
        
        for fingerId in range(actuator_number):
            plt.plot(time_plot, gripper_position[fingerId], label=f'finger {fingerId+1}')
        plt.xlabel('time (s)')
        plt.ylabel('positions (mm)')
        plt.title('object and gripper positions')
        plt.legend()
        plt.show()


    # convert and scale all units to N, mm
    
    
    vertical_velocity = vertical_velocity * 1e3 / X  # convert to mm/step
    H = vertical_velocity * (upward_endtime - upward_starttime) / time_step # in mm, the height the gripper is moving up during the lifting phase
    
    object_velocity = object_position[1:] - object_position[:-1]
    object_velocity = np.insert(object_velocity, 0, 0)
    
    start_idx = int(upward_starttime/time_step) - 1
    end_idx = int(upward_endtime/time_step)-1
    lift_time = int((upward_endtime - upward_starttime)/time_step)
    
    grasping_mode = 0
    # if all variables are zero, the gripper is not even touching the object
    # sometimes the object_position is not zero, 
    # but the contact_normal and contact_friction are zero
    if np.mean(object_position) <= 0.1*H and np.mean(contact_normal) <= 0 and np.mean(contact_friction) <= 0:
        grasping_mode = 1 # the gripper is not even touching the object
        
    # if object_position is mostly zero, 
    # but the contact_forces are non-zero for a while, 
    # the gripper is touching the object but not grasping it
    if object_position[-1] <= 0.1*H and np.mean(contact_normal) > 0 and np.mean(contact_friction) > 0:
        grasping_mode = 2 # the gripper is touching the object but not grasping it
        
    # the gripper is grasping the object but also slipping 
    # (the reward should be proportional to the non-zero time of the object_position and contact_force)
    if object_position[-1] > 0.1*H and np.mean(object_velocity[start_idx:end_idx]) < vertical_velocity and np.mean(contact_normal[:, :, object_position.argmax()]) > 0 and np.mean(contact_friction[:, :, object_position.argmax()]) > 0:
        grasping_mode = 3 # the gripper is grasping the object but also slipping
        
    # if the gripper is applying a too large force to the object and the object is flying away
    if np.max(object_velocity) >= 10*vertical_velocity:
        grasping_mode = 4 
        
    # if object_position is stably increasing and then keeps constant, 
    # contact_force is non-zero for the whole time, 
    # the gripper is lifting the object successfully
    if object_position[-1] >= 0.5*H and np.mean(object_velocity[start_idx:end_idx]) >= vertical_velocity and np.max(object_velocity) < 10*vertical_velocity and np.mean(contact_normal[:, :, object_position.argmax()]) > 0 and np.mean(contact_friction[:, :, object_position.argmax()]) > 0:
        grasping_mode = 5 
        
    # compressed observation
    eps = 1e-9
    obj_z_max = float(np.max(object_position))
    obj_z_end = float(object_position[-1])
    obj_v_lift_mean = float(np.mean(np.diff(object_position[start_idx:end_idx], prepend=object_position[start_idx])))
    cn_sum_lift = float(np.sum(contact_normal[:, :, start_idx:end_idx]))
    cf_sum_lift = float(np.sum(contact_friction[:, :, start_idx:end_idx]))
    contact_link_eff = float(np.sum(contact_normal[:, :, end_idx] > 0)/(actuator_number*module_number))
    slip_metric = float(max(0.0, vertical_velocity - obj_v_lift_mean))
    slip_ratio = max(0.0, 1.0 - obj_v_lift_mean / (vertical_velocity + eps))

    obs_compact = dict(
        obj_z_max=obj_z_max,
        obj_z_end=obj_z_end,
        obj_v_lift_mean=obj_v_lift_mean,
        contact_normal_sum=cn_sum_lift,
        contact_friction_sum=cf_sum_lift,
        contact_link_eff=contact_link_eff,
        slip_ratio=slip_ratio,
        grasping_mode=int(grasping_mode),
    )
    print(
        "\n[Surrogate model] 🔩 predicted parameters:"
        f"\n   ├─ m_act            = {m_act: .6f}"
        f"\n   ├─ b_act            = {b_act: .6f}"
        f"\n   ├─ m_ela            = {m_ela: .6f}"
        f"\n   ├─ b_ela            = {b_ela: .6f}"
        f"\n   └─ k_torque_theta0  = {k_torque_theta0: .6f}\n"
    )
    
    if return_traces:
        return obs_compact, object_position, contact_normal[:, :, int(cycle/time_step)-1:-1], contact_friction[:, :, int(cycle/time_step)-1:-1]
    else:
        return obs_compact


# RL optimized parameters for different objects
