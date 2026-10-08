"""Run repeated gripper simulation trials and export grasp metrics.

Calls the grasp simulator across repeated trials and stores task outcomes
as CSV files for sim-to-real analysis."""

from run_grasp_simulation import gripper_simulator
import numpy as np
import matplotlib.pyplot as plt
import os
def grasping_sim_log(init_grasp_height, maximum_pressure, grasp_radius_distance, inclination_angle, inner_radius, wall_thickness, average_radius, module_length, module_number, actuator_number, object_type, extra_weight, weight_num, add_noise):
    metadata = []
    metadata_header = ['trial_num', 'sim_time (s)', 'maxP (kPa)', 'frictionCoefficient', 'objectX (mm)', 'objectY (mm)', 'maxZ (mm)', 'lastZ (mm)']
    
    for trial_num in range(1, 30+1):  # Run 2 trials
        output = gripper_simulator(
            init_grasp_height=init_grasp_height,
            maximum_pressure=maximum_pressure,
            grasp_radius_distance=grasp_radius_distance,
            inclination_angle=inclination_angle,
            inner_radius=inner_radius,
            wall_thickness=wall_thickness,
            average_radius=average_radius,
            module_length=module_length,
            module_number=module_number,
            actuator_number=actuator_number,
            object_type=object_type,
            extra_weight=extra_weight,
            weight_num=weight_num,
            add_noise=add_noise
        )
        sim_time, maxP, frictionCoefficient, objectX, objectY, maxZ, lastZ = output
        datarow = [trial_num, sim_time, maxP, frictionCoefficient, objectX, objectY, maxZ, lastZ]
        metadata.append(datarow)
    metadata = np.array(metadata)
    # Save metadata to a CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "mid_air_sim_results")
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{object_type}_{maximum_pressure}_{weight_num}.csv"
    file_path = os.path.join(results_dir, filename)
    np.savetxt(file_path, metadata, delimiter=",", header=",".join(metadata_header), comments='')


init_grasp_height = 8 #mm
maximum_pressure = 13 #kPa
inclination_angle = np.pi/6
inner_radius = 2 #mm
wall_thickness = 1.5 #mm
average_radius = 6 #mm
module_length = 11 #mm 
module_number = 6 #number of modules 
actuator_number = 3 #number of actuators
object_type = "cylinder" 

if object_type == "egg":
    grasp_radius_distance = -16 
elif object_type == "cylinder":
    grasp_radius_distance = -2

extra_weight = True # True or False
weight_num = 4
add_noise = True # True or False, add noise to the simulation

grasping_sim_log(init_grasp_height, maximum_pressure, grasp_radius_distance, inclination_angle, inner_radius, wall_thickness, average_radius, module_length, module_number, actuator_number, object_type, extra_weight, weight_num, add_noise)