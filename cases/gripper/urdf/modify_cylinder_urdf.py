"""Configure the cylinder URDF for a selected load and simulation scale.

Updates mesh filenames, scale, mass and inertia in a URDF template."""

import os
import xml.etree.ElementTree as ET
import math

def modify_cylinder_urdf(template_path, output_path, weight_num, X):
    # Load the URDF template
    current_path = os.path.dirname(__file__)  # .../Gripper/urdf

    if not os.path.isabs(template_path):
        template_path = os.path.join(current_path, template_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(current_path, output_path)
    tree = ET.parse(template_path)
    root = tree.getroot()

    scale = X/1000 
    # Convert scale to string format for the mesh element
    scale_str = f"{scale} {scale} {scale}"

    # Update the scale for the mesh in visual and collision tags
    for mesh in root.iter('mesh'):
        mesh.set('scale', scale_str)
        
        # Get the directory of the original mesh file
        old_filename = mesh.get('filename')
        old_dir = os.path.dirname(old_filename)
        
        # Build new filename
        if weight_num == 0:
            new_filename = os.path.join(old_dir, "cylinder.stl")
        else:
            new_filename = os.path.join(old_dir, f"cylinder_{weight_num}_w.stl")
        # Replace backslashes for URDF compatibility (always use forward slashes)
        new_filename = new_filename.replace('\\', '/')
        mesh.set('filename', new_filename)

    # Update the mass value
    mass = 9.8*1e-3 + 0.02*weight_num # kg
    for mass_tag in root.iter('mass'):
        mass_tag.set('value', str(mass))
    
    # Update the inertia values
    inertia_values = calculate_inertia(weight_num, X)
    for inertia in root.iter('inertia'):
        inertia.set('ixx', str(inertia_values[0]))
        inertia.set('iyy', str(inertia_values[1]))
        inertia.set('izz', str(inertia_values[2]))
        inertia.set('ixy', "0.0")
        inertia.set('ixz', "0.0")
        inertia.set('iyz', "0.0")

    # Write the modified URDF to the output path
    tree.write(output_path)

def calculate_inertia(weight_num, X):
    # solidworks values
    
    # inertia values wrt the center of mass
    if weight_num == 0:

        sld_Ixx = 1628.29*1e-9 # kg m^2
        sld_Iyy = 1805.78*1e-9 # kg m^2
        sld_Izz = 1805.78*1e-9 # kg m^2
    elif weight_num == 1:
        sld_Ixx = 3216.81*1e-9
        sld_Iyy = 3794.86*1e-9
        sld_Izz = 3907.35*1e-9
    elif weight_num == 2:
        sld_Ixx = 4841.73*1e-9
        sld_Iyy = 5876.64*1e-9
        sld_Izz = 6065.22*1e-9
    elif weight_num == 3:
        sld_Ixx = 6501.89*1e-9
        sld_Iyy = 8765.38*1e-9
        sld_Izz = 8994.82*1e-9
    elif weight_num == 4:
        sld_Ixx = 8156.58*1e-9
        sld_Iyy = 12849.15*1e-9
        sld_Izz = 13124.90*1e-9
    elif weight_num == 5:
        sld_Ixx = 9802.00*1e-9
        sld_Iyy = 18486.88*1e-9
        sld_Izz = 18818.22*1e-9

    Ixx = sld_Ixx * X * X
    Iyy = sld_Iyy * X * X
    Izz = sld_Izz * X * X
    
    return Ixx, Iyy, Izz