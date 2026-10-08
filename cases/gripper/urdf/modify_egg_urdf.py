"""Configure the egg URDF for a selected load and simulation scale.

Updates mesh filenames, scale, mass and inertia in a URDF template."""

import os
import xml.etree.ElementTree as ET
import math

def modify_egg_urdf(template_path, output_path, weight_num, X):
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
        
        # Update mesh filename to use egg_{weight_num}_w.stl
        # Get the directory of the original mesh file
        old_filename = mesh.get('filename')
        old_dir = os.path.dirname(old_filename)
        
        # Build new filename
        if weight_num == 0:
            new_filename = os.path.join(old_dir, "egg.stl")
        else:
            new_filename = os.path.join(old_dir, f"egg_{weight_num}_w.stl")
        # Replace backslashes for URDF compatibility (always use forward slashes)
        new_filename = new_filename.replace('\\', '/')
        mesh.set('filename', new_filename)

    # Update the mass value
    mass = 0.05 + 0.02*weight_num # kg
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

        sld_Ixx = 21357.40*1e-9 # kg m^2
        sld_Iyy = 23069.27*1e-9 # kg m^2
        sld_Izz = 23069.27*1e-9 # kg m^2
    elif weight_num == 1:
        sld_Ixx = 22989.22*1e-9
        sld_Iyy = 23926.85*1e-9
        sld_Izz = 23996.03*1e-9
    elif weight_num == 2:
        sld_Ixx = 24619.69*1e-9
        sld_Iyy = 25168.79*1e-9
        sld_Izz = 25308.51*1e-9
    elif weight_num == 3:
        sld_Ixx = 26248.67*1e-9
        sld_Iyy = 27598.09*1e-9
        sld_Izz = 27809.85*1e-9
    elif weight_num == 4:
        sld_Ixx = 27878.34*1e-9
        sld_Iyy = 31737.82*1e-9
        sld_Izz = 32020.92*1e-9
    elif weight_num == 5:
        sld_Ixx = 29508.20*1e-9
        sld_Iyy = 38017.73*1e-9
        sld_Izz = 38371.98*1e-9
    elif weight_num == 6:
        sld_Ixx = 31138.05*1e-9
        sld_Iyy = 46829.14*1e-9
        sld_Izz = 47254.55*1e-9
    elif weight_num == 7:
        sld_Ixx = 32767.87*1e-9
        sld_Iyy = 58545.18*1e-9
        sld_Izz = 59041.78*1e-9
    Ixx = sld_Ixx * X * X
    Iyy = sld_Iyy * X * X
    Izz = sld_Izz * X * X
    
    return Ixx, Iyy, Izz