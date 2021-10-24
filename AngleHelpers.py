'''
Authors: Stefan Balle & Lucas Essmann 
2021 
'''

import pandas as pd 
import numpy as np 
from tqdm import tqdm


def calculate_norm_dir_from_unity_angles(unity_angle_x,unity_angle_y,unity_angle_z):
    '''
    As suggested by 
    https://forum.unity.com/threads/convert-a-vector3-containing-eulerangles-into-a-normalized-direction-vector3.124171/#post-846164
    Inversion of the x angle added for the calculation of the elevation in order to produce correct results that
    match the results of Unity's built-in method.

    Return a Unity unit direction vector.
    '''

    elevation = np.deg2rad( (-1) * unity_angle_x) # (-1) multiplication added here in comparison to original code
    heading = np.deg2rad(unity_angle_y)
    dir_vec = (np.cos(elevation) * np.sin(heading), np.sin(elevation), np.cos(elevation) * np.cos(heading))
    dir_norm = dir_vec/np.linalg.norm(dir_vec)

    return dir_norm 


def calculate_thetas_from_unity_coords(dir_vec,ret_degree=True):
    '''
    Calculates the horizontal and vertical angles of the direction vector. 
    Coordinate system: X to the right, Y up, Z forward.    
    Refers to Unity coordinates.
    '''
    
    theta_horizontal = np.arctan2(dir_vec[0], dir_vec[2])
    theta_vertical = np.arctan2(dir_vec[1], dir_vec[2])
    
    if ret_degree:
        theta_horizontal = theta_horizontal / 2 / np.pi * 360
        theta_vertical = theta_vertical / 2 / np.pi * 360
    
    return theta_horizontal, theta_vertical 


def calculate_rotation_matrix(rotate_around_axis, angle, angle_in_degree=True):
    '''
    Calculate a rotation matrix around a specified axis.
    Possible values for rotate_around_axis: 'x', 'y', 'z'. 
    Refers to a right-handed cartesian coordinate system. 
    
    https://en.wikipedia.org/wiki/Rotation_matrix 
    '''
    
    rot_mat = None
     
    if rotate_around_axis.lower() == 'x': 
        theta_x = angle
        if angle_in_degree:
            theta_x = theta_x / 360 * np.pi * 2 
        rot_mat_x = np.array([
            [1,0,0],
            [0,np.cos(theta_x),-np.sin(theta_x)],
            [0,np.sin(theta_x),np.cos(theta_x)]
        ])
        rot_mat = rot_mat_x
    
    elif rotate_around_axis.lower() == 'y':
        theta_y = angle
        if angle_in_degree:
            theta_y = theta_y / 360 * np.pi * 2 
        rot_mat_y = np.array([
            [np.cos(theta_y),0,np.sin(theta_y)],
            [0,1,0],
            [-np.sin(theta_y),0,np.cos(theta_y)]
        ])
        rot_mat = rot_mat_y
        
    elif rotate_around_axis.lower() == 'z':  
        theta_z = angle
        if angle_in_degree:
            theta_z = theta_z / 360 * np.pi * 2 
        rot_mat_z = np.array([
            [np.cos(theta_z),-np.sin(theta_z),0],
            [np.sin(theta_z),np.cos(theta_z),0],
            [0,0,1]
        ])
        rot_mat = rot_mat_z
        
    else:
        print("Incorrect axis specified!")
    
    return rot_mat

def unity_pts_to_right_handed_cartesian_coords(pts):
    '''
    Transforms multiple points in Unity coordinates to
    right-handed cartesian coordinates.
    '''
    
    transformed_pts = []
    
    for elem in pts:
        transformed_pts.append([elem[0],elem[2],elem[1]]) # y and z are switched
    
    return transformed_pts 

def right_handed_cartesian_coords_to_unity_pts(pts):
    '''
    Transforms multiple points in right-handed cartesian 
    to Unity coordinates.
    '''
    
    transformed_pts = []
    
    for elem in pts:
        transformed_pts.append([elem[0],elem[2],elem[1]]) # z and y are switched
    
    return transformed_pts 
    
    
def angle_between(v1, v2, degree=True):
    
    v1_u = v1 / np.linalg.norm(v1) # unit vector
    v2_u = v2 / np.linalg.norm(v2) # unit vector
    
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    if degree:
        angle = angle / 2 / np.pi * 360
    
    return angle   


def angle_in_180s(angle):
    
    angle = angle % 360
    
    if angle > 180:
        angle = -1 * (360 - angle)
    elif angle < -180:
        angle = 360 + angle
    
    return angle

    
def cart2sph(vec):
    '''
    Cartesian coordinates to spherical coordinates.
    https://github.com/numpy/numpy/issues/5228#issue-46746558
    '''
    x = vec[0]
    y = vec[1]
    z = vec[2]
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def sph2cart(az, el, r):
    '''
    Spherical coordinates to cartesian coordinates.
    Azimuth on groundplane, elevation from groundplane to up-axis. 
    https://github.com/numpy/numpy/issues/5228#issue-46746558
    '''
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def deg2rad(angle):
    return angle / 360 * 2* np.pi

def rad2deg(angle):
    return angle / 2 / np.pi * 360 







def create_relative_direction(inp_dir_unity,ref_dir_unity,method="anglediff_sphere_coords",log_verbose=False):
    '''
    Mathematically correct, but does not consider roll of reference!
    
    Calculate a direction vector that has the same offset from the Unity forward vector as an input direction vector has from a reference direction vector.
    inp_dir_unity: Vector of actual direction data.
    ref_dir_unity: Vector of reference the actual data should be taken relative to. 
    log_verbose: Log debug information.
    
    Methods: 
    - "anglediff": Take the difference between input and reference vector for horizontal angle and elevation angle 
                   and apply that difference to the Unity forward vector.
    - "anglediff_sphere_coords" : Same as "anglediff", but with sphere coordinates.
    - "unit_sphere_rotation": Rotate the unit sphere the directions are located in such that the reference vector will point 
                              towards the Unity forward vector.
    
    Due to the reference vector being rotatable around itself when pointing in the direction of the forward vector, 
    still yielding a valid solution, different final results exist. For the purpose of Unity directions, the "anglediff"/ "anglediff_sphere_coords" 
    approach seems to be most useful.
    
    The resulting relative direction vector will be in Unity coordinates. 
    '''
    
    # make sure np.array exists 
    inp_dir_unity = np.array(inp_dir_unity)
    ref_dir_unity = np.array(ref_dir_unity)
    forward_dir_unity = np.array([0,0,1])
    
    # To cartesian from Unity 
    inp_dir_cartesian = unity_pts_to_right_handed_cartesian_coords([inp_dir_unity])[0]
    ref_dir_cartesian = unity_pts_to_right_handed_cartesian_coords([ref_dir_unity])[0]
    forward_dir_cartesian = unity_pts_to_right_handed_cartesian_coords([forward_dir_unity])[0]

    # Debug Info
    if log_verbose:
        print("Method: " + method)
    
    if method == "anglediff":
        
        # Calculate thetas, can only use horizontal, since vertical depends on different plane than projected one 
        inp_horizontal, _ = calculate_thetas_from_unity_coords(inp_dir_unity,ret_degree=True)
        ref_horizontal, _ = calculate_thetas_from_unity_coords(ref_dir_unity,ret_degree=True)

        # Calculate vertical angle offset as angle between direction vector and projection on x,y plane (cartesian)
        inp_vertical = angle_between(inp_dir_cartesian,[inp_dir_cartesian[0],inp_dir_cartesian[1],0])
        ref_vertical = angle_between(ref_dir_cartesian,[ref_dir_cartesian[0],ref_dir_cartesian[1],0])

        # if z (cartesian) is negative, vertical angle is negative 
        if inp_dir_cartesian[2] < 0:
            inp_vertical *= -1
        if ref_dir_cartesian[2] < 0:
            ref_vertical *= -1

        # Calculate angle offsets 
        offset_horizontal = inp_horizontal - ref_horizontal
        offset_vertical = inp_vertical - ref_vertical

        # Debug Info 
        if log_verbose:
            print("Input horizontal: " + str(inp_horizontal))
            print("Reference horizontal: " + str(ref_horizontal))
            print("Offset horizontal: " + str(offset_horizontal)) # rotate around Unity y, Right-Handed cartesian z
            print("Input vertical: " + str(inp_vertical))
            print("Reference vertical: " + str(ref_vertical))
            print("Offset vertical: " + str(offset_vertical)) # rotate around Unity/ Right-Handed cartesian x 

        # calculate rot matrices
        # axis are for right-handed coordinate system
        # angle inverted for horizontal
        horizontal_rot_mat = calculate_rotation_matrix(rotate_around_axis="z", angle= -1 * offset_horizontal, angle_in_degree=True)
        vertical_rot_mat = calculate_rotation_matrix(rotate_around_axis="x", angle=offset_vertical, angle_in_degree=True)

        # apply calculated relative rotations to forward Unity vector (0,0,1) (in right-handed cartesian (0,1,0)) to get final direction 
        # first vertical, then horizontal! Rotation axis must align 
        final_dir_unity = forward_dir_unity
        final_dir_cartesian = unity_pts_to_right_handed_cartesian_coords([final_dir_unity])[0]
        final_dir_cartesian = np.dot(vertical_rot_mat, final_dir_cartesian)
        final_dir_cartesian = np.dot(horizontal_rot_mat,final_dir_cartesian)

        # Verify final direction has offset values 
        final_vertical = angle_between(final_dir_cartesian,[final_dir_cartesian[0],final_dir_cartesian[1],0]) # angle between vec and projection on xy-plane (cartesian)
        final_horizontal = angle_between([final_dir_cartesian[0],final_dir_cartesian[1],0],[0,1,0]) # angle between projection on xy-plane and forward vector (cartesian)
       
        # Debug Info 
        if log_verbose:
            print("---")
            print("Final horizontal (should match offset horizontal): " + str(final_horizontal)) 
            print("Final vertical (should match offset vertical): " + str(final_vertical))




    elif method == "anglediff_sphere_coords":
       
        # compute spherical coordinates 
        inp_dir_spherical = cart2sph(inp_dir_cartesian)
        ref_dir_spherical = cart2sph(ref_dir_cartesian) 
        forward_dir_spherical = cart2sph(forward_dir_cartesian)

        # compute difference between ref and input, modulo and add to forward dir 
        final_dir_spherical = (inp_dir_spherical[0] - ref_dir_spherical[0], inp_dir_spherical[1] - ref_dir_spherical[1], 1)
        final_dir_spherical = (final_dir_spherical[0] % (np.pi * 2), final_dir_spherical[1] % (np.pi * 2), 1)
        final_dir_spherical = (final_dir_spherical[0] + forward_dir_spherical[0], final_dir_spherical[1] + forward_dir_spherical[1],1)
        final_dir_spherical = (final_dir_spherical[0] % (np.pi * 2), final_dir_spherical[1] % (np.pi * 2), 1)

        # convert final direction back to cartesian coordinates
        final_dir_cartesian = sph2cart(*final_dir_spherical)

        # Debug Info
        if log_verbose:

            # verify vectors and directions 
            inp_vert_angle = angle_between(inp_dir_cartesian,[inp_dir_cartesian[0],inp_dir_cartesian[1],0]) # angle between vec and projection on xy-plane (cartesian)
            inp_horiz_angle = angle_between([inp_dir_cartesian[0],inp_dir_cartesian[1],0],[0,1,0]) # angle between projection on xy-plane and forward vector (cartesian)

            ref_vert_angle = angle_between(ref_dir_cartesian,[ref_dir_cartesian[0],ref_dir_cartesian[1],0]) # angle between vec and projection on xy-plane (cartesian)
            ref_horiz_angle = angle_between([ref_dir_cartesian[0],ref_dir_cartesian[1],0],[0,1,0]) # angle between projection on xy-plane and forward vector (cartesian)

            final_vert_angle = angle_between(final_dir_cartesian,[final_dir_cartesian[0],final_dir_cartesian[1],0]) # angle between vec and projection on xy-plane (cartesian)
            final_horiz_angle = angle_between([final_dir_cartesian[0],final_dir_cartesian[1],0],[0,1,0]) # angle between projection on xy-plane and forward vector (cartesian)

            # if z (cartesian) is negative, vertical angle is negative 
            if inp_dir_cartesian[2] < 0:
                inp_vert_angle *= -1
            if ref_dir_cartesian[2] < 0:
                ref_vert_angle *= -1
            if final_dir_cartesian[2] < 0:
                final_vert_angle *= -1

            # if x (cartesian) is negative, horizontal angle is negative 
            if inp_dir_cartesian[0] < 0:
                inp_horiz_angle *= -1
            if ref_dir_cartesian[0] < 0:
                ref_horiz_angle *= -1
            if final_dir_cartesian[0] < 0:
                final_horiz_angle *= -1

            # print verification infos 
            print("Horizontal angles:")
            print("\tInput: " + str(inp_horiz_angle))
            print("\tReference: " + str(ref_horiz_angle))
            print("\tFinal: " + str(final_horiz_angle))
            print("Vertical angles:")
            print("\tInput: " + str(inp_vert_angle))
            print("\tReference:" + str(ref_vert_angle))
            print("\tFinal: " + str(final_vert_angle))
            print("Vectors:")
            print("\tInput: " + str(inp_dir_cartesian))
            print("\tReference: " + str(ref_dir_cartesian))
            print("\tFinal: " + str(final_dir_cartesian))


        
    
    elif method == "unit_sphere_rotation":

        # Calculate horizontal angle offset as angle between direction vector's projection on x,y plane (cartesian) and forward vector
        # Alternatively, use calculate_thetas_from_unity_coords (but can only use horizontal value here, because projection would skew vertical value)
        ref_horizontal = angle_between([ref_dir_cartesian[0],ref_dir_cartesian[1],0],[0,1,0])

        # Calculate vertical angle offset as angle between direction vector and projection on x,y plane (cartesian)
        ref_vertical = angle_between(ref_dir_cartesian,[ref_dir_cartesian[0],ref_dir_cartesian[1],0])

        # if z (cartesian) is negative, vertical angle is negative 
        if ref_dir_cartesian[2] < 0:
            ref_vertical *= -1

        # if x (cartesian) is negative, horizontal angle is negative 
        if ref_dir_cartesian[0] < 0:
            ref_horizontal *= -1

        # Calculate horizontal angle offset for input 
        inp_horizontal = angle_between([inp_dir_cartesian[0],inp_dir_cartesian[1],0],[0,1,0])

        # Calculate vertical angle offset for input
        inp_vertical = angle_between(inp_dir_cartesian,[inp_dir_cartesian[0],inp_dir_cartesian[1],0])

        # if z (cartesian) is negative, vertical angle is negative 
        if inp_dir_cartesian[2] < 0:
            inp_vertical *= -1

        # if x (cartesian) is negative, horizontal angle is negative 
        if inp_dir_cartesian[0] < 0:
            inp_horizontal *= -1

        # Debug Info
        if log_verbose:
            print("Reference horizontal: " + str(ref_horizontal))
            print("Reference vertical: " + str(ref_vertical))
            print("Input horizontal: " + str(inp_horizontal))
            print("Input vertical: " + str(inp_vertical))

        # calculate rot matrices
        # axis are for right-handed coordinate system
        # angle inverted for vertical
        horizontal_rot_mat = calculate_rotation_matrix(rotate_around_axis="z", angle= ref_horizontal, angle_in_degree=True)
        vertical_rot_mat = calculate_rotation_matrix(rotate_around_axis="x", angle= -1 *ref_vertical, angle_in_degree=True)

        # Apply calculated rotations to reference direction, in order to rotate it to forward direction (Unity 0,0,1)
        # First horizontal, then vertical! Rotation axis must align 
        rotated_ref = ref_dir_cartesian.copy()
        rotated_ref = np.dot(horizontal_rot_mat,rotated_ref)
        rotated_ref = np.dot(vertical_rot_mat, rotated_ref)
        
        # Debug Info 
        if log_verbose:
            
            # Calculate horizontal angle offset of rotated reference
            rotated_ref_horizontal = angle_between([rotated_ref[0],rotated_ref[1],0],[0,1,0])

            # Calculate vertical angle offset of rotated reference
            rotated_ref_vertical = angle_between(rotated_ref,[rotated_ref[0],rotated_ref[1],0])

            # if z (cartesian) is negative, vertical angle is negative 
            if rotated_ref[2] < 0:
                rotated_ref_vertical *= -1

            # if x (cartesian) is negative, horizontal angle is negative 
            if ref_dir_cartesian[0] < 0:
                rotated_ref_horizontal *= -1

            print("Rotated reference: " + str(rotated_ref))
            print("Rotated reference horizontal: " + str(rotated_ref_horizontal))
            print("Rotated reference vertical: " + str(rotated_ref_vertical))


        # Apply calculated rotations to input direction
        # Same order, as for reference. First horizontal, then vertical! Rotation axis must align 
        rotated_inp = inp_dir_cartesian.copy()
        rotated_inp = np.dot(horizontal_rot_mat,rotated_inp)
        rotated_inp = np.dot(vertical_rot_mat, rotated_inp)

        # Debug Info
        if log_verbose:

            # Calculate horizontal angle offset of rotated input
            rotated_inp_horizontal = angle_between([rotated_inp[0],rotated_inp[1],0],[0,1,0])

            # Calculate vertical angle offset of rotated input
            rotated_inp_vertical = angle_between(rotated_inp,[rotated_inp[0],rotated_inp[1],0])

            # if z (cartesian) is negative, vertical angle is negative 
            if rotated_inp[2] < 0:
                rotated_inp_vertical *= -1

            # if x (cartesian) is negative, horizontal angle is negative 
            if rotated_inp[0] < 0:
                rotated_inp_horizontal *= -1

            print("Rotated input: " + str(rotated_inp))
            print("Rotated input horizontal: " + str(rotated_inp_horizontal))
            print("Rotated input vertical: " + str(rotated_inp_vertical))

        
        # Copy result to variable of same name as in other cases 
        final_dir_cartesian = rotated_inp
    
    else:
        print("Specified unknown method!")
        return
        
    
    # Make result a unit vector 
    final_dir_cartesian = final_dir_cartesian / np.linalg.norm(final_dir_cartesian)
    
    # Transform result to Unity coordinates 
    final_dir_unity = right_handed_cartesian_coords_to_unity_pts([final_dir_cartesian])[0]
   
    # log verbose information
    if log_verbose:
        print("Final direction vector: " + str(final_dir_unity))

    return final_dir_unity

   


def create_relative_directions(inp_unity_x,inp_unity_y,inp_unity_z,ref_unity_x,ref_unity_y,ref_unity_z,method="anglediff_sphere_coords"):
    '''
    Mathematically correct, but does not consider roll of reference!
    
    Creates for a Pandas Series of Unity x,y,z inputs and references directions that are 
    relative to the Unity forward vector. 
    Methods: 
    - "anglediff"
    - "anglediff_sphere_coords" 
    - "unit_sphere_rotation"
    See create_relative_direction() for more infos. 
    
    Return Unity x,y,z series for the relative directions.
    '''
    
    print("Creating relative directions, method: " + str(method))
    
    # Progress bar 
    tqdm.pandas()
    
    # Create dataframe 
    dataframe = pd.DataFrame(columns=["inp_x","inp_y","inp_z","ref_x","ref_y","ref_z","final_x","final_y","final_z"])

    # Set data
    dataframe["inp_x"] = inp_unity_x
    dataframe["inp_y"] = inp_unity_y
    dataframe["inp_z"] = inp_unity_z
    dataframe["ref_x"] = ref_unity_x
    dataframe["ref_y"] = ref_unity_y
    dataframe["ref_z"] = ref_unity_z
    
    # Init final columns
    dataframe["final_x"] = np.NAN
    dataframe["final_y"] = np.NAN
    dataframe["final_z"] = np.NAN

    # Define method that is to be applied per row 
    def apply_dir_calc(arg):

        # Calculate relative direction 
        result = create_relative_direction((arg["inp_x"],arg["inp_y"],arg["inp_z"]),(arg["ref_x"],arg["ref_y"],arg["ref_z"]),method,log_verbose=False)

        # Add result to row of dataframe
        arg["final_x"] = result[0]
        arg["final_y"] = result[1]
        arg["final_z"] = result[2]
        
        return arg


    # Apply transform 
    dataframe = dataframe.progress_apply(lambda row: apply_dir_calc(row), axis=1)
    
    return dataframe["final_x"], dataframe["final_y"], dataframe["final_z"]
    
    
def create_relative_direction_consider_roll(inp_dir_unity,ref_angles_unity,log_verbose=False):
    '''
    Rotate an input vector in the reverse order of angles that were used to create a reference vector from the Unity forward vector. 
    This will effectively result in (the virtual) reference vector being rotated (by the same rotations) to the Unity forward vector.
    And the input vector will be relative to the reference vector.
    The rotations consider yaw, pitch and roll of the reference vector. 

    In detail:
    Take Unity angles (order of original application is z, x, y) that lead to the creation of the reference direction.
    Use these angles to create rotation matrices that inverse those rotations. 
    Apply the same inversion to the input vector.  

    The resulting relative direction vector will be in Unity coordinates. 
    '''

    # Init 
    ref_dir_unity = None
    ref_angles_unity = np.array(ref_angles_unity)
    inp_dir_cartesian = unity_pts_to_right_handed_cartesian_coords([inp_dir_unity])[0]

    # Log verbose information         
    if log_verbose:

        # Calculate ref dir from ref angles 
        ref_dir_unity = calculate_norm_dir_from_unity_angles(ref_angles_unity[0],ref_angles_unity[1],ref_angles_unity[2])

        # To cartesian from Unity 
        ref_dir_cartesian = unity_pts_to_right_handed_cartesian_coords([ref_dir_unity])[0]

        # Calculate horizontal angle offset as angle between direction vector's projection on x,y plane (cartesian) and forward vector
        # Alternatively, use calculate_thetas_from_unity_coords (but can only use horizontal value here, because projection would skew vertical value)
        ref_horizontal = angle_between([ref_dir_cartesian[0],ref_dir_cartesian[1],0],[0,1,0])

        # Calculate vertical angle offset as angle between direction vector and projection on x,y plane (cartesian)
        ref_vertical = angle_between(ref_dir_cartesian,[ref_dir_cartesian[0],ref_dir_cartesian[1],0])

        # if z (cartesian) is negative, vertical angle is negative 
        if ref_dir_cartesian[2] < 0:
            ref_vertical *= -1
            
        # if x (cartesian) is negative, horizontal angle is negative 
        if ref_dir_cartesian[0] < 0:
            ref_horizontal *= -1
            
        # Calculate horizontal angle offset for input 
        inp_horizontal = angle_between([inp_dir_cartesian[0],inp_dir_cartesian[1],0],[0,1,0])

        # Calculate vertical angle offset for input
        inp_vertical = angle_between(inp_dir_cartesian,[inp_dir_cartesian[0],inp_dir_cartesian[1],0])

        # if z (cartesian) is negative, vertical angle is negative 
        if inp_dir_cartesian[2] < 0:
            inp_vertical *= -1
            
        # if x (cartesian) is negative, horizontal angle is negative 
        if inp_dir_cartesian[0] < 0:
            inp_horizontal *= -1

        # Log 
        print("Reference horizontal: " + str(ref_horizontal))
        print("Reference vertical: " + str(ref_vertical))
        print("Input horizontal: " + str(inp_horizontal))
        print("Input vertical: " + str(inp_vertical))


    # Calculate rot matrices, Axis are for right-handed coordinate system
    # Unity original order of applied angles: z, x, y 
    # Inverse Unity order: y, x, z 
    # Inverse Unity order in cartesian: z, x, y 
    
    horizontal_angle = ref_angles_unity[1] # unity y
    vertical_angle = ref_angles_unity[0] # unity x
    around_itself_angle = ref_angles_unity[2] # unity z
    horizontal_rot_mat = calculate_rotation_matrix(rotate_around_axis="z", angle = horizontal_angle, angle_in_degree=True)
    vertical_rot_mat = calculate_rotation_matrix(rotate_around_axis="x", angle = vertical_angle, angle_in_degree=True)
    around_itself_rot_mat = calculate_rotation_matrix(rotate_around_axis="y", angle = around_itself_angle, angle_in_degree=True)
    
    # log verbose information
    if log_verbose:

        # Apply calculated rotations to reference direction, in order to rotate it to forward direction (Unity 0,0,1)
        # Keep order of rotations!
        rotated_ref = ref_dir_cartesian.copy()
        rotated_ref = np.dot(horizontal_rot_mat,rotated_ref)
        rotated_ref = np.dot(vertical_rot_mat, rotated_ref)
        rotated_ref = np.dot(around_itself_rot_mat, rotated_ref)

        # Calculate horizontal angle offset of rotated reference
        rotated_ref_horizontal = angle_between([rotated_ref[0],rotated_ref[1],0],[0,1,0])

        # Calculate vertical angle offset of rotated reference
        rotated_ref_vertical = angle_between(rotated_ref,[rotated_ref[0],rotated_ref[1],0])

        # if z (cartesian) is negative, vertical angle is negative 
        if rotated_ref[2] < 0:
            rotated_ref_vertical *= -1
            
        # if x (cartesian) is negative, horizontal angle is negative 
        if ref_dir_cartesian[0] < 0:
            rotated_ref_horizontal *= -1

        print("Rotated reference: " + str(rotated_ref))
        print("Rotated reference horizontal: " + str(rotated_ref_horizontal))
        print("Rotated reference vertical: " + str(rotated_ref_vertical))



    # Apply calculated rotations to input direction
    # Keep order of rotation!
    rotated_inp = inp_dir_cartesian.copy()
    rotated_inp = np.dot(horizontal_rot_mat,rotated_inp)
    rotated_inp = np.dot(vertical_rot_mat, rotated_inp)
    rotated_inp = np.dot(around_itself_rot_mat, rotated_inp)

    # log verbose information
    if log_verbose:

        # Calculate horizontal angle offset of rotated input
        rotated_inp_horizontal = angle_between([rotated_inp[0],rotated_inp[1],0],[0,1,0])

        # Calculate vertical angle offset of rotated input
        rotated_inp_vertical = angle_between(rotated_inp,[rotated_inp[0],rotated_inp[1],0])

        # if z (cartesian) is negative, vertical angle is negative 
        if rotated_inp[2] < 0:
            rotated_inp_vertical *= -1
            
        # if x (cartesian) is negative, horizontal angle is negative 
        if rotated_inp[0] < 0:
            rotated_inp_horizontal *= -1

        print("Rotated input: " + str(rotated_inp))
        print("Rotated input horizontal: " + str(rotated_inp_horizontal))
        print("Rotated input vertical: " + str(rotated_inp_vertical))


    # Make result a unit vector 
    final_dir_cartesian = rotated_inp / np.linalg.norm(rotated_inp)
    
    # Transform result to Unity coordinates 
    final_dir_unity = right_handed_cartesian_coords_to_unity_pts([final_dir_cartesian])[0]

    # log verbose information
    if log_verbose:
        print("Final direction vector: " + str(final_dir_unity))

    return final_dir_unity


def create_relative_directions_consider_roll(inp_unity_x,inp_unity_y,inp_unity_z,ref_angle_unity_x,ref_angle_unity_y,ref_angle_unity_z):
    '''
    
    Creates for a Pandas Series of Unity x,y,z input directions and Unity reference angles x,y,z
    a Pandas Series of Unity x,y,z directions that are relative to (the virtual) reference vector 
    (as would be constructed by rotation from the Unity forward vector with the provided angles). 

    See create_relative_direction_consider_roll() for more infos. 
    
    Return Unity x,y,z series for the relative directions.
    '''
    
    print("Creating relative directions considering roll.")
    
    # Progress bar 
    tqdm.pandas()
    
    # Create dataframe 
    dataframe = pd.DataFrame(columns=["inp_x","inp_y","inp_z","ref_angle_x","ref_angle_y","ref_angle_z","final_x","final_y","final_z"])

    # Set data
    dataframe["inp_x"] = inp_unity_x
    dataframe["inp_y"] = inp_unity_y
    dataframe["inp_z"] = inp_unity_z
    dataframe["ref_angle_x"] = ref_angle_unity_x
    dataframe["ref_angle_y"] = ref_angle_unity_y
    dataframe["ref_angle_z"] = ref_angle_unity_z
    
    # Init final columns
    dataframe["final_x"] = np.NAN
    dataframe["final_y"] = np.NAN
    dataframe["final_z"] = np.NAN

    # Define method that is to be applied per row 
    def apply_dir_calc(arg):

        # Calculate relative direction 
        result = create_relative_direction_consider_roll((arg["inp_x"],arg["inp_y"],arg["inp_z"]),(arg["ref_angle_x"],arg["ref_angle_y"],arg["ref_angle_z"]),log_verbose=False)

        # Add result to row of dataframe
        arg["final_x"] = result[0]
        arg["final_y"] = result[1]
        arg["final_z"] = result[2]
        
        return arg

    # Apply transform 
    dataframe = dataframe.progress_apply(lambda row: apply_dir_calc(row), axis=1)
    
    return dataframe["final_x"], dataframe["final_y"], dataframe["final_z"]
    