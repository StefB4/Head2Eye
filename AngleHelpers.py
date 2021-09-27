'''
Authors: Stefan Balle & Lucas Essmann 
2021 
'''

import pandas as pd 
import numpy as np 


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






