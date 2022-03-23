'''
Authors: Stefan Balle & Lucas Essmann 
2021 
'''

import pandas as pd 
import numpy as np 
import pickle 
import json 
import glob 
import os 
from scipy.spatial.transform import Rotation as R

def read_normalized_json_to_df(filepath):
    full_file_df = ""
    with open(filepath, 'r', encoding="utf-8") as json_file:
        json_full = json.load(json_file)
    full_file_df = pd.json_normalize(json_full)
    return full_file_df
def save_to_disk(data, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)
def load_from_disk(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
        return data

def create_rolling_windows(a, window_size):
    '''
    Credit: https://stackoverflow.com/a/59974675
    '''
    shape = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



def intersection(lst1, lst2): 
    tup1 = map(tuple, lst1) 
    tup2 = map(tuple, lst2)  
    return list(map(list, set(tup1).intersection(tup2))) 

def reject_outliers(data, m=2):
    # create index of data
    index = list(data.index)
    # check where to remove outliers with 2 sigma distance
    outlier_bool = abs(data - np.mean(data)) < m * np.std(data)
    
    # apply to index and data and return
    return data[outlier_bool], list(compress(index, outlier_bool))

def anglebetween(v1, v2):
    v1Norm = v1/np.linalg.norm(v1)
    v2Norm = v2/np.linalg.norm(v2)
    Dot = np.dot(v1Norm, v2Norm)
    angle = math.degrees(np.arccos(Dot))
    
    return angle


def eye_outlier_removal_sigma(pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, m=1.5):
    '''
    Find and remove temporally correlated outliers in pos_x, pos_y, pos_z, dir_x, dir_y, dir_z. 
    Anchor is pos_x. 
    m is the number of standard deviations, datapoints need to be distanced from the mean to be classified as outliers. 

    For left-bound clusters, use next correct value to the right, 
    for right-bound clusters, use next correct value to the left,
    for clusters within the signal, linearly interpolate between value to the left and to the right. 
    '''

    # preparation
    outlier_df = pd.DataFrame()
    info_df = pd.DataFrame(columns=['Cluster', 'Length', 'Data Prop (%)'])
    pos_x = pos_x.copy()
    pos_y = pos_y.copy()
    pos_z = pos_z.copy()
    dir_x = dir_x.copy()
    dir_y = dir_y.copy()
    dir_z = dir_z.copy()
    
    # check where to remove outliers with m * sigma distance
    outlier_bool = abs(pos_x - np.mean(pos_x)) > m * np.std(pos_x)
    # get outlier indices 
    outlier_df['outlier_index'] = pos_x[outlier_bool].index
    # check for sample clusters
    outlier_df['clusters'] = (outlier_df['outlier_index']-1 != outlier_df['outlier_index'].shift()).cumsum()

    for cluster in range(1, outlier_df['clusters'].max()+1):

        min_c = np.min(outlier_df.outlier_index[outlier_df.clusters == cluster])
        max_c = np.max(outlier_df.outlier_index[outlier_df.clusters == cluster])
 
        # check the min edge case 
        if min_c == 0:
            pos_x[min_c:max_c+1] = pos_x[max_c+1]
            pos_y[min_c:max_c+1] = pos_y[max_c+1]
            pos_z[min_c:max_c+1] = pos_z[max_c+1]
            dir_x[min_c:max_c+1] = dir_x[max_c+1]
            dir_y[min_c:max_c+1] = dir_y[max_c+1]
            dir_z[min_c:max_c+1] = dir_z[max_c+1]
            
        # check the max edge case
        elif max_c == len(pos_x)-1:
            pos_x[min_c:max_c+1] = pos_x[min_c-1]
            pos_y[min_c:max_c+1] = pos_y[min_c-1] 
            pos_z[min_c:max_c+1] = pos_z[min_c-1] 
            dir_x[min_c:max_c+1] = dir_x[min_c-1]
            dir_y[min_c:max_c+1] = dir_y[min_c-1] 
            dir_z[min_c:max_c+1] = dir_z[min_c-1] 
            
        # all other cases
        elif (max_c-min_c)<=21:

            # interpolate linearly between left and right bound
            pos_x[min_c:max_c+1] = np.linspace(pos_x[min_c], pos_x[max_c], len(pos_x[min_c:max_c+1]))
            pos_y[min_c:max_c+1] = np.linspace(pos_y[min_c], pos_y[max_c], len(pos_y[min_c:max_c+1]))
            pos_z[min_c:max_c+1] = np.linspace(pos_z[min_c], pos_z[max_c], len(pos_z[min_c:max_c+1]))
            dir_x[min_c:max_c+1] = np.linspace(dir_x[min_c], dir_x[max_c], len(dir_x[min_c:max_c+1]))
            dir_y[min_c:max_c+1] = np.linspace(dir_y[min_c], dir_y[max_c], len(dir_y[min_c:max_c+1]))
            dir_z[min_c:max_c+1] = np.linspace(dir_z[min_c], dir_z[max_c], len(dir_z[min_c:max_c+1]))

        else:
            pass
        
        info_df.loc[cluster, 'Cluster'] = cluster
        info_df.loc[cluster, 'Length'] = max_c - min_c
        info_df.loc[cluster, 'Data Prop (%)'] = (max_c - min_c)/len(pos_x)*100

    return pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, info_df, outlier_df


def eye_outlier_removal_zero_values(pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, padding = 8):
    '''
    Finds and removes temporally correlated outliers in pos_x, pos_y, pos_z, dir_x, dir_y, dir_z. 
    Works by finding candidates in all supplied data where the data goes to 0. 
    Replaces clusters of those candidates with surrounding padding by average of environment. 
    
    For left-bound clusters, use next (including padding) correct value to the right, 
    for right-bound clusters, use next (including padding) correct value to the left,
    for clusters within the signal, linearly interpolate between value to the left and to the right (both including padding). 
    '''

    # preparation
    info_df = pd.DataFrame(columns=['Cluster', 'Length', 'Amount of data (%)','Length incl. padding','Amount of data (incl. padding) (%)'])
    pos_y = pos_y.copy()
    pos_x = pos_x.copy()
    pos_z = pos_z.copy()
    dir_x = dir_x.copy()
    dir_y = dir_y.copy()
    dir_z = dir_z.copy()
    
        
    # find candidates that have datapoints going to zero and combine candidates
    bools_df = pd.DataFrame(columns=["pos_x","pos_y","pos_z","dir_x","dir_y","dir_z","combined"])
    bools_df["pos_x"] = pos_x == 0
    bools_df["pos_y"] = pos_y == 0
    bools_df["pos_z"] = pos_z == 0
    bools_df["dir_x"] = dir_x == 0
    bools_df["dir_y"] = dir_y == 0
    bools_df["dir_z"] = dir_z == 0
    bools_df["is_outlier"] = bools_df["pos_x"] & bools_df["pos_y"] & bools_df["pos_z"] & bools_df["dir_x"] & bools_df["dir_y"] & bools_df["dir_z"]
    
    
    # create clusters of similar non-outlier/ outlier condition
    bools_df["all_clusters"] = bools_df["is_outlier"].diff() 
    bools_df.iloc[0, bools_df.columns.get_loc("all_clusters")] = True 
    bools_df.loc[bools_df["all_clusters"],"all_clusters"] = np.arange(len(bools_df[bools_df["all_clusters"]])) + 1
    bools_df.loc[bools_df["all_clusters"] == False,"all_clusters"] = np.nan
    bools_df["all_clusters"].fillna(method="ffill",inplace=True)
    
    # get outlier cluster names 
    outlier_clusters = bools_df.loc[bools_df["is_outlier"],"all_clusters"].unique()

    # change values in data for each outlier cluster 
    for cluster in outlier_clusters:

        # find min and max of cluster 
        min_c = np.min(bools_df.loc[bools_df["all_clusters"] == cluster, "all_clusters"].index)
        max_c = np.max(bools_df.loc[bools_df["all_clusters"] == cluster, "all_clusters"].index)
        
        # add infos     
        info_df.loc[cluster, 'Cluster'] = cluster
        info_df.loc[cluster, 'Length'] = max_c - min_c
        info_df.loc[cluster, 'Amount of data (%)'] = (max_c - min_c)/len(pos_x)*100
 
        # min edge case, use value from the right  
        if min_c == 0:
            
            right_bound = max_c+padding
            if right_bound >= len(pos_x):
                print("\033[93mEyeOutlierRemovalZeroValues: WARNING. Cluster spans entire segment.\033[0m")
                right_bound = len(pos_x) - 1

            pos_x[min_c:right_bound+1] = pos_x[right_bound]
            pos_y[min_c:right_bound+1] = pos_y[right_bound]
            pos_z[min_c:right_bound+1] = pos_z[right_bound]
            dir_x[min_c:right_bound+1] = dir_x[right_bound]
            dir_y[min_c:right_bound+1] = dir_y[right_bound]
            dir_z[min_c:right_bound+1] = dir_z[right_bound]
            
            # add info 
            info_df.loc[cluster, 'Length incl. padding'] = right_bound - min_c
            info_df.loc[cluster, 'Amount of data (incl. padding) (%)'] = (right_bound - min_c)/len(pos_x)*100
        
            
            
        # max edge case, use value from the left 
        elif max_c == len(pos_x)-1:

            left_bound = min_c-padding
            if left_bound < 0:
                print("EyeOutlierRemovalZeroValues: WARNING. Cluster spans entire segment.")
                left_bound = 0

            pos_x[left_bound:max_c+1] = pos_x[left_bound]
            pos_y[left_bound:max_c+1] = pos_y[left_bound] 
            pos_z[left_bound:max_c+1] = pos_z[left_bound] 
            dir_x[left_bound:max_c+1] = dir_x[left_bound]
            dir_y[left_bound:max_c+1] = dir_y[left_bound] 
            dir_z[left_bound:max_c+1] = dir_z[left_bound] 
            
            # add info 
            info_df.loc[cluster, 'Length incl. padding'] = max_c - left_bound
            info_df.loc[cluster, 'Amount of data (incl. padding) (%)'] = (max_c - left_bound)/len(pos_x)*100
            
        # in between case 
        else:
            
            # check if padding does not exceed boundaries 
            if max_c + padding > len(pos_x) - 1:
                padding = len(pos_x) - 1 - max_c
            if min_c - padding < 0: 
                padding = min_c
            
            # interpolate linearly between left and right bound
            pos_x[min_c-padding:max_c+padding+1] = np.linspace(pos_x[min_c-padding], pos_x[max_c+padding], len(pos_x[min_c-padding:max_c+padding+1]))
            pos_y[min_c-padding:max_c+padding+1] = np.linspace(pos_y[min_c-padding], pos_y[max_c+padding], len(pos_y[min_c-padding:max_c+padding+1]))
            pos_z[min_c-padding:max_c+padding+1] = np.linspace(pos_z[min_c-padding], pos_z[max_c+padding], len(pos_z[min_c-padding:max_c+padding+1]))
            dir_x[min_c-padding:max_c+padding+1] = np.linspace(dir_x[min_c-padding], dir_x[max_c+padding], len(dir_x[min_c-padding:max_c+padding+1]))
            dir_y[min_c-padding:max_c+padding+1] = np.linspace(dir_y[min_c-padding], dir_y[max_c+padding], len(dir_y[min_c-padding:max_c+padding+1]))
            dir_z[min_c-padding:max_c+padding+1] = np.linspace(dir_z[min_c-padding], dir_z[max_c+padding], len(dir_z[min_c-padding:max_c+padding+1]))


            # add info 
            info_df.loc[cluster, 'Length incl. padding'] = max_c + padding - (min_c - padding)
            info_df.loc[cluster, 'Amount of data (incl. padding) (%)'] = (max_c + padding - (min_c - padding))/len(pos_x)*100
                      

    # copy outlier info about data points 
    outlier_df = bools_df[["is_outlier","all_clusters"]]
    

    return pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, info_df, outlier_df



def validate_participant_seen_areas(eyetracking_data_paths, exclude_participant_ids = None):
    '''
    Check which areas a participant has seen during the experiment. 
    '''

    # get the participant ids from all eyetracking data locations
    participant_ids = []
    participant_paths = []
    for eyetracking_path in eyetracking_data_paths:
        participant_paths += glob.glob(eyetracking_path + "*.txt")
    participant_ids = [os.path.basename(path).split("_EyeTracking_")[0] for path in participant_paths]
    participant_ids = list(set(participant_ids))

    # search for data per participant 
    for participant in participant_ids:
        
        if (exclude_participant_ids is not None) and (participant in exclude_participant_ids):
            continue

        # eyetracking data files
        eyetracking_data = []
        for path in eyetracking_data_paths:
            eyetracking_data += glob.glob(path + "/" + str(participant) + "*.txt")

        # Participant did not see all areas (Training, Mountain Road, Country Road, Westbrueck, Autobahn)
        if len(eyetracking_data) < 5:
            print("\nParticipant", participant, "only has reduced number of segments available in eye tracking data:", [os.path.basename(path).split("_EyeTracking_")[1] for path in eyetracking_data])
  