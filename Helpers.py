'''
Authors: Stefan Balle & Lucas Essmann 
2021 
'''

import pandas as pd 
import numpy as np 
import pickle 
import json 

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
