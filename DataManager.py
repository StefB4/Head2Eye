'''
Authors: Stefan Balle & Lucas Essmann 
2021 
'''

import os 
import glob 
import numpy as np 
import pandas as pd

from Helpers import read_normalized_json_to_df, save_to_disk, load_from_disk, create_rolling_windows, eye_outlier_removal_sigma, eye_outlier_removal_zero_values
from AngleHelpers import create_relative_directions, create_relative_directions_consider_roll

RESAMPLE_STRATEGY = "MEAN" # "FILL"
TIMESTAMP_DECIMALS = 2 
TIME_DELTA = 0.01
BOOTSTRAP_BASEPATH = "./bootstrapped_participant_data/"

REF_APPLY_CONSIDER_ROLL = True
REF_APPLIED_REMOVE_OUTLIERS = True 
OUTLIER_REMOVAL_STRATEGY = "ZEROVALUES" #"SIGMA" 


class ParticipantData:
    '''
    Class for data of an individual participant.
    (Data includes eyetracking, input, participantcalibrationdata and scenedata)
    '''
   
    def __init__(self, eyetracking_filepaths, input_filepaths, calibration_filepath, scenedata_filepaths, bootstrap_file_loading=False, verbose = False):
        '''
        Constructor.
        @ Paths to the relevant files 
        '''
        
        # sanity checks 
        # after this, we can be sure, that all paths exist, the files have labels corresponding to what they should contain
        # the participant id is always the same, the number of files per list corresponds and there is one calibration path 
        # the lists are lists of strings and the calibration data is a string 
        if (eyetracking_filepaths is None or input_filepaths is None or calibration_filepath is None or scenedata_filepaths is None):
            raise AssertionError("Some of the supplied paths are None! Exiting.")
        if (not isinstance(eyetracking_filepaths, list) or not isinstance(input_filepaths, list) or not isinstance(calibration_filepath, str) or not isinstance(scenedata_filepaths, list)):
            raise AssertionError("Some of the paths are not supplied in the correct format! Exiting.")
        if (len(eyetracking_filepaths) != len(input_filepaths) or len(eyetracking_filepaths) != len(scenedata_filepaths)):
            raise AssertionError("The number of supplied paths per category does not match! Exiting.")
        if (len(eyetracking_filepaths) < 1):
            raise AssertionError("Some supplied path lists have less than one element! Exiting.")
        if (not all(isinstance(elem,str) for elem in eyetracking_filepaths) or not all(isinstance(elem,str) for elem in input_filepaths) or not all(isinstance(elem,str) for elem in scenedata_filepaths)):
            raise AssertionError("The supplied path lists do not contain string contents! Exiting.")
        participant_id = "-8"
        for elem in eyetracking_filepaths + input_filepaths + [calibration_filepath] + scenedata_filepaths:
            if not os.path.isfile(elem):
                raise AssertionError("Some of the supplied paths do not link to existing files! Exiting.")
            try:
                if (participant_id == "-8"):
                    participant_id = os.path.basename(elem).split("_")[0]
                else:
                    next_participant_id = os.path.basename(elem).split("_")[0]
                    if next_participant_id != participant_id:
                        raise AssertionError("Some of the supplied paths do not link to files of the same participant! Exiting.")
                    else:
                        participant_id = next_participant_id 
            except:
                raise AssertionError("Some of the supplied paths do not link to files of the same participant! Exiting.")
        for elem in eyetracking_filepaths:
            if not "_EyeTracking_" in elem:
                raise AssertionError("Some of the supplied paths for eye tracking data link to files that are labeled for something else! Exiting.")
        for elem in input_filepaths:
            if not "_Input_" in elem:
                raise AssertionError("Some of the supplied paths for input data link to files that are labeled for something else! Exiting.")
        if not "_ParticipantCalibrationData" in calibration_filepath:
            raise AssertionError("The supplied path for participant calibration data links to a file that is labeled for something else! Exiting.")
        for elem in scenedata_filepaths:
            if not "_SceneData_" in elem:
                raise AssertionError("Some of the supplied paths for scene data link to files that are labeled for something else! Exiting.")
        
        
        # store filepaths
        self.eyetracking_filepaths = eyetracking_filepaths
        self.input_filepaths = input_filepaths
        self.calibration_filepath = calibration_filepath
        self.scenedata_filepaths = scenedata_filepaths
        
        # store participant id and number of recorded areas and verbosity and whether reference data has been applied
        self.participant_id = os.path.basename(eyetracking_filepaths[0]).split("_")[0]
        self.number_of_recorded_areas = len(eyetracking_filepaths)
        self.verbose = verbose
        self.reference_data_applied = False 
        
        # init data dictionaries 
        print("ParticipantData: Initialising participant " + self.participant_id + ".")
        self.eyetracking_data = {}
        self.input_data = {}
        self.calibration_data = {}
        self.scene_data = {} 
        self.golden_segment_data_vanilla = {}
        self.golden_event_info = {}
        self.golden_segment_data_ref_applied = {} # generated later, not during construction
        
        # bootstrap file loading
        if bootstrap_file_loading and os.path.isfile(os.path.join(BOOTSTRAP_BASEPATH, "./bootstrap_" + str(self.participant_id) + ".pickle")):
            bootstrap_data = load_from_disk(os.path.join(BOOTSTRAP_BASEPATH, "./bootstrap_" + str(self.participant_id) + ".pickle"))
            self.participant_id = bootstrap_data["participant_id"]
            self.eyetracking_data = bootstrap_data["eyetracking_data"]
            self.input_data = bootstrap_data["input_data"]
            self.calibration_data = bootstrap_data["calibration_data"]
            self.scene_data = bootstrap_data["scene_data"]
            self.golden_segment_data_vanilla = bootstrap_data["golden_segment_data_vanilla"]
            self.golden_event_info = bootstrap_data["golden_event_info"] 
            print("ParticipantData: Loaded data (bootstrapped) for participant " + str(self.participant_id) + ".")
    
        else:
            # process raw data 
            self._load_raw_data()
            self._extract_event_information()
            self._extract_path_segments()
            self._resample_path_segments()  
            self._construct_segment_data()
            self._construct_event_info()
            print("ParticipantData: Loaded data (raw) for participant " + str(self.participant_id) + ".")
            
            # save to disk 
            if bootstrap_file_loading:
                bootstrap_data = {}
                bootstrap_data["participant_id"] = self.participant_id
                bootstrap_data["eyetracking_data"] = self.eyetracking_data
                bootstrap_data["input_data"] = self.input_data
                bootstrap_data["calibration_data"] = self.calibration_data
                bootstrap_data["scene_data"] = self.scene_data
                bootstrap_data["golden_segment_data_vanilla"] = self.golden_segment_data_vanilla
                bootstrap_data["golden_event_info"] = self.golden_event_info
                save_to_disk(bootstrap_data,os.path.join(BOOTSTRAP_BASEPATH, "./bootstrap_" + str(self.participant_id) + ".pickle"))
                if self.verbose:
                    print("ParticipantData: Saved bootstrapped data to disk.")
                    print("")
    
    def _read_raw_data(self, filepaths):
        '''
        Read raw files into dictionary. 
        '''
        
        data = {}
        
        for idx, filename in enumerate(filepaths):
            if self.verbose: 
                print("ParticipantData: Loading " + filename + " (file " + str(idx+1) + "/" + str(len(filepaths)) + ")...")
            
            if "Westbrueck" in filename:
                token = "Westbrueck"
            elif "MountainRoad" in filename:
                token = "MountainRoad"
            elif "CountryRoad" in filename:
                token = "CountryRoad"
            elif "Autobahn" in filename:
                token = "Autobahn"  
            elif "TrainingScene" in filename:
                token = "TrainingScene"
            else:  # not defined 
                print("ParticipantData: Found undefined area token in filename" + filename + "!")
                continue # in the loop     
            data[token] = {}
            data[token]["filename"] = filename
            data[token]["full_df"] = read_normalized_json_to_df(filename)
        
        return data 
    
    def _load_raw_data(self):
        
        # Eye tracking data
        if self.verbose: 
            print("ParticipantData: Loading raw eyetracking data files...")
        self.eyetracking_data = self._read_raw_data(self.eyetracking_filepaths)
        
        # Input data 
        if self.verbose: 
            print("ParticipantData: Loading raw input data files...")
        self.input_data = self._read_raw_data(self.input_filepaths)
        
        # Scene data  
        if self.verbose: 
            print("ParticipantData: Loading raw scene data files...")
        self.scene_data = self._read_raw_data(self.scenedata_filepaths)

        # Calibration data  
        if self.verbose:
            print("ParticipantData: Loading raw calibration data...")
        self.calibration_data = {}
        self.calibration_data['filename'] = self.calibration_filepath
        self.calibration_data['full_df'] = read_normalized_json_to_df(self.calibration_filepath)
        
        
        
    def _extract_event_information(self):
        
        # Extract most important event information 
        for area in ["Westbrueck","MountainRoad","CountryRoad","Autobahn","TrainingScene"]:
            self.scene_data[area]["number_of_events"] = len(self.scene_data[area]["full_df"]["EventBehavior"][0])
            self.scene_data[area]["events"] = {}
            for idx, event in enumerate(self.scene_data[area]["full_df"]["EventBehavior"][0]):
                self.scene_data[area]["events"][idx] = {'name':event["EventName"],'start':event["StartofEventTimeStamp"],'stop':event["EndOfEventTimeStamp"],'succeeded':event["SuccessfulCompletionState"]}
        
        
    
    def _extract_path_segments(self):
        '''
        Extract the path segments. 
        Skip the TrainingScene, no relevant data. 
        '''
        
        
        # Copy of entire dataframe input dataframe to prepare processing
        for area in ["Westbrueck","MountainRoad","CountryRoad","Autobahn"]:
            self.eyetracking_data[area]["processed_df"] = self.eyetracking_data[area]["full_df"].copy(deep=True)
            self.eyetracking_data[area]["processed_df"].drop(columns=["TobiiTimeStamp","FPS","hitObjects","RightEyeIsBlinkingWorld","RightEyeIsBlinkingLocal","LeftEyeIsBlinkingWorld","LeftEyeIsBlinkingLocal"],inplace=True)
            self.eyetracking_data[area]["processed_df"]["path_segment_label"] = -9 # event label 

        # Give label to individual path segments, -9 is event label 
        for area in ["Westbrueck","MountainRoad","CountryRoad","Autobahn"]:
            for event_idx in range(len(self.scene_data[area]["events"]) + 1):
                cond = None 
                if event_idx == 0: # find all datapoints with timestamps before event start timestamp 
                    cond = (self.eyetracking_data[area]["processed_df"]["UnixTimeStamp"] < self.scene_data[area]["events"][event_idx]["start"])
                elif event_idx < len(self.scene_data[area]["events"]): # find all datapoints with timestamp between prev and next event 
                    cond = (self.eyetracking_data[area]["processed_df"]["UnixTimeStamp"] > self.scene_data[area]["events"][event_idx - 1]["stop"]) & (self.eyetracking_data[area]["processed_df"]["UnixTimeStamp"] < self.scene_data[area]["events"][event_idx]["start"])
                elif event_idx == len(self.scene_data[area]["events"]): # find all datapoints with timestamp after last event
                    cond = (self.eyetracking_data[area]["processed_df"]["UnixTimeStamp"] > self.scene_data[area]["events"][event_idx - 1]["stop"])

                # Filter     
                self.eyetracking_data[area]["processed_df"].loc[cond, "path_segment_label"] = event_idx

        # Extract path segments, add timestamps beginning at zero 
        for area in ["Westbrueck","MountainRoad","CountryRoad","Autobahn"]:
            self.eyetracking_data[area]["path_segments_no_resample"] = {}
            for label in self.eyetracking_data[area]["processed_df"]["path_segment_label"].unique():
                if (label != -9):

                    # copy segment data 
                    cond = (self.eyetracking_data[area]["processed_df"]["path_segment_label"] == label)
                    self.eyetracking_data[area]["path_segments_no_resample"][label] = self.eyetracking_data[area]["processed_df"].loc[cond].copy(deep=True)

                    # add timestamp starting at zero 
                    if label == 0: # take first recorded datapoint as base timestamp 
                        ref_timestamp = self.eyetracking_data[area]["path_segments_no_resample"][label]["UnixTimeStamp"].iloc[0]
                    else: # take last event's stop time as base timestamp 
                        ref_timestamp = self.scene_data[area]["events"][label - 1]["stop"]

                    self.eyetracking_data[area]["path_segments_no_resample"][label]["rebased_timestamp"] = self.eyetracking_data[area]["path_segments_no_resample"][label]["UnixTimeStamp"] - ref_timestamp

                else: # skip events 
                    pass 
         
    
    
    def _resample_path_segments(self):
        '''
        Resample the path segments.
        Exclude TrainingScene, no relevant data. 
        '''
        

        if self.verbose:
            print("ParticipantData: Resampled path segments (excl. events):")
        for area in ["Westbrueck","MountainRoad","CountryRoad","Autobahn"]:

            self.eyetracking_data[area]["path_segments_resampled"] = {}
            for segment in self.eyetracking_data[area]["path_segments_no_resample"]:
                # copy segments 
                self.eyetracking_data[area]["path_segments_resampled"][segment] = self.eyetracking_data[area]["path_segments_no_resample"][segment].copy(deep=True)

                # Hard match datapoints to closest timebin and fill arising holes by forward fill 
                if RESAMPLE_STRATEGY == "FILL":

                    # round the timestamps to specified number of decimals 
                    self.eyetracking_data[area]["path_segments_resampled"][segment]["rebased_timestamp_rounded"] = self.eyetracking_data[area]["path_segments_resampled"][segment]["rebased_timestamp"].round(TIMESTAMP_DECIMALS)

                    # "resample" timestamps by reindexing with time delta (default 0.01s) steps and filling holes; first drop duplicate timestamps
                    self.eyetracking_data[area]["path_segments_resampled"][segment]["resampled_timestamp"] = self.eyetracking_data[area]["path_segments_resampled"][segment]["rebased_timestamp_rounded"] 
                    self.eyetracking_data[area]["path_segments_resampled"][segment].drop_duplicates(subset="resampled_timestamp",keep="first", inplace=True)
                    start_time = 0
                    end_time = self.eyetracking_data[area]["path_segments_resampled"][segment]["resampled_timestamp"].iloc[-1]
                    time_delta = TIME_DELTA 
                    new_index = pd.Index(np.arange(start_time,end_time,time_delta), name="resampled_timestamp")
                    self.eyetracking_data[area]["path_segments_resampled"][segment] = self.eyetracking_data[area]["path_segments_resampled"][segment].set_index("resampled_timestamp").reindex(new_index).reset_index()

                    # keep track of where data was interpolated 
                    self.eyetracking_data[area]["path_segments_resampled"][segment]["is_interpolated"] = self.eyetracking_data[area]["path_segments_resampled"][segment]["rebased_timestamp"].isnull()

                    # fill nans ("interpolate") 
                    exclude_columns = ["rebased_timestamp","rebased_timestamp_rounded","is_interpolated","UnixTimeStamp","resampled_timestamp"]
                    for column in self.eyetracking_data[area]["path_segments_resampled"][segment].columns:  
                        if column not in exclude_columns:
                            self.eyetracking_data[area]["path_segments_resampled"][segment][column].fillna(method='ffill', inplace = True)
                            self.eyetracking_data[area]["path_segments_resampled"][segment][column].fillna(method='bfill', inplace = True)

                    # drop unneeded columns
                    self.eyetracking_data[area]["path_segments_resampled"][segment].drop(columns=["UnixTimeStamp","rebased_timestamp","rebased_timestamp_rounded","path_segment_label"], inplace = True)


                # Resample using pandas' resample, fill with mean 
                if RESAMPLE_STRATEGY == "MEAN":

                    # copy timestamp 
                    self.eyetracking_data[area]["path_segments_resampled"][segment]["resampled_timestamp"] = self.eyetracking_data[area]["path_segments_resampled"][segment]["rebased_timestamp"] 

                    # create datetime from rebased timestamp
                    self.eyetracking_data[area]["path_segments_resampled"][segment]["rebased_datetime"] = pd.to_datetime(self.eyetracking_data[area]["path_segments_resampled"][segment]['resampled_timestamp'],unit='s')

                    # resample with time delta (default 0.01s) interval and keep track of holes in the data before interpolation
                    self.eyetracking_data[area]["path_segments_resampled"][segment] = self.eyetracking_data[area]["path_segments_resampled"][segment].resample(str(TIME_DELTA) + 'S',on="rebased_datetime").mean()
                    self.eyetracking_data[area]["path_segments_resampled"][segment]["is_interpolated"] = self.eyetracking_data[area]["path_segments_resampled"][segment]["resampled_timestamp"].isnull()
                    self.eyetracking_data[area]["path_segments_resampled"][segment].reset_index(inplace=True)

                    # interpolate linearly
                    self.eyetracking_data[area]["path_segments_resampled"][segment].interpolate(method="linear",inplace=True)

                    # get resampled_timestamp from rebased_datetime again 
                    self.eyetracking_data[area]["path_segments_resampled"][segment]["resampled_timestamp"] = (self.eyetracking_data[area]["path_segments_resampled"][segment]["rebased_datetime"] - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')

                    # drop unneeded columns
                    self.eyetracking_data[area]["path_segments_resampled"][segment].drop(columns=["rebased_datetime","UnixTimeStamp","rebased_timestamp","path_segment_label"], inplace = True)

                if self.verbose:
                    print("Area: " + area + " Segment: " + str(segment) + " Total datapoints (incl. resampled): " + str(len(self.eyetracking_data[area]["path_segments_resampled"][segment]["is_interpolated"]))  + " Filled NaNs: " + str(self.eyetracking_data[area]["path_segments_resampled"][segment]["is_interpolated"].values.sum()))
        
        

    def _construct_segment_data(self):
        
        # extract most important infos
        for area in ["Westbrueck","MountainRoad","CountryRoad","Autobahn"]:
            self.golden_segment_data_vanilla[area] = {}
            for segment in self.eyetracking_data[area]["path_segments_resampled"]:
                self.golden_segment_data_vanilla[area][segment] = self.eyetracking_data[area]["path_segments_resampled"][segment].copy(deep = True)
            
                
    def _construct_event_info(self):
    
        # extract event infos 
        for area in ["Westbrueck","MountainRoad","CountryRoad","Autobahn"]:
            self.golden_event_info[area] = {}
            for event in self.scene_data[area]["events"]:
                self.golden_event_info[area][event] = {}
                self.golden_event_info[area][event]['name'] = self.scene_data[area]["events"][event]['name']
                self.golden_event_info[area][event]['start'] = self.scene_data[area]["events"][event]['start']
                self.golden_event_info[area][event]['stop'] = self.scene_data[area]["events"][event]['stop']
                self.golden_event_info[area][event]['succeeded'] = self.scene_data[area]["events"][event]['succeeded']
    
    def apply_reference_data(self, ref_data_dict):
        
        self.golden_segment_data_ref_applied = {}
        for area in ["Westbrueck","MountainRoad","CountryRoad","Autobahn"]:
            
            self.golden_segment_data_ref_applied[area] = {}
            
            for segment in self.golden_segment_data_vanilla[area]:
                max_len = len(self.golden_segment_data_vanilla[area][segment])
                if len(ref_data_dict[area][segment]) < max_len:
                    max_len = len(ref_data_dict[area][segment])
                if self.verbose:
                    print("ParticipantData: Applying reference data to back of " + area + "'s segment " + str(segment) + ". Number of datapoints used: " + str(max_len) + ".") 
                    
                # potentially cut length of data if reference data is shorter; cut off in the front to align data in the back 
                self.golden_segment_data_ref_applied[area][segment] = self.golden_segment_data_vanilla[area][segment].iloc[-max_len:].copy(deep=True).reset_index(drop=True)
                
                # remove temporally correlated outliers
                # needs to be done before applying reference data, since zero value method relies on datapoints going to zero  
                if REF_APPLIED_REMOVE_OUTLIERS:

                    if OUTLIER_REMOVAL_STRATEGY == "SIGMA":
                        self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.x"], \
                        self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.y"], \
                        self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.z"], \
                        self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.x"], \
                        self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.y"], \
                        self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.z"], _, _ = \
                            eye_outlier_removal_sigma(self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.x"], \
                                                self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.y"], \
                                                self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.z"], \
                                                self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.x"], \
                                                self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.y"], \
                                                self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.z"], m=1.5)

                    elif OUTLIER_REMOVAL_STRATEGY == "ZEROVALUES": 
                        self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.x"], \
                        self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.y"], \
                        self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.z"], \
                        self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.x"], \
                        self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.y"], \
                        self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.z"], _, _ = \
                            eye_outlier_removal_zero_values(self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.x"], \
                                                self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.y"], \
                                                self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.z"], \
                                                self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.x"], \
                                                self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.y"], \
                                                self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.z"], padding=8) # 7 starts to produce good results

                    else:
                        raise AssertionError("ParticipantData: Specified outlier removal strategy " + str(OUTLIER_REMOVAL_STRATEGY) + " does not exist!")


                # Apply reference data values, Positions 
                self.golden_segment_data_ref_applied[area][segment]["HmdPosition.x"] -= ref_data_dict[area][segment].iloc[-max_len:]["CarPosition.x"].reset_index(drop=True)
                self.golden_segment_data_ref_applied[area][segment]["HmdPosition.y"] -= ref_data_dict[area][segment].iloc[-max_len:]["CarPosition.y"].reset_index(drop=True)
                self.golden_segment_data_ref_applied[area][segment]["HmdPosition.z"] -= ref_data_dict[area][segment].iloc[-max_len:]["CarPosition.z"].reset_index(drop=True)
                self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.x"] -= ref_data_dict[area][segment].iloc[-max_len:]["CarPosition.x"].reset_index(drop=True)
                self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.y"] -= ref_data_dict[area][segment].iloc[-max_len:]["CarPosition.y"].reset_index(drop=True)
                self.golden_segment_data_ref_applied[area][segment]["EyePosWorldCombined.z"] -= ref_data_dict[area][segment].iloc[-max_len:]["CarPosition.z"].reset_index(drop=True)
            
                # Reference for directional data 
                if REF_APPLY_CONSIDER_ROLL:
                    ref_angle_unity_x = ref_data_dict[area][segment].iloc[-max_len:]["car_rotation_angles.x"].reset_index(drop=True)
                    ref_angle_unity_y = ref_data_dict[area][segment].iloc[-max_len:]["car_rotation_angles.y"].reset_index(drop=True)
                    ref_angle_unity_z = ref_data_dict[area][segment].iloc[-max_len:]["car_rotation_angles.z"].reset_index(drop=True)
                else:
                    ref_dir_unity_x = ref_data_dict[area][segment].iloc[-max_len:]["car_rotation_direction.x"].reset_index(drop=True)
                    ref_dir_unity_y = ref_data_dict[area][segment].iloc[-max_len:]["car_rotation_direction.y"].reset_index(drop=True)
                    ref_dir_unity_z = ref_data_dict[area][segment].iloc[-max_len:]["car_rotation_direction.z"].reset_index(drop=True)
        
                # Apply reference data values, Nose Vector 
                inp_dir_unity_x = self.golden_segment_data_ref_applied[area][segment]["NoseVector.x"]
                inp_dir_unity_y = self.golden_segment_data_ref_applied[area][segment]["NoseVector.y"]
                inp_dir_unity_z = self.golden_segment_data_ref_applied[area][segment]["NoseVector.z"]
                if REF_APPLY_CONSIDER_ROLL:
                    res_dir_x, res_dir_y, res_dir_z = create_relative_directions_consider_roll(inp_dir_unity_x,inp_dir_unity_y,inp_dir_unity_z,ref_angle_unity_x,ref_angle_unity_y,ref_angle_unity_z) 
                else:
                    res_dir_x, res_dir_y, res_dir_z = create_relative_directions(inp_dir_unity_x,inp_dir_unity_y,inp_dir_unity_z,ref_dir_unity_x,ref_dir_unity_y,ref_dir_unity_z,method="anglediff_sphere_coords") # method="unit_sphere_rotation"
                self.golden_segment_data_ref_applied[area][segment]["NoseVector.x"] = res_dir_x 
                self.golden_segment_data_ref_applied[area][segment]["NoseVector.y"] = res_dir_y 
                self.golden_segment_data_ref_applied[area][segment]["NoseVector.z"] = res_dir_z 

                # Apply reference data values, Eye Direction 
                inp_dir_unity_x = self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.x"]
                inp_dir_unity_y = self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.y"]
                inp_dir_unity_z = self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.z"]
                if REF_APPLY_CONSIDER_ROLL:
                    res_dir_x, res_dir_y, res_dir_z = create_relative_directions_consider_roll(inp_dir_unity_x,inp_dir_unity_y,inp_dir_unity_z,ref_angle_unity_x,ref_angle_unity_y,ref_angle_unity_z) 
                else:
                    res_dir_x, res_dir_y, res_dir_z = create_relative_directions(inp_dir_unity_x,inp_dir_unity_y,inp_dir_unity_z,ref_dir_unity_x,ref_dir_unity_y,ref_dir_unity_z,method="anglediff_sphere_coords") # method="unit_sphere_rotation"
                self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.x"] = res_dir_x
                self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.y"] = res_dir_y 
                self.golden_segment_data_ref_applied[area][segment]["EyeDirWorldCombined.z"] = res_dir_z 
                    

                # drop local columns 
                #self.golden_segment_data_ref_applied[area][segment].drop(columns = ['EyePosLocalCombined.x','EyePosLocalCombined.y','EyePosLocalCombined.z','EyeDirLocalCombined.x','EyeDirLocalCombined.y','EyeDirLocalCombined.z'],inplace = True)
                
                
    def set_verbosity(self,verbosity):
        self.verbose = verbosity 
                

    def get_segment_data(self, use_vanilla=False, filter_data=False, filter_by_corr_coeff_dict=None, corr_coeff_threshold=0, get_first_segment=True, after_event_type_only=[True,False],exclude_areas=[],exclude_segments=[]):
        '''
        Get either vanilla or reference applied segment data.
        @use_vanilla: Use vanilla segment data for filtering instead of reference applied segment data. 
        @filter_data: Apply filtering or do not.
        @filter_by_corr_coeff_dict: Dictionary that holds correlation coefficients between reference data runs for each segment; if supplied, filter by corr coeffs and ignore all other parameters.
        @corr_coeff_threshold: Filter out all segments below threshold value. 
        @get_first_segment: Include the first segment (before first event) in filtered data or not.
        @after_event_type_only: Include only segments that happened after succeeded (True) or failed (False) events. 
        @exclude_areas: Areas to exclude in filtered data.
        @exclude_segments: Segments to exclude in filtered data. 
        '''
        
        # determine use of vanilla or ref applied data 
        copy_from_data = None 
        if use_vanilla:
            copy_from_data = self.golden_segment_data_vanilla
        else:
            copy_from_data = self.golden_segment_data_ref_applied
        
        # filter data
        if filter_data:
            
            # init filtered data 
            filtered_data = {}

            # filter by correlation coefficients
            if filter_by_corr_coeff_dict is not None:
                if self.verbose:
                    print("ParticipantData: Filtering data by Correlation Coefficients with threshold " + str(corr_coeff_threshold) + ".")
                
                for area in ["Westbrueck","MountainRoad","CountryRoad","Autobahn"]:
                    for segment in copy_from_data[area]:
                        
                        # use data, as corr coeffs are bigger than threshold for all measured values in that segment
                        if ((filter_by_corr_coeff_dict[area][segment] >= corr_coeff_threshold).all().all()):
                            
                            # make sure keys exist
                            if not area in filtered_data:
                                filtered_data[area] = {}
                            if not segment in filtered_data[area]:
                                filtered_data[area][segment] = {}

                            # add data 
                            filtered_data[area][segment] = copy_from_data[area][segment].copy(deep=True)


                        # exclude data, corr coeffs smaller than threshold 
                        else:
                            if self.verbose:
                                print("ParticipantData: Corr-coeff filtering excluded " + str(area) + " segment " + str(segment) + " with min corr-coeffs " + str(filter_by_corr_coeff_dict[area][segment].min(axis=1).item()) + ".")
                            pass


             

            # filter by manual specifications
            else:
                if self.verbose:
                    print("ParticipantData: Filtering data with manual settings.")
               

                for area in ["Westbrueck","MountainRoad","CountryRoad","Autobahn"]:
                    
                    # check that area is wished 
                    if area not in exclude_areas:
                        filtered_data[area] = {}
                        
                        for segment in copy_from_data[area]:
                            
                            # check that segment is wished 
                            if segment not in exclude_segments:
                                
                                # check that first segment is wished 
                                if segment == 0 and get_first_segment:
                                    filtered_data[area][segment] = {}
                                    filtered_data[area][segment] = copy_from_data[area][segment].copy(deep=True)
                                
                                # check that event before segment has wished condition 
                                if segment > 0 and self.golden_event_info[area][segment - 1]["succeeded"] in after_event_type_only:
                                    filtered_data[area][segment] = {}
                                    filtered_data[area][segment] = copy_from_data[area][segment].copy(deep=True)


            return filtered_data  
            
        # get unfiltered data
        else:
            return copy_from_data
    
    def get_event_info(self):
        return self.golden_event_info
    
    def get_participant_id(self):
        return self.participant_id 
        


#######################
#######################
#######################
#######################




class MeasurementData:
    '''
    Full measurement data. Contains multiple ParticipantDatas. 
    '''
    
    def __init__(self, eyetracking_data_paths,input_data_paths,participant_calibration_data_paths,scene_data_paths,verbosity=False):
        
        # sanity check  
        for path in eyetracking_data_paths + input_data_paths + participant_calibration_data_paths + scene_data_paths:
            if not os.path.isdir(path):
                raise AssertionError("Specified path " + path + " does not exist!")
        
        # store parameters 
        self.eyetracking_data_paths = eyetracking_data_paths
        self.input_data_paths = input_data_paths
        self.participant_calibration_data_paths = participant_calibration_data_paths
        self.scene_data_paths = scene_data_paths
        self.verbose=verbosity 
        
        # init 
        self.measurement_data = {}
        self.participant_ids = {}
        
        # extract participants 
        self._extract_participants() 
        self._generate_participant_data()
        
    
    def _extract_participants(self):
        '''
        Extract unique participant ids and store corresponding file paths. 
        '''
        
        # get the participant ids from all eyetracking data locations
        participant_ids = []
        for eyetracking_path in self.eyetracking_data_paths:
            participant_ids += [os.path.basename(path).split("_EyeTracking_")[0] for path in glob.glob(eyetracking_path + "*.txt")]
        participant_ids = list(set(participant_ids))
        self.participant_ids = participant_ids
        
        # search for data per participant 
        # create dictionary entries for each participant
        for participant in participant_ids:
            
            # input data files 
            input_data = []
            for path in self.input_data_paths:
                input_data += glob.glob(path + "/" + str(participant) + "*.txt")
            
            # eyetracking data files
            eyetracking_data = []
            for path in self.eyetracking_data_paths:
                eyetracking_data += glob.glob(path + "/" + str(participant) + "*.txt")
            
            # calibration data files
            calibration_data = []
            for path in self.participant_calibration_data_paths:
                calibration_data += glob.glob(path + "/" + str(participant) + "*.txt")
            
            # scene data files
            scene_data = []
            for path in self.scene_data_paths:
                scene_data += glob.glob(path + "/" + str(participant) + "*.txt")
        
            # store file paths
            self.measurement_data[participant] = {}
            self.measurement_data[participant]["filepaths"] = {}
            self.measurement_data[participant]["filepaths"]["input_data"] = input_data
            self.measurement_data[participant]["filepaths"]["eyetracking_data"] = eyetracking_data
            self.measurement_data[participant]["filepaths"]["calibration_data"] = calibration_data
            self.measurement_data[participant]["filepaths"]["scene_data"] = scene_data
            
            # log 
            if self.verbose: 
                print("MeasurementData: Found files for participant " + participant + ".")
                print("Input data files: " + str(input_data))
                print("Eyetracking data files: " + str(eyetracking_data))
                print("Calibration data files: " + str(calibration_data))
                print("Scene data files: " + str(scene_data))
                print("")
            
            
            
    def _generate_participant_data(self):
        '''
        Generate ParticipantData objects for all participants. 
        '''
        
        for participant in self.participant_ids:
            
            self.measurement_data[participant]["data"] = \
                ParticipantData(self.measurement_data[participant]["filepaths"]["eyetracking_data"], \
                                self.measurement_data[participant]["filepaths"]["input_data"], \
                                self.measurement_data[participant]["filepaths"]["calibration_data"][0], \
                                self.measurement_data[participant]["filepaths"]["scene_data"], 
                                True, self.verbose)
    
    def get_participant_list(self):
        return self.participant_ids
    
            
    def apply_reference_data(self,reference_data):
        '''
        Apply supplied reference data to all subjects.
        '''
        print("MeasurementData: Applying reference data to all participants...")
        for participant in self.participant_ids:
            self.measurement_data[participant]["data"].apply_reference_data(reference_data)
        print("MeasurementData: Done applying reference data to all participants.")
            
    
    def get_data(self, use_vanilla=False, filter_data=False, filter_by_corr_coeff_dict=None, corr_coeff_threshold=0, \
                 get_first_segment=True, after_event_type_only=[True,False], \
                 exclude_areas=[], exclude_segments=[], \
                 exclude_participants=[]):
        '''
        Get possibly filtered data from multiple participants. 
        Filtering by correlation coefficients overrides all other settings, except for excluding participants.
        '''      
        
        data = {}
        
        for participant in self.participant_ids:
            
            # check if participant should be skipped 
            if not participant in exclude_participants:
                data[participant] = self.measurement_data[participant]["data"].get_segment_data(use_vanilla=use_vanilla, filter_data=filter_data, filter_by_corr_coeff_dict=filter_by_corr_coeff_dict, corr_coeff_threshold=corr_coeff_threshold, get_first_segment=get_first_segment, after_event_type_only=after_event_type_only,exclude_areas=exclude_areas,exclude_segments=exclude_segments)
        
        return data 
    
    
    def truncate_data(self,data_dict,window_size,method='numpy'):
        '''
        Truncate data from the provided data dictionary.
        Making sure that the window never slides across areas/ segments/ dataframe length. 
        Methods: 'rolling', 'indexing', 'numpy'
        '''
        
        result = []
        
        if method == 'rolling':
            method = 0
        elif method == 'indexing':
            method = 1
        elif method == 'numpy':
            method = 2
            result = None 
        
        # go through all participants, areas and segments
        for participant in data_dict:
            for area in data_dict[participant]:
                for segment in data_dict[participant][area]:
                    
                    # rolling
                    if method == 0:
                    
                        # cast bools to int 
                        data_dict[participant][area][segment]["is_interpolated"] = data_dict[participant][area][segment]["is_interpolated"].astype("int16")
                         
                        # create rolling windows 
                        rolling = list(data_dict[participant][area][segment].rolling(window=window_size))
                            
                        # exclude first few samples that do not have wished length
                        del rolling[:window_size-1]
                                   
                        result.extend(rolling)
                            
                    # indexing    
                    elif method == 1:   
                        
                        # as long as sliding window fits into data frame, append windowed data  
                        i = 0
                        while (i + window_size) <= len(data_dict[participant][area][segment]):
                            result.append(data_dict[participant][area][segment].iloc[i:i+window_size])
                            i += 1
                    
                    # numpy 
                    elif method == 2:
                        
                        # cast bools to int 
                        data_dict[participant][area][segment]["is_interpolated"] = data_dict[participant][area][segment]["is_interpolated"].astype("int16")
                        
                        # create numpy representation
                        numpy_array = data_dict[participant][area][segment].to_numpy()
                        
                        # rolling windows 
                        windows = create_rolling_windows(numpy_array,window_size)
                        
                        # append to results 
                        if result is None:
                            result = windows
                        else:
                            result = np.append(result, windows,axis=0)
                                            
        
        return result 
        
    def combine_data(self,data_dict):
        '''
        Combine the data provided sequentially into one dataframe.
        '''
    
        result = None
        
        # go through all participants, areas and segments
        for participant in data_dict:
            for area in data_dict[participant]:
                for segment in data_dict[participant][area]:
                    
                    if result is None:
                        result = data_dict[participant][area][segment].copy(deep=True) 
                    else:
                        result = pd.concat([result,data_dict[participant][area][segment]])

        return result 
    
    def average_data(self, data_dict):
        '''
        Calculate the average of the provided data per segment 
        by adding datapoints together (back aligned) and dividing by the number of available segments. 
        '''
        
        result = {}
        available_segments = {}
        
        # go through all participants, areas and segments
        for participant in data_dict:
            for area in data_dict[participant]:
                
                # add area if not yet added 
                if not area in available_segments.keys():
                    available_segments[area] = {}
                    result[area] = {}
                
                for segment in data_dict[participant][area]:
                    
                    # add segment if not yet added 
                    if not segment in available_segments[area].keys():
                        available_segments[area][segment] = 0
                        result[area][segment] = None 
                    
                    # increase counter for segment 
                    available_segments[area][segment] += 1
                    
                    # add data to segment 
                    if result[area][segment] is None:
                        result[area][segment] = data_dict[participant][area][segment]
                    else:
                        
                        # find max_len 
                        max_len = len(result[area][segment])
                        if max_len > len(data_dict[participant][area][segment]):
                            max_len = len(data_dict[participant][area][segment])
                        
                        # add values 
                        result[area][segment] = result[area][segment].iloc[-max_len:].reset_index(drop=True).add(data_dict[participant][area][segment].iloc[-max_len:].reset_index(drop=True))
                    
         
        # go through available segments and average 
        for area in available_segments:
            for segment in available_segments[area]:
                result[area][segment] = result[area][segment] / available_segments[area][segment]
        return result
                
    
        
            
