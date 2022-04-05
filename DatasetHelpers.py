import pandas as pd 
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow import keras
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import tensorflow_datasets as tfds



def create_seq_to_datapoint_ds(combined_data, input_feature_dim, seq_slice_len, batch_size = 64, do_log = False, do_plot = False):
    '''
    Generates a dataset of inputs and targets, where inputs are sequence slices and 
    targets are single values (the ones corresponding in the target sequence to the last values of the current slice).
    Like this: 
    inp_1 inp_2 inp_3 inp_4 inp_5
                            target 
    combined_data should have the shape (full_sequence_length, input_feature_dim + target_feature_dim)
    '''

    # Create dataset from full input and target timeseries 
    # The individual timeseries are stacked together (each timeseries is a column vector, axis 0)
    dataset = tf.keras.utils.timeseries_dataset_from_array(
      data = combined_data, targets = None, sequence_length = seq_slice_len, sequence_stride=1, sampling_rate=1,
      batch_size=None, shuffle=False, seed=None, start_index=None, end_index=None
      # for data: Axis 0 is expected to be the time dimension.
    )

    # Map purely input dataset to dataset that contains input and target, split at input_feature_dim
    # For target use only very last value
    dataset = dataset.map(lambda x: (x[:,0:input_feature_dim], x[-1,np.newaxis,input_feature_dim:]))

    # Get total size of dataset 
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
   
    # Batch
    dataset = dataset.batch(batch_size)

    if do_log:
        print("Total number of samples:", str(dataset_size))
        print("\nDataset:")
        for inputs, targets in dataset.take(1):
            print(f'inputs.shape: {inputs.shape}')
            print(f"targets.shape: {targets.shape}")

    if do_plot:

        # Plot some example data 
        for sample in dataset.skip(10).take(1):

            input_sample = sample[0].numpy()[2]
            target_sample = sample[1].numpy()[2]

            for feature in range(input_feature_dim):
                plt.plot(input_sample[:,feature], label = "Multi Input {}".format(feature + 1))
                plt.plot(seq_slice_len -1, target_sample[:,feature], 'o', label = "Multi Target {}".format(feature + 1))
            plt.legend()
            plt.show()

    return dataset

  
def seq_to_dp_ds_from_df_list(dfs, input_feature_dim, seq_slice_len, batch_size = 64, do_size_log = False):
    '''
    Creates a dataset of sequence (inputs) and single datapoints (targets) tuples
    from a list of dataframes including timeseries data.
    '''
    
    print("Creating dataset from Dataframe List.")
    dataset = None
    for idx, df in enumerate(dfs):
        if (idx % 10 == 0) and (idx > 0):
            print(idx, end = " ")
        curr_ds = create_seq_to_datapoint_ds(df.to_numpy(), input_feature_dim, seq_slice_len, batch_size, do_log = False, do_plot = False)
        if idx == 0:
            dataset = curr_ds
        else:
            dataset = dataset.concatenate(curr_ds)
    print()
    
    if do_size_log:
        # Get total size of dataset 
        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        print("Total number of elements in dataset:", str(dataset_size))
        for inputs, targets in dataset.take(1):
            print(f'Usual inputs.shape: {inputs.shape}')
            print(f"Usual targets.shape: {targets.shape}")
            break

    
    return dataset


def create_pretrain_ds(input_data, seq_slice_len, batch_size = 64, do_log = False, do_plot = False):
    '''
    Generates a dataset of inputs and targets, where inputs are sequence slices and 
    targets are the very same sequence slices.
    I.e. targets are a copy of inputs.
    input_data should have the shape (full_sequence_length, input_feature_dim)
    '''

    # Create dataset from full input timeseries 
    # The individual timeseries are stacked together (each timeseries is a column vector, axis 0)
    dataset = tf.keras.utils.timeseries_dataset_from_array(
      data = input_data, targets = None, sequence_length = seq_slice_len, sequence_stride=1, sampling_rate=1,
      batch_size=None, shuffle=False, seed=None, start_index=None, end_index=None
      # for data: Axis 0 is expected to be the time dimension.
    )

    # Map purely input dataset to dataset that contains input and target (target is copy of input)
    dataset = dataset.map(lambda x: (x, x))

    # Get total size of dataset 
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
   
    # Batch
    dataset = dataset.batch(batch_size)

    if do_log:
        print("Total number of samples:", str(dataset_size))
        print("\nDataset:")
        for inputs, targets in dataset.take(1):
            print(f'inputs.shape: {inputs.shape}')
            print(f"targets.shape: {targets.shape}")


    if do_plot:

      input_feature_dim = input_data.shape[1]

      # Plot some example data 
      for sample in dataset.skip(10).take(1):

          input_sample = sample[0].numpy()[2]
          target_sample = sample[1].numpy()[2]

          for feature in range(input_feature_dim):
              plt.plot(input_sample[:,feature], label = "Multi Input {}".format(feature + 1))
              plt.plot(target_sample[:,feature], label = "Multi Target {}".format(feature + 1))
          plt.legend()
          plt.show()

    return dataset


  
def pretrain_ds_from_df_list(dfs, feature_list, seq_slice_len, batch_size = 64, do_size_log = False):
    '''
    Creates a dataset of same input and target from a list of dataframes including timeseries data.
    Use dataframe columns listed in feature_list.
    '''
    
    print("Creating dataset from Dataframe List.")
    dataset = None
    for idx, df in enumerate(dfs):
        if (idx % 10 == 0) and (idx > 0):
            print(idx, end = " ")
        curr_ds = create_pretrain_ds(df[feature_list].to_numpy(), seq_slice_len, batch_size, do_log = False, do_plot = False)
        if idx == 0:
            dataset = curr_ds
        else:
            dataset = dataset.concatenate(curr_ds)
    print()
    
    if do_size_log:
        # Get total size of dataset 
        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        print("Total number of elements in dataset:", str(dataset_size))
        for inputs, targets in dataset.take(1):
            print(f'Usual inputs.shape: {inputs.shape}')
            print(f"Usual targets.shape: {targets.shape}")
            break

    
    return dataset


def train_validation_split_ds(dataset, val_frac = 0.1, shuffle = True, shuffle_buffer = 2048):
    
    # Shuffle 
    if shuffle:
        print("Shuffling dataset")
        dataset = dataset.shuffle(buffer_size = shuffle_buffer)

    # Get total size of dataset 
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()

    # Split into train and validate and test 
    dataset_train = dataset.take(int(dataset_size * (1 - val_frac)))
    dataset_validate = dataset.skip(int(dataset_size * (1 - val_frac)))
    
    # Get sizes of sub datasets
    train_dataset_size = tf.data.experimental.cardinality(dataset_train).numpy()
    validate_dataset_size = tf.data.experimental.cardinality(dataset_validate).numpy()
    
    print("Full Dataset size:", dataset_size)
    print("Train Dataset size:", train_dataset_size)
    print("Validate Dataset size:", validate_dataset_size)
    
    return dataset_train, dataset_validate


    