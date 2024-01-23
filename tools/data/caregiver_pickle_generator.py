import os
import glob
import numpy as np
import pandas as pd
import csv
import pickle
import copy

points_map = {0: 12, 1: 4, 2: 2, 3: 1, 4: 7, 5: 8, 6: 9, 7: 9, 8: 3, 9: 5, 10: 6, 11: 6, 12: 11, 13: 20, 14: 22, 15: 24, 16: 9, 17: 14, 18: 16, 19: 18, 20: 4, 21: 9, 22: 9, 23: 6, 24: 6}
activity_label_map = {2: 0, 3: 1, 4: 2, 6: 3, 9: 4, 12: 5}
missing_map = {6: [5], 9: [8], 5: [3, 6], 8: [7, 9], 11: [20, 7], 10: [3, 14], 0: [1, 2], 12: [4, 14, 20], 4: [2, 12, 3, 7], 28: [24, 23, 22], 2: [0, 1], 7: [0, 8], 3: [0, 5], 20: [21, 19], 21: [20, 22], 22: [23, 24], 23: [28], 14: [13, 15], 15: [14, 16], 17: [26], 26: [16, 17, 18], 24: [22, 23, 28], 18: [16, 17, 26], 16: [17, 18, 26]}
num_markers = 29
final_num_markers = 25

is_original_indexes = False

def add_noise(data, noise_level=0.01):
    """
    Add random noise to the input data.
    
    Parameters:
    - data: Input data (numpy array)
    - noise_level: Magnitude of the noise to be added
    
    Returns:
    - Noisy data (numpy array)
    """
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def rotate_frame(frame_data, angle, axis='x'):
    """
    Rotate the frame data around the given axis by the given angle.
    
    Parameters:
    - frame_data: Input frame data (numpy array)
    - angle: Rotation angle in degrees
    - axis: Axis to rotate around ('x', 'y', or 'z')
    
    Returns:
    - Rotated frame data (numpy array)
    """
    angle = np.deg2rad(angle)
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle), -np.sin(angle)],
                                    [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                                    [0, 1, 0],
                                    [-np.sin(angle), 0, np.cos(angle)]])
    else:  # axis == 'z'
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]])
    return np.dot(frame_data, rotation_matrix)

def scale_frame(frame_data, scaling_factor):
    """
    Uniformly scale the frame data by the scaling factor.
    
    Parameters:
    - frame_data: Input frame data (numpy array)
    - scaling_factor: Scaling factor
    
    Returns:
    - Scaled frame data (numpy array)
    """
    return frame_data * scaling_factor

def mirror_frame(frame_data, axis=0):
    """
    Mirror the frame data along the given axis.
    
    Parameters:
    - frame_data: Input frame data (numpy array)
    - axis: Axis to mirror along (0 for x, 1 for y, 2 for z)
    
    Returns:
    - Mirrored frame data (numpy array)
    """
    mirrored_data = np.copy(frame_data)
    mirrored_data[:, axis] = -mirrored_data[:, axis]
    return mirrored_data

def augment_data(frame_list, augmentation_method):
    """
    Apply a random augmentation to the frame data.
    
    Parameters:
    - frame_list: List of frames (each frame is a numpy array)
    
    Returns:
    - Augmented list of frames
    """
    choice = augmentation_method

    if choice == 'rotate':
        angle = np.random.uniform(-30, 30)  # rotate between -30 and 30 degrees
        axis = np.random.choice(['x', 'y', 'z'])
    elif choice == 'scale':
        factor = np.random.uniform(0.8, 1.2)  # scale between 0.8 and 1.2
    elif choice == 'mirror':
        axis = np.random.choice([0, 1, 2])

    augmented_list = []
    for frame_data in frame_list:
        if choice == 'noise':
            augmented_list.append(add_noise(frame_data))
        elif choice == 'rotate':
            augmented_list.append(rotate_frame(frame_data, angle, axis))
        elif choice == 'scale':
            augmented_list.append(scale_frame(frame_data, factor))
        elif choice == 'mirror':
            augmented_list.append(mirror_frame(frame_data, axis))

    return augmented_list

def augment_selected_samples(name_list, annot_list):
    """
    Perform data augmentation on specific samples in the annot_list.

    Parameters:
    - name_list: List of names of the samples to augment.
    - annot_list: List of all annotations (dictionaries).
    - augmentation_methods: List of augmentation methods to apply.

    Returns:
    - Updated annot_list with augmented data for the specified samples.
    """
    augmented_annot_list = []

    for sample in annot_list:
        if sample['frame_dir'] in name_list:
            augmentation_method = np.random.choice(['noise', 'rotate', 'scale', 'mirror'])
            # Perform augmentation
            frames = sample['keypoint'][0]  # Assuming keypoints are stored as [1, num_frames, num_markers, 3]
            augmented_frames = copy.deepcopy(frames)

            augmented_frames = augment_data(augmented_frames, augmentation_method)

            # Update sample info with augmented data
            sample['keypoint'][0] = augmented_frames
            augmented_annot_list.append(sample)
        else:
            # If the sample is not in the list, just add it without changes
            augmented_annot_list.append(sample)

    return augmented_annot_list

def fill_missing(frame_data, target_index):
    counter = 0
    sum = np.zeros(3)
    for filler_index in missing_map[target_index]:
        if not np.any(np.isnan(frame_data[filler_index])):
            counter = counter + 1
            sum = sum + frame_data[filler_index]
    
    if counter == 0:
        return None
    else:
        return sum / counter
    
def split_frames_into_samples(frames, num_samples=10):
    samples = [[] for _ in range(num_samples)]

    for i, frame in enumerate(frames):
        sample_index = i % num_samples
        samples[sample_index].append(frame)

    return samples

base_data_path = "data"

train_path = os.path.join(base_data_path, 'train', 'train')
test_path = os.path.join(base_data_path, 'test', 'test')

train_ann_reader = csv.DictReader(open(os.path.join(train_path, 'activities_train.csv')))
test_ann_reader = csv.DictReader(open(os.path.join(test_path, 'activities_test.csv')))

train_list = glob.glob(os.path.join(train_path, 'mocap', '*.csv'))
test_list = glob.glob(os.path.join(test_path, 'mocap', '*.csv'))

train_name_list = []
test_name_list = []
train_labels = {}
test_labels = {}
subject_map = {}

null_counter = {}

for sample in train_ann_reader:
    train_labels[sample['segment_id']] = str(activity_label_map[int(sample['activity_id'])])
    subject_map[sample['segment_id']] = str(sample['subject'])

for sample in test_ann_reader:
    test_labels[sample['segment_id']] = str(activity_label_map[int(sample['activity_id'])])
    subject_map[sample['segment_id']] = str(sample['subject'])

subject_ids = list(sorted({ele for val in subject_map.values() for ele in val}))

annot_list = []
for train_sample_path in train_list:
    segment_id = train_sample_path.split('\\')[-1].split('.')[0].replace('segment', '')
    marker_columns = [str(i) for i in range(2, 2 + num_markers * 3)]
    df = pd.read_csv(train_sample_path)

    frame_list = []
    for index, row in df.iterrows():
        if index == len(df.index) - 1: break
        frame_data = row[marker_columns].values.reshape(-1, 3).astype(float)  # Reshape to (29, 3)
        new_frame_data = []
        for joint_index in range(final_num_markers):
            if is_original_indexes: 
                target_index = joint_index
            else:
                target_index = points_map[joint_index]
            if np.any(np.isnan(frame_data[target_index])):
                frame_data[target_index] = fill_missing(frame_data, target_index)
                if frame_data[target_index] is None: print(segment_id, target_index)
            new_data = frame_data[target_index]
            new_data = add_noise(new_data)
            new_frame_data.append(new_data)
        new_frame_data = np.array(new_frame_data)
        frame_list.append(new_frame_data)
    
    windows = split_frames_into_samples(frame_list)

    for index, window in enumerate(windows):
        train_name_list.append("{}_{}_{}_{}".format(subject_map[segment_id], segment_id, train_labels[segment_id], str(index)))
        sample_info = {"frame_dir": "{}_{}_{}_{}".format(subject_map[segment_id], segment_id, train_labels[segment_id], str(index)),
                   "label": int(train_labels[segment_id]),
                   "total_frames": len(window)}
        keypoints = np.empty([1, len(window), final_num_markers, 3])
        if is_original_indexes: keypoints = np.empty([1, len(window), num_markers, 3])
        keypoints[0] = np.array(window)
        sample_info["keypoint"] = keypoints
        annot_list.append(sample_info)

for test_sample_path in test_list:
    segment_id = test_sample_path.split('\\')[-1].split('.')[0].replace('segment', '')
    marker_columns = [str(i) for i in range(2, 2 + num_markers * 3)]
    df = pd.read_csv(test_sample_path)

    frame_list = []
    for index, row in df.iterrows():
        if index == len(df.index) - 1: break
        frame_data = row[marker_columns].values.reshape(-1, 3).astype(float)  # Reshape to (29, 3)
        new_frame_data = []
        for joint_index in range(final_num_markers):
            if is_original_indexes: 
                target_index = joint_index
            else:
                target_index = points_map[joint_index]
            if np.any(np.isnan(frame_data[target_index])):
                frame_data[target_index] = fill_missing(frame_data, target_index)
                if frame_data[target_index] is None: print(segment_id, target_index)
            new_data = frame_data[target_index]
            new_data = add_noise(new_data)
            new_frame_data.append(new_data)
        new_frame_data = np.array(new_frame_data)
        frame_list.append(new_frame_data)
    
    windows = split_frames_into_samples(frame_list)

    for index, window in enumerate(windows):
        test_name_list.append("{}_{}_{}_{}".format(subject_map[segment_id], segment_id, test_labels[segment_id], str(index)))
        sample_info = {"frame_dir": "{}_{}_{}_{}".format(subject_map[segment_id], segment_id, test_labels[segment_id], str(index)),
                   "label": int(test_labels[segment_id]),
                   "total_frames": len(window)}
        keypoints = np.empty([1, len(window), final_num_markers, 3])
        if is_original_indexes: keypoints = np.empty([1, len(window), num_markers, 3])
        keypoints[0] = np.array(window)
        sample_info["keypoint"] = keypoints
        annot_list.append(sample_info)

augmented_annot_list = augment_selected_samples(train_name_list, annot_list)

split_dict = {'xsub_train': train_name_list, 'xsub_val': test_name_list}

caregiver_3d = {'split': split_dict, 'annotations': augmented_annot_list}

with open('caregiver_3d.pkl', 'wb') as file:
    pickle.dump(caregiver_3d, file)

full_list = train_name_list + test_name_list
for subject_id in subject_ids:
    new_train_name_list = []
    new_test_name_list = []
    for name in full_list:
        if name.startswith(subject_id):
            new_test_name_list.append(name)
        else:
            new_train_name_list.append(name)
    
    augmented_annot_list = augment_selected_samples(new_train_name_list, annot_list)

    split_dict = {'xsub_train': new_train_name_list, 'xsub_val': new_test_name_list}
    caregiver_3d = {'split': split_dict, 'annotations': augmented_annot_list}
    print(split_dict)

    with open('caregiver_3d_loso_{}.pkl'.format(subject_id), 'wb') as file:
        pickle.dump(caregiver_3d, file)