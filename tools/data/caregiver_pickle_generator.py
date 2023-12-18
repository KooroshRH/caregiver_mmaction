import os
import glob
import numpy as np
import pandas as pd
import csv
import pickle
import random

points_map = {0: 12, 1: 4, 2: 2, 3: 1, 4: 7, 5: 8, 6: 9, 7: 9, 8: 3, 9: 5, 10: 6, 11: 6, 12: 11, 13: 20, 14: 22, 15: 24, 16: 9, 17: 14, 18: 16, 19: 18, 20: 4, 21: 9, 22: 9, 23: 6, 24: 6}
activity_label_map = {2: 0, 3: 1, 4: 2, 6: 3, 9: 4, 12: 5}
missing_map = {6: [5], 9: [8], 5: [3, 6], 8: [7, 9], 11: [20, 7], 10: [3, 14], 0: [1, 2], 12: [4, 14, 20], 4: [2, 12, 3, 7], 28: [24, 23, 22], 2: [0, 1], 7: [0, 8], 3: [0, 5], 20: [21, 19], 22: [23, 24], 14: [13, 15], 26: [16, 17, 18], 24: [22, 23, 28], 18: [16, 17, 26], 16: [17, 18, 26]}
num_markers = 29
final_num_markers = 25

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

null_counter = {}

for sample in train_ann_reader:
    train_labels[sample['segment_id']] = str(activity_label_map[int(sample['activity_id'])])

for sample in test_ann_reader:
    test_labels[sample['segment_id']] = str(activity_label_map[int(sample['activity_id'])])

# full_list = train_name_list + test_name_list
# random.shuffle(full_list)

# test_name_list = []
# for index in range(int(len(full_list)/10)):
#     test_name_list.append(full_list.pop())

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
        for joint_index in range(25):
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
        train_name_list.append("{}_{}_{}".format(segment_id, train_labels[segment_id], str(index)))
        sample_info = {"frame_dir": "{}_{}_{}".format(segment_id, train_labels[segment_id], str(index)),
                   "label": int(train_labels[segment_id]),
                   "total_frames": len(window)}
        keypoints = np.empty([1, len(window), final_num_markers, 3])
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
        for joint_index in range(25):
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
        test_name_list.append("{}_{}_{}".format(segment_id, test_labels[segment_id], str(index)))
        sample_info = {"frame_dir": "{}_{}_{}".format(segment_id, test_labels[segment_id], str(index)),
                   "label": int(test_labels[segment_id]),
                   "total_frames": len(window)}
        keypoints = np.empty([1, len(window), final_num_markers, 3])
        keypoints[0] = np.array(window)
        sample_info["keypoint"] = keypoints
        annot_list.append(sample_info)

split_dict = {'xsub_train': train_name_list, 'xsub_val': test_name_list}

caregiver_3d = {'split': split_dict, 'annotations': annot_list}

with open('caregiver_3d.pkl', 'wb') as file:
    pickle.dump(caregiver_3d, file)