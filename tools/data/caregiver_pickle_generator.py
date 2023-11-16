import os
import glob
import numpy as np
import pandas as pd
import csv
import pickle

trimmed_mode = True
points_to_trim = [1, 2, 4, 12]
activity_label_map = {2: 0, 3: 1, 4: 2, 6: 3, 9: 4, 12: 5}
num_markers = 29
final_num_markers = 29
if trimmed_mode: final_num_markers = final_num_markers - len(points_to_trim)

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

for sample in train_ann_reader:
    train_name_list.append("{}_{}".format(sample['segment_id'], str(activity_label_map[int(sample['activity_id'])])))
    train_labels[sample['segment_id']] = str(activity_label_map[int(sample['activity_id'])])

for sample in test_ann_reader:
    test_name_list.append("{}_{}".format(sample['segment_id'], str(activity_label_map[int(sample['activity_id'])])))
    test_labels[sample['segment_id']] = str(activity_label_map[int(sample['activity_id'])])

split_dict = {'xsub_train': train_name_list, 'xsub_val': test_name_list}

annot_list = []
for train_sample_path in train_list:
    segment_id = train_sample_path.split('\\')[-1].split('.')[0].replace('segment', '')
    marker_columns = [str(i) for i in range(2, 2 + num_markers * 3)]
    df = pd.read_csv(train_sample_path)
    
    sample_info = {"frame_dir": "{}_{}".format(segment_id, train_labels[segment_id]),
                   "label": int(train_labels[segment_id]),
                   "total_frames": len(df.index) - 1}
    keypoints = np.empty([1, len(df.index) - 1, final_num_markers, 3])

    for index, row in df.iterrows():
        if index == len(df.index) - 1: break
        frame_data = row[marker_columns].values.reshape(-1, 3)  # Reshape to (29, 3)
        if trimmed_mode: frame_data = np.delete(frame_data, points_to_trim, 0)
        keypoints[0][index] = frame_data
    
    sample_info['keypoint'] = keypoints
    annot_list.append(sample_info)

for test_sample_path in test_list:
    segment_id = test_sample_path.split('\\')[-1].split('.')[0].replace('segment', '')
    marker_columns = [str(i) for i in range(2, 2 + num_markers * 3)]
    df = pd.read_csv(train_sample_path)
    
    sample_info = {"frame_dir": "{}_{}".format(segment_id, test_labels[segment_id]),
                   "label": int(test_labels[segment_id]),
                   "total_frames": len(df.index) - 1}
    keypoints = np.empty([1, len(df.index) - 1, final_num_markers, 3])

    for index, row in df.iterrows():
        if index == len(df.index) - 1: break
        frame_data = row[marker_columns].values.reshape(-1, 3)  # Reshape to (29, 3)
        if trimmed_mode: frame_data = np.delete(frame_data, points_to_trim, 0)
        keypoints[0][index] = frame_data
    
    sample_info['keypoint'] = keypoints
    annot_list.append(sample_info)

caregiver_3d = {'split': split_dict, 'annotations': annot_list}

with open('caregiver_3d.pkl', 'wb') as file:
    pickle.dump(caregiver_3d, file)