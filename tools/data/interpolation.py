import os
import glob
import pandas as pd

base_data_path = "data"

train_path = os.path.join(base_data_path, 'train', 'train')
test_path = os.path.join(base_data_path, 'test', 'test')

train_list = glob.glob(os.path.join(train_path, 'mocap', '*.csv'))
test_list = glob.glob(os.path.join(test_path, 'mocap', '*.csv'))

def fill_missing_coordinates(df):
    # Assuming columns are named as 'Joint_1_X', 'Joint_1_Y', 'Joint_1_Z', ..., 'Joint_29_X', 'Joint_29_Y', 'Joint_29_Z'
    joints_columns = [f'{i}' for i in range(2, 89)]

    # Interpolate missing values for each joint separately
    for joint_col in joints_columns:
        df[joint_col] = df[joint_col].interpolate(method='spline', limit_direction='backward', order=1)

    return df

def process_csv_file(file_path):
    # Read CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Fill missing coordinates
    df = fill_missing_coordinates(df)

    # Save the updated DataFrame back to CSV
    df.to_csv(file_path, index=False)

# Replace 'file1.csv', 'file2.csv', etc. with your actual file names
file_list = train_list + test_list

# Process each CSV file
for file_path in file_list:
    process_csv_file(file_path)

print('interpolation is finished')