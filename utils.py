import os
import glob
import pandas as pd
import numpy as np


def get_dataframe(dataset_dir):
    """
    :param dataset_dir: Path to directory containing dataset as a .csv file
    :return: Panda DataFrame
    """

    # use the csv first file in the dataset directory
    dataset_path = glob.glob(os.path.join(dataset_dir, "*.csv"))[0]

    my_data = pd.read_csv(dataset_path, error_bad_lines=False)
    df = pd.DataFrame(my_data)

    column_names = list(df)
    if 'Demand' in column_names:
        # RTE dataset format
        df = df.filter(items=['Day', 'Month', 'Hours', 'Temperature', 'Demand'])
        return df, 'rte'
    elif 'SYSLoad' in column_names:
        # ERCOT dataset format.
        df = df.filter(items=['Day', 'Month', 'Minutes', 'SYSLoad'])
        return df, 'ercot'
    else:
        raise Exception('Unknown dataset format with columns: {}'.format(column_names))


def feature_extraction(dataset_dir):
    df, dataset_format = get_dataframe(dataset_dir)

    values = df.values
    minima = np.amin(values[:, -1])
    maxima = np.amax(values[:, -1])
    scaling_parameter = maxima - minima

    if dataset_format == 'rte':
        values[:, 0] = (values[:, 0] - np.amin(values[:, 0])) / (np.amax(values[:, 0]) - np.amin(values[:, 0]))
        values[:, 1] = (values[:, 1] - np.amin(values[:, 1])) / (np.amax(values[:, 1]) - np.amin(values[:, 1]))
        values[:, 2] = (values[:, 2] - np.amin(values[:, 2])) / (np.amax(values[:, 2]) - np.amin(values[:, 2]))
        values[:, 3] = (values[:, 3] - np.amin(values[:, 3])) / (np.amax(values[:, 3]) - np.amin(values[:, 3]))
        values[:, 4] = (values[:, 4] - minima) / scaling_parameter
    elif dataset_format == 'ercot':
        values[:, 0] = (values[:, 0] - np.amin(values[:, 0])) / (np.amax(values[:, 0]) - np.amin(values[:, 0]))
        values[:, 1] = (values[:, 1] - np.amin(values[:, 1])) / (np.amax(values[:, 1]) - np.amin(values[:, 1]))
        values[:, 2] = (values[:, 2] - np.amin(values[:, 2])) / (np.amax(values[:, 2]) - np.amin(values[:, 2]))
        values[:, 3] = (values[:, 3] - minima) / scaling_parameter

    df = pd.DataFrame(values)
    return df, minima, maxima, scaling_parameter


def split_features(features_data_frame, seq_len):
    amount_of_features = len(features_data_frame.columns)
    data = features_data_frame.as_matrix()
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.8 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
    return [x_train, y_train, x_test, y_test]
