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


def load_data(data_frame, seq_len):
    amount_of_features = len(data_frame.columns)
    data = data_frame.as_matrix()
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
