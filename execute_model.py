import os
import glob
import json
import numpy as np
import pandas as pd
from math import sqrt
from keras.models import model_from_json
from sklearn.metrics import mean_absolute_error
import argparse


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_data(dataset_dir):
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


def load_data(my_data, seq_len):
    amount_of_features = len(my_data.columns)
    data = my_data.as_matrix()
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


def main(settings):
    df, dataset_format = get_data(settings.dataset_dir)

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
        # TODO: do something with the Date column (values[:, 0])
        values[:, 0] = (values[:, 0] - np.amin(values[:, 0])) / (np.amax(values[:, 0]) - np.amin(values[:, 0]))
        values[:, 1] = (values[:, 1] - np.amin(values[:, 1])) / (np.amax(values[:, 1]) - np.amin(values[:, 1]))
        values[:, 2] = (values[:, 2] - np.amin(values[:, 2])) / (np.amax(values[:, 2]) - np.amin(values[:, 2]))
        values[:, 3] = (values[:, 3] - minima) / scaling_parameter

    df = pd.DataFrame(values)
    window = 5
    X_train, y_train, X_test, y_test = load_data(df[::-1], window)
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)

    # load json and create model
    layout_path = glob.glob(os.path.join(settings.model_dir, "*layout.json"))[0]
    json_file = open(layout_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    weights_path = glob.glob(os.path.join(settings.model_dir, "*weights.h5"))[0]
    model.load_weights(weights_path)
    print("Loaded model from disk")

    predicted2 = model.predict(X_test)
    actual = y_test
    predicted2 = (predicted2 * scaling_parameter) + minima
    actual = (actual * scaling_parameter) + minima

    mape2 = sqrt(mean_absolute_percentage_error(predicted2, actual))
    mse2 = mean_absolute_error(actual, predicted2)

    print(json.dumps({
        "mape": mape2,
        "mse": mse2
    }))


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/valohai/inputs/dataset")
    parser.add_argument("--model_dir", type=str, default="/valohai/inputs/model")
    settings = parser.parse_args()
    main(settings)


if __name__ == "__main__":
    cli()
