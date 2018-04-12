import os
import glob
import argparse
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


def get_data():
    # use the csv first file in the dataset directory
    filename = glob.glob(os.path.join(settings.dataset_dir, "*.csv"))[0]
    dataset_path = os.path.join(settings.dataset_dir, filename)

    my_data = pd.read_csv(dataset_path, header=1, error_bad_lines=False)
    df = pd.DataFrame(my_data)
    df.drop(df.columns[[0, 3, 7, 8]], axis=1, inplace=True)
    return df


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


def build_model(layers):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    return model


def main(settings):
    df = get_data()
    values = df.values
    minima = np.amin(values[:, -1])
    maxima = np.amax(values[:, -1])
    scaling_parameter = maxima - minima
    values[:, 4] = (values[:, 4] - minima) / scaling_parameter
    values[:, 0] = (values[:, 0] - np.amin(values[:, 0])) / (np.amax(values[:, 0]) - np.amin(values[:, 0]))
    values[:, 1] = (values[:, 1] - np.amin(values[:, 1])) / (np.amax(values[:, 1]) - np.amin(values[:, 1]))
    values[:, 2] = (values[:, 2] - np.amin(values[:, 2])) / (np.amax(values[:, 2]) - np.amin(values[:, 2]))
    values[:, 3] = (values[:, 3] - np.amin(values[:, 3])) / (np.amax(values[:, 3]) - np.amin(values[:, 3]))

    df = pd.DataFrame(values)
    window = 5
    X_train, y_train, X_test, y_test = load_data(df[::-1], window)
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)

    model = build_model([5, window, 1])
    model.fit(
        X_train,
        y_train,
        batch_size=settings.batch_size,
        epochs=settings.epochs,
        validation_split=settings.validation_split,
        verbose=2)

    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(settings.output_dir, "model-layout.json"), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(os.path.join(settings.output_dir, "model-weights.h5"))
    print("Saved model to disk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--validation_split", type=float, required=True)
    parser.add_argument("--dataset_dir", type=str, default="/valohai/inputs/dataset")
    parser.add_argument("--output_dir", type=str, default="/valohai/outputs")
    settings, unparsed = parser.parse_known_args()
    main(settings)
