import os
import json
import argparse
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import LambdaCallback
from utils import get_dataframe, load_data


def build_single_lstm(layers):
    model = Sequential()
    model.add(LSTM(50, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])
    return model


def build_double_lstm(layers):
    dropout = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    return model


model_architectures = {
    "single_lstm": build_single_lstm,
    "double_lstm": build_double_lstm,
}


def main(settings):
    df, dataset_format = get_dataframe(settings.dataset_dir)

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
    window = 5
    X_train, y_train, X_test, y_test = load_data(df[::-1], window)
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)

    json_logging_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(json.dumps({
            "epoch": epoch,
            "loss": logs["loss"],
            "acc": logs["acc"],
            "val_loss": logs["val_loss"],
            "val_acc": logs["val_acc"],
        })),
    )

    # figure out which model architecture to use
    arch = settings.model_architecture
    assert arch in model_architectures, "Unknown model architecture '%s'." % arch
    builder = model_architectures[arch]

    # build and train the model
    if dataset_format == 'rte':
        shape_param = 5
    else:
        shape_param = 4

    model = builder([shape_param, window, 1])
    model.fit(
        X_train,
        y_train,
        batch_size=settings.batch_size,
        epochs=settings.epochs,
        validation_split=settings.validation_split,
        callbacks=[json_logging_callback],
        verbose=0)

    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(settings.output_dir, "model-layout.json"), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(os.path.join(settings.output_dir, "model-weights.h5"))
    print("Saved model to disk")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--validation_split", type=float, required=True)
    parser.add_argument("--model_architecture", type=str, required=True, help="'single_lstm' or 'double_lstm'")
    parser.add_argument("--dataset_dir", type=str, default="/valohai/inputs/dataset")
    parser.add_argument("--output_dir", type=str, default="/valohai/outputs")
    settings = parser.parse_args()
    main(settings)


if __name__ == "__main__":
    cli()
