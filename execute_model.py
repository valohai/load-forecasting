import os
import glob
import json
import numpy as np
import pandas as pd
from math import sqrt
from keras.models import model_from_json
from sklearn.metrics import mean_absolute_error
import argparse
from utils import feature_extraction, load_data


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main(settings):
    features, minima, maxima, scaling_parameter, dataset_format = feature_extraction(settings.dataset_dir)
    window = 5
    X_train, y_train, X_test, y_test = load_data(features[::-1], window)
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
