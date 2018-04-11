import numpy as np
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.metrics import mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_data(normalized=0):
    my_data = pd.read_csv('rte.csv', header=1, error_bad_lines=False)
    df = pd.DataFrame(my_data)
    df.drop(df.columns[[0, 3, 7, 8]], axis=1, inplace=True)
    #     df.drop(df.index[1000:])
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
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
    print(x_train[:3])
    print(y_train[:3])
    return [x_train, y_train, x_test, y_test]


df = get_data(0)
# min_max_scaler = preprocessing.MinMaxScaler()
# nomalized = min_max_scaler.fit_transform(df.values)
values = df.values
minima = np.amin(values[:, -1])
maxima = np.amax(values[:, -1])
scaling_parameter = maxima - minima
values[:, 4] = (values[:, 4]-minima)/scaling_parameter
values[:, 0] = (values[:, 0]-np.amin(values[:, 0]))/(np.amax(values[:, 0])-np.amin(values[:, 0]))
values[:, 1] = (values[:, 1]-np.amin(values[:, 1]))/(np.amax(values[:, 1])-np.amin(values[:, 1]))
values[:, 2] = (values[:, 2]-np.amin(values[:, 2]))/(np.amax(values[:, 2])-np.amin(values[:, 2]))
values[:, 3] = (values[:, 3]-np.amin(values[:, 3]))/(np.amax(values[:, 3])-np.amin(values[:, 3]))

df = pd.DataFrame(values)
window = 5
X_train, y_train, X_test, y_test = load_data(df[::-1], window)

# load json and create model
json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model2.h5")
print("Loaded model from disk")

predicted1 = model.predict(X_test)
actual = y_test
predicted1 = (predicted1*scaling_parameter)+minima
actual = (actual*scaling_parameter)+minima

mape1 = sqrt(mean_absolute_percentage_error(predicted1, actual))
print('Test MAPE1: %.3f' % mape1)
mse1  = mean_absolute_error(actual, predicted1)
print('Test MSE1: %.3f' % mse1)
