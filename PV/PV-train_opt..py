
# LSTM 
from numpy import array
from numpy import hstack
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as met
from sklearn.metrics import r2_score
from sklearn import datasets, linear_model
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as met
from os import system
from sklearn.metrics import r2_score
import math
import joblib
# load and clean-up data
from numpy import nan
from numpy import isnan
from pandas import read_csv
from pandas import to_numeric
import seaborn as sns
from datetime import datetime
from matplotlib import pyplot
from numpy.random import seed
seed(1234)  # seed random numbers for Keras
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score

import tensorflow 
tensorflow.random.set_seed(2)  # seed random numbers for Tensorflow backend
np.random.seed(1337)  # for reproducibility
## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
import re
# fill missing values with a value at the same time one day ago
def fill_missing(values):
 one_day = 60 * 24
 for row in range(values.shape[0]):
    for col in range(values.shape[1]):
        if isnan(values[row, col]):
            values[row, col] = values[row - one_day, col]
# ----------   Read DataSet from diractory  ---------------
data = pd.read_csv('HSBLL_PV.csv', header=0, infer_datetime_format=True, parse_dates=['latest_ref_time'], index_col=['latest_ref_time'], delimiter=',', quotechar='"')

tag='HSBLL_PV'
data.dropna(inplace=True)

# mark all missing values
data.replace('?', nan, inplace=True)
DataSet = data[['DateAndTime','Day','Hour','SolarDownwardRadiation','RelativeHumidity','CloudCover', 'Temperature', 'WindDirection', 'WindSpeed', 'PV']]
DataSet1 = data[['Hour','SolarDownwardRadiation','RelativeHumidity', 'CloudCover', 'Temperature', 'WindDirection', 'WindSpeed', 'PV']]
Predictors=DataSet[['Hour','SolarDownwardRadiation','RelativeHumidity', 'CloudCover', 'Temperature', 'WindDirection', 'WindSpeed']]
PV=DataSet['PV']# PV1=DataSet['PV']
DataSet.dropna(inplace=True)
from sklearn.preprocessing import MinMaxScaler
PV=DataSet['PV']
TestingDate='2022-09-01'
DataTrain=DataSet[DataSet['DateAndTime']<TestingDate].copy()
DataTest=DataSet[DataSet['DateAndTime']>=TestingDate].copy()
DataTrain=DataSet.iloc[:len(DataTrain)+1,:]
DataTest=DataSet.iloc[len(DataTrain)+1:,:]
from sklearn.preprocessing import Normalizer



X_tr=DataTrain[['Hour','SolarDownwardRadiation','RelativeHumidity', 'CloudCover', 'Temperature', 'WindDirection', 'WindSpeed']]
Y_tr=DataTrain['PV']
X_tes=DataTest[['Hour','SolarDownwardRadiation','RelativeHumidity', 'CloudCover', 'Temperature', 'WindDirection', 'WindSpeed']]
Y_tes=DataTest['PV']

sc = MinMaxScaler(feature_range = (-1, 1))
dataset = sc.fit_transform(DataSet1)
#Just for saving the minmax scalers------------------------------

DataSet_scalar = Predictors

scaler_filename ='Scalar_predictors_'+tag +'.save'

sc2 = MinMaxScaler(feature_range = (-1, 1))
DataSet_scalar_scaled = sc2.fit_transform(DataSet_scalar)
joblib.dump(sc2, scaler_filename) 

#--------------------------------------------------------------
#Just for saving the minmax scalers------------------------------

DataSet3 = DataSet[['PV']]

scaler_filename  ='Scalar_Y_'+tag +'.save'

sc3 = MinMaxScaler(feature_range = (-1, 1))
dataset3 = sc3.fit_transform(DataSet3)
joblib.dump(sc3, scaler_filename)


# -------------- Eliminationg of duplicated samples in PV cell -----------------

# dataset = DataSet.to_numpy()
# split into input (X) and output (y) variables
X = dataset[:, 0:-1]
y=dataset[:,-1]
X, y = X.astype('float'), y.astype('float')
n_features = X.shape[1]


DataTrain_Scal=dataset[0:len(DataTrain)+1]
DataTest_Scal=dataset[len(DataTrain)+1:]
X_train=DataTrain_Scal[:,0:-1]
X_test=DataTest_Scal[:,0:-1]
y_train=DataTrain_Scal[:,-1]
y_test=DataTest_Scal[:,-1]

# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape) 
# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].
#Model architecture: 1) LSTM with 100 neurons in the first visible layer, 3) dropout 20%, 4) 1 neuron in the output layer for predicting Global_active_power, 5) The input shape will be 1 time step with 7 features, 6) I use the Mean Absolute Error (MAE) loss function and the efficient Adam version of stochastic gradient descent, 7) The model will be fit for 20 training epochs with a batch size of 70.
model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

from keras.callbacks import History 
history = History()

# fit network
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), verbose=2, shuffle=False)
model.save('LSTM for PV Output Prediction.hdf5')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
# evaluate on test set

yhat = model.predict(X_test)
error = mean_absolute_error(y_test, yhat)
# the nRMSE error
#def nrmse(y_test,yhat):
#    rmse=np.sqrt(mean_squared_error(y_test,yhat))
#    return rmse/(y_test.max()-y_test.min())

Y_Predicted=sc3.inverse_transform(yhat)


model.save('trainedLSTM_PV_'+tag +'.save') #Svaes the trained model

#---------------------------Distribution function and statistic metrics calculation-----------------------
ytest=Y_tes.to_numpy()

residuals = [ytest[i]-Y_Predicted[i] for i in range(len(Y_tes))]
residuals = pd.DataFrame(residuals)
# summary statistics
stat_data=residuals.describe()
mean=stat_data.loc['mean'].values
std=stat_data.loc['std'].values

# Save statistic to pass ot with the predictions
stats_filename = "stats_" + tag +'.save'
stats=[[mean, std]] 
df=pd.DataFrame(stats,columns=['mean','standard deviation'])
joblib.dump(df, stats_filename) 

#------------------------------Plots------------------------------------------------------------------------
# Y_Predicted=(yhat*(max(PV)-min(PV)))+(L.reshape(len(L),1))
PP=Y_tes.to_numpy()
plt.plot(Y_Predicted, color='blue',label='Predicted')
plt.plot(PP, color='red',label='Actual')

#plt.axis([0, 150, 0, 30])
plt.title('LSTM  Model')
plt.xlabel('Time(Hours)')
plt.ylabel('PV (kW)')
plt.legend()
plt.show()

print('MAE: %.3f' % error)
# Y_Predicted=(Predicted_test*(max(yy1)-min(yy1))).flatten()+[np.ones(len(Predicted_test))*(min(yy1))]
print( '------- Results & Accuracy For Test   -----------')
print('-------- MAE ------')
mae = met.mean_absolute_error(Y_tes, Y_Predicted)
print(mae)

print('-------- RMSE ---------')
rmse = met.mean_squared_error(Y_tes, Y_Predicted)**0.5
print(rmse)

print('--------- MSE ------')
mse = met.mean_squared_error(Y_tes, Y_Predicted)
print(mse)

# For PV we have zero and very small values so MAPE will be infinite
print('-------- R2 -------')
r2 = math.sqrt(r2_score(Y_tes, Y_Predicted))
print(r2)

