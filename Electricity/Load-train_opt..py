from numpy import array
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
from os import system as sys
from sklearn.metrics import r2_score
import math
import holidays
from datetime import date
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle
import seaborn as sns
from datetime import datetime
from numpy import nan
from numpy import isnan

# ----------   Read historical DataSet from directory  ---------------
DataAsli = pd.read_csv('HSBLL_Load.csv')

Load=DataAsli['Load']
tag='HSBLL_Load'



 #----------- Selecting the Historical Data () ----------

 # Load of 24  and 168 hour before
DataAsli['Load24'] = DataAsli.Load.shift(1)
DataAsli['Load168'] = DataAsli.Load.shift(24)
  
#DataAsli['Load'] = DataAsli.Load.shift(168)

#Eliminate the nan values
DataAsli.dropna(inplace=True)
#Making the data weather a time table with respect to the timestamp
DataAsli['valid_datetime2'] = pd.to_datetime(DataAsli['valid_datetime']).copy()
DataAsli['date2'] = pd.DatetimeIndex(DataAsli['valid_datetime2']).date.copy()
DataAsli['Year'] = pd.DatetimeIndex(DataAsli['valid_datetime2']).year
DataAsli['Month'] = pd.DatetimeIndex(DataAsli['valid_datetime2']).month
DataAsli['Day'] = pd.DatetimeIndex(DataAsli['valid_datetime2']).day
DataAsli['hour'] = pd.DatetimeIndex(DataAsli['valid_datetime2']).hour
DataAsli['dayofweek'] = pd.DatetimeIndex(DataAsli['valid_datetime2']).weekday

#-------------------------------------------------------------------------------
#List of Holidays
holiday_list = []
for holiday in holidays.Sweden(years=[2019, 2021]).items():
    holiday_list.append(holiday)

holidays_df = pd.DataFrame(holiday_list, columns=["dateofHol", "holiday"])
holidays_df

# see if the date is an holiday or not
Swe_hol=holidays.Sweden(years=[2019, 2022])

DataAsli['is_holiday'] = [date in Swe_hol for date in DataAsli['date2']]

#---------------------------------------------------------------------------------

data=DataAsli.copy()

#We use dataset to find out the testing dates

DataSet = data[['date','hour','is_holiday','dayofweek','Load24','Load168','Load']]

#Data Set1 goes as input to the ANN so define the features here, the last one should be the load 

DataSet1 = data[['hour','is_holiday','dayofweek','Load24','Load168','Load']]
DataSet.dropna(inplace=True)

Load=DataSet['Load']

DataSet.dropna(inplace=True)

TestingDate='2022-07-01'

DataTrain=DataSet[DataSet['date'] < TestingDate].copy()
DataTest=DataSet[DataSet['date'] >= TestingDate].copy()


#array of dataTrain and datatest

DataTrain=DataSet[0:len(DataTrain)+1]
DataTest=DataSet[len(DataTrain)+1:]

#raise SystemExit(0)

#----------------------------Scaling------------------
sc = MinMaxScaler(feature_range = (0, 1))
dataset = sc.fit_transform(DataSet1)


#Just for saving the minmax scalers------------------------------

DataSet2 = data[['hour','is_holiday','dayofweek','Load24','Load168']]

scaler_filename = 'scaler_predictors_' + tag +'.save'

sc2 = MinMaxScaler(feature_range = (0, 1))
dataset22 = sc2.fit_transform(DataSet2)
joblib.dump(sc2, scaler_filename) 

#--------------------------------------------------------------
#Just for saving the minmax scalers------------------------------

DataSet3 = data[['Load24']]

scaler_filename = 'Scalar_Load_'+tag +'.save'

sc3 = MinMaxScaler(feature_range = (0, 1))
dataset3 = sc3.fit_transform(DataSet3)
joblib.dump(sc3, scaler_filename) 



# -------------- number of features -----------------

# dataset = DataSet.to_numpy()
# split into input (X) and output (y) variables
X = dataset[:, 0:-1]
y=dataset[:,-1]
X, y = X.astype('float'), y.astype('float')
n_features = X.shape[1]

#
DataTrain_Scal=dataset[0:len(DataTrain)]
DataTest_Scal=dataset[len(DataTrain)+1:]
X_train=DataTrain_Scal[:,0:-1]
X_test=DataTest_Scal[:,0:-1]
y_train=DataTrain_Scal[:,-1]
y_test=DataTest_Scal[:,-1]
import time
from keras.layers import LSTM
from tensorflow.keras import layers
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Sequential, load_model
# univariate multi-step encoder-decoder convlstm
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import MaxPooling2D

from numpy.random import seed
seed(1234)  # seed random numbers for Keras
import tensorflow 
tensorflow.random.set_seed(2)  # seed random numbers for Tensorflow backend
import pandas as pd
# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape) 
# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].
#Model architecture: 1) LSTM with 100 neurons in the first visible layer, 3) dropout 20%, 4) 1 neuron in the output layer for predicting Global_active_power, 5) The input shape will be 1 time step with 7 features, 6) I use the Mean Absolute Error (MAE) loss function and the efficient Adam version of stochastic gradient descent, 7) The model will be fit for 20 training epochs with a batch size of 70.
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(70, input_shape=(X_train.shape[0], X_train.shape[1])))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=2, shuffle=False)
model.save('LSTM for PV Output Prediction.hdf5')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

from keras.callbacks import History 
history = History()
    
# evaluate on test set
yhat_tr=model.predict(X_train)
yhat = model.predict(X_test)


error = mean_absolute_error(y_test, yhat)

#Save  scalars for prediction script
Y_Predicted=sc3.inverse_transform(yhat)

#L=np.ones(len(yhat))*(min(Load))

#Y_Predicted=(yhat*(max(Load)-min(Load)))+(L.reshape(len(L),1))


#--------------- we use these just for plots-------------------------------
X_tr=DataTrain[['hour','is_holiday','dayofweek','Load24','Load168','Load',]]
Y_tr=DataTrain['Load']
X_tes=DataTest[['hour','is_holiday','dayofweek','Load24','Load168','Load',]]
Y_tes=DataTest['Load']
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



#-------------------------Plots-----------------------------------------------------


PP=Y_tes.to_numpy()
plt.plot(Y_Predicted, color='blue',label='Predicted')
plt.plot(PP, color='red',label='Actual')
#plt.axis([0, 168, 0, 15])
plt.title('LSTM Model Validation')
plt.xlabel('Time(Hours)')
plt.ylabel('Load (kW)')
plt.legend()
plt.show()
# error of Scaled values
print('MAE of scaled: %.3f' % error)

# Y_Predicted=(Predicted_test*(max(yy1)-min(yy1))).flatten()+[np.ones(len(Predicted_test))*(min(yy1))]
print( '------- Results & Accuracy For Test   -----------')
print('--------- MSE ------')
mse = met.mean_squared_error(Y_tes, Y_Predicted)
print(mse)
print('-------- RMSE ---------')
rmse = met.mean_squared_error(Y_tes, Y_Predicted)**0.5
print(rmse)
print('-------- MAE ------')
mae = met.mean_absolute_error(Y_tes, Y_Predicted)
print(mae)
print('-------- MAPE -------')
mape = 100*met.mean_absolute_percentage_error(Y_tes, Y_Predicted)
print(mape)

print('-------- R2 -------')
r2 = math.sqrt(r2_score(Y_tes, Y_Predicted))
print(r2)
#Y_tes=np.array(Y_tes)
#fig, ax = plt.subplots()
#ax.scatter(Y_tes, Y_Predicted)
#ax.plot([Y_tes.min(), Y_tes.max()], [Y_tes.min(), Y_tes.max()], 'k--', lw=4)
#ax.set_xlabel('Actual PV')
#ax.set_ylabel('Predicted PV')
#regression line
#Y_tes, Y_Predicted = Y_tes.reshape(-1,1), Y_Predicted.reshape(-1,1)
#ax.set_title('R2 Test: ' + str(math.sqrt(r2_score(Y_tes, Y_Predicted))))
#plt.grid()
#plt.show()
 
# -----------------------------------Save the Trained model--------------------
model.save('trainedLSTM'+tag +'.h5')  # Saves the trained model with tag name.h5'
 
 
#------------------------------------------------------------------------------
