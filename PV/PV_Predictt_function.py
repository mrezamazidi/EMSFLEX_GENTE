##3##
import pandas as pd
import sys, os
import requests
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as met
from sklearn.metrics import r2_score
# from sklearn import datasets, linear_model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# import scipy.optimize as opt
# import matplotlib.pyplot as plt
# import pandas as pd
# import sklearn.metrics as met
# from os import system
# from keras.models import model_from_json
# from sklearn.metrics import r2_score
# from sklearn.preprocessing import MinMaxScaler
# from colorama import Fore
from keras.models import load_model
# from tensorflow import keras
import joblib
# from datetime import date, datetime, timedelta
import datetime
import pytz
import tensorflow 

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)




tensorflow.random.set_seed(2)  # seed random numbers for Tensorflow backend


def Predict_PV(tag, startdate, enddate):
    def helper(x):
        '''
        this helper returns the earliest ref_datetime to a given valid_datetime
        '''

        # convert string formatted datetime to numeric date time format, thus allowing mathematical arithmetic

        x['valid_datetime'] = pd.to_datetime(x['valid_datetime'], format="%Y-%m-%dT%H:%M:%SZ")
        x['ref_datetime'] = pd.to_datetime(x['ref_datetime'], format="%Y-%m-%dT%H:%M:%SZ")
        global indx1
        # find the index of the ref_datetime that is closest to the given valid_datetime
        indx = (x['valid_datetime'].values[0] - x['ref_datetime'].values).argmin(0)
        indx1 = (x['valid_datetime'].values[0] - x['ref_datetime'].values).argmin(0)
        # for hours 6,12,18 24 when the weather model runs the last reftime should be used

        t1 = pd.Timestamp(x['valid_datetime'].values[0])
        if t1.hour == 0 or t1.hour == 6 or t1.hour == 12 or t1.hour == 18:
            indx = indx - 1
            if indx == -1:
                indx = 0
        # convert back the  numeric datetime to string format and return the latest ref_datetime
        out = x['ref_datetime'].values[indx]
        out = pd.to_datetime(str(out))
        out = out.strftime("%Y-%m-%dT%H:%M:%SZ")
        return   pd.Series({'latest_ref_time': out})

    # print("groupby the dateframe by valid_datetime and find the latest ref_time")
    tag = 'HSBLL_PV'

    # Read the weather data for 'DateToPredict'



    dt_start = datetime.datetime.strptime(startdate, "%Y-%m-%d %H:%M%z")
    dt_end = datetime.datetime.strptime(enddate, "%Y-%m-%d %H:%M%z")
    dt_start = dt_start.astimezone(pytz.UTC)
    dt_end = dt_end.astimezone(pytz.UTC)
#    start = dt_start.replace(tzinfo=None)
#    end = dt_end.replace(tzinfo=None)

    fc_api_tag = 'HSBLL_PV'

    # DataWeather = pd.read_csv(r'C:\Users\maryamar\OneDrive - Chalmers\Desktop\Prediction python\Rebase\'model_7MetNo_MEPS_2022-05-24_to_2022-05-24.csv')

    url = "https://api.rebase.energy/weather/v2/query"
    headers = {"Authorization": 'YCERPUQ7VUj0mPmM1reB6PrVBhORfP6iKAvpaZUDmDQ'}
    params = {
        'model': 'MetNo_MEPS',
        'start-date': dt_start,
        'end-date': dt_end,
        'reference-time-freq': '6H',
        'forecast-horizon': 'full',
        'latitude': '57.688684',
        'longitude': '11.977383',
        'variables': 'CloudCover,PressureReducedMSL,RelativeHumidity,SolarDownwardRadiation,Temperature,WindDirection,WindSpeed'
    }
    response = requests.get(url, headers=headers, params=params)
    result = response.json()
    DataWeather = pd.DataFrame.from_dict(result)

    # Get the updated weather predictions from the dataframe of weather API

    latest_ref_time = DataWeather.groupby(['valid_datetime']).apply(lambda x: helper(x))

    DataWeather_processed = latest_ref_time.reset_index().merge(DataWeather,
                                                                left_on=['valid_datetime', 'latest_ref_time'],
                                                                right_on=['valid_datetime', 'ref_datetime'], how='left')



    for i in DataWeather_processed.index:
        DataWeather_processed.loc[i, "valid_datetime"] = datetime.datetime.strptime(
            DataWeather_processed.loc[i, 'valid_datetime'], "%Y-%m-%dT%H:%M:%SZ")
        DataWeather_processed.loc[i, "valid_datetime"] = DataWeather_processed.loc[i, "valid_datetime"].replace(
            tzinfo=pytz.UTC)
    #    DataWeather_processed.loc[i, "valid_datetime2"]=datetime.datetime.strftime(DataWeather_processed.loc[i,'valid_datetime'],"%Y-%m-%d %H:%M%z")
    #    DataWeather_processed.loc[i, "valid_datetime2"]=datetime.datetime.strptime(DataWeather_processed.loc[i,'valid_datetime2'],"%Y-%m-%d %H:%M")

    mask = (DataWeather_processed['valid_datetime'] >= (dt_start)) & (
                DataWeather_processed['valid_datetime'] <= (dt_end))
    DataWeather_processed = (DataWeather_processed.loc[mask])
#    DataWeather_processed = DataWeather_processed.interpolate()

    # --------------------------------
    # Making a data frame with timestamps of forecast horizon

    DataAsli = pd.DataFrame()
    DataAsli['timestamp'] = pd.date_range(start=dt_start, end=dt_end, freq='H')
    # Set the timestamp as the index to be able to use resample for datetime dfs
    xx = DataAsli.copy()

    DataAsli.set_index("timestamp", inplace=True)
    # Extracting calander needed features for the model

    # This extract the hour in UTC which was also used for training
    UTCstart = dt_start.astimezone(pytz.UTC)
    UTCend = dt_end.astimezone(pytz.UTC)
    xx['timestamp'] = pd.date_range(start=UTCstart, end=UTCend, freq='H')
    xx.set_index("timestamp", inplace=True)
    # DataAsli['Hour'] =DataAsli.index.hour
    DataAsli['Hour'] = xx.index.hour
    DataAsli['Day'] = DataAsli.index.day
    DataAsli['Month'] = DataAsli.index.month

    DataAsli['SolarDownwardRadiation'] = DataWeather_processed['SolarDownwardRadiation'].values
    DataAsli['RelativeHumidity'] = DataWeather_processed['RelativeHumidity'].values
    DataAsli['CloudCover'] = DataWeather_processed['CloudCover'].values
    DataAsli['Temperature'] = DataWeather_processed['Temperature'].values
    DataAsli['WindDirection'] = DataWeather_processed['WindDirection'].values
    DataAsli['WindSpeed'] = DataWeather_processed['WindSpeed'].values

    # If the values of radiation is negative put it to zero
    DataAsli.loc[DataAsli['SolarDownwardRadiation'] < 0, 'SolarDownwardRadiation'] = 0
    # the data of solar which is used by training is the Flux so to change  W/s solar radiation  to flux we multiply it by 3600
    #DataAsli.loc[:, 'SolarDownwardRadiation'] *= 3600
    # The data of humidity for training was in between 0-1 but now it returns it in percent
    #DataAsli.loc[:, 'RelativeHumidity'] /= 100

    data = DataAsli.copy()
    DataSet = data[['Hour','SolarDownwardRadiation','RelativeHumidity', 'CloudCover', 'Temperature', 'WindDirection', 'WindSpeed']]

    DataTest = DataSet
    DataTest = DataSet[0:len(DataTest)]

    X_tes = DataTest[['Hour','SolarDownwardRadiation','RelativeHumidity', 'CloudCover', 'Temperature', 'WindDirection', 'WindSpeed']]

    # this is th scaling metrics for scaling the inputs to the model, should be imported from the train script to be consistent
    sc1 = joblib.load(resource_path(r'Prediction Models\PV\Scalar_predictors_') + fc_api_tag + '.save')

    dataset = sc1.transform(DataSet)

    DataTest_Scal = dataset[0:len(DataTest)]
    X_test = DataTest_Scal
    X_test = tensorflow.expand_dims(X_test, axis=1)

    # Load the trained model
    model = load_model(resource_path(r'Prediction Models\PV\trainedLSTM_PV_') + fc_api_tag + '.save')
    # predict with the loaded model
    yhat_te = model.predict(X_test)
    # Unscale with loaded scale metrics
    sc3 = joblib.load(resource_path(r'Prediction Models\PV\Scalar_Y_') + fc_api_tag + '.save')
    Y_Predicted = sc3.inverse_transform(yhat_te)

    # Load the statistics of the prediction and pass it to output
    stats = joblib.load(resource_path(r'Prediction Models\PV\stats_') + fc_api_tag + '.save')
    mean_temp = stats['mean'].loc[0]
    std_temp = stats['standard deviation'].loc[0]

    # Create dataframe with the predicted values and put the statitic metrics as well
    predicted_PV = pd.DataFrame(Y_Predicted, columns=['predictedValue'])
    predicted_PV.loc[predicted_PV['predictedValue'] < 0.9, 'predictedValue'] = 0

     
    predicted_PV['timestamp'] = pd.date_range(start=dt_start, end=dt_end, freq='H')
    
    predicted_PV.set_index('timestamp', inplace=True)

    for i in predicted_PV.index:
        predicted_PV.loc[i, 'mean'] = mean_temp
        predicted_PV.loc[i, 'standardDeviation'] = std_temp
        predicted_PV.to_csv('Predicted_PV_' + fc_api_tag + '.csv')

    return (predicted_PV)


# ----------------------------------------Main------------------------------------
# tag = 'HSBLL_PV'

# # Read the weather data for 'DateToPredict'

# startdate = "2023-08-03 00:00+0000"
# enddate =   "2023-08-04 23:00+0000"

# Predicted_PV = Predict_PV(tag, startdate, enddate)
# print(Predicted_PV)
# # Save predicted results as CSV
# predicted_file = 'Predicted_PV.csv'
# Predicted_PV.to_csv(predicted_file, index=True)
