import pandas as pd
import json
import requests
from datetime import date, datetime, timedelta
from requests.auth import HTTPBasicAuth
from dateutil import parser
import numpy as np
import holidays
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import joblib
import tensorflow
import datetime
import pytz
import sys, os

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def Predict_Load(tag,startdate,enddate):
    def read_data(dt_start, resolution='H', dt_end=''):
        
        if dt_end == '':
            dt_end = dt_start

        Timestamp1 = dt_end.strftime("%d-%b-%Y %H:%M:%S")
        Timestamp1 = Timestamp1.replace(' ', '%20')
        Timestamp2 = dt_start.strftime("%d-%b-%Y %H:%M:%S")
        Timestamp2 = Timestamp2.replace(' ', '%20')
        url = 'https://hll-api.livinglab.chalmers.se:3001/api/keyqueries?key=$2a$06$ebFvep2QAJa1uQ0OMWvmA.oV5VZP8uso08IgJ30rmKv0W.82JAx1m&start='+Timestamp2+'&stop='+Timestamp1
        response = requests.get(url)
        data = pd.DataFrame(response.json(), columns=['lo_time', 'hi_time', 'avg_pload', 'avg_pvp', 'avg_ulq1', 'avg_ulq2', 'avg_ulq3', 'timestamp'])
        df = data.loc[:,[ 'lo_time', 'avg_pload']]
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])

        # Removes very big and negetive values-----------------------------------------
        # bigNum = 20000
        # for x in df.index:
        #     if df.loc[x, "avg_pload"] < 0 or df.loc[x, "avg_pload"] > bigNum:
        #         df.drop(x, inplace=True)
        # ------------------------------------------------------------------------------

        # Set the timestamp as the index to be able to use resample for datetime dfs
        df.set_index("lo_time", inplace=True)

        # Resample the data by the resolution and the mean
        df = df.resample(resolution).mean()
        # convert the unit to kW (BES is in W, the rest are in kW)
        df.loc[:, "avg_pload"] = df.loc[:, "avg_pload"]
        df = df.loc[dt_start:dt_end]/1000
        #df.to_csv(saving_path)
        df.index.name = 'timestamp'
        
        return df


# --------------------------Main part of program-------------------------------------


    # prediction interval
    dt_start = datetime.datetime.strptime(startdate, "%Y-%m-%d %H:%M%z")
    dt_end = datetime.datetime.strptime(enddate, "%Y-%m-%d %H:%M%z")
    #returned_df = read_data(dt_start=dt_start, fc_api_tag=fc_api_tag, \
    #                       saving_path=saving_path, resolution='H', dt_end=dt_end)

    # reading 24 hour before historical data
    yesterday_start = dt_start - timedelta(days=1)
    yesterday_end = dt_end - timedelta(days=1)

    Yestreday_df = read_data(dt_start=yesterday_start, resolution='H', dt_end=yesterday_end)
    print(Yestreday_df)
    # reading 168 hour before historical data
    LastWeek_start = dt_start - timedelta(days=7)
    LastWeek_end = dt_end - timedelta(days=7)

    LastWeek_df = read_data(dt_start=LastWeek_start, resolution='H', dt_end=LastWeek_end)
    print(LastWeek_df)
    Load_Yes_ave = Yestreday_df.copy()
    Load_LastWeek_ave = LastWeek_df.copy()

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
    DataAsli['hour'] = xx.index.hour
    DataAsli['dayofweek'] = DataAsli.index.weekday
    DataAsli['date'] = DataAsli.index.date

    Swe_hol = holidays.Sweden(years=[2019, 2024])
    DataAsli['is_holiday'] = [date in Swe_hol for date in DataAsli['date']]
    print(Load_Yes_ave.values)
    DataAsli['Load24'] = Load_Yes_ave.values/1000
    DataAsli['Load168'] = Load_LastWeek_ave.values/1000

    data = DataAsli.copy()
    DataSet = data[['hour', 'is_holiday', 'dayofweek', 'Load24', 'Load168']]
    print(DataSet)
    # Data Set1 goes as input to the ANN so define the features here, the last one should be the load

    # DataSet1 = data['hour','is_holiday','dayofweek','Load24','Load168','Load']

    DataTest = DataSet
    DataTest = DataSet[0:len(DataTest)]

    X_tes = DataTest[['hour', 'is_holiday', 'dayofweek', 'Load24', 'Load168']]
    # this is th scaling metrics for scaling the inputs to the model, should be imported from the train script to be consistent
    sc1 = joblib.load(resource_path(r'Prediction Models\Electricity\scaler_predictors_') + tag + '.save')

    dataset = sc1.transform(DataSet)
    
    DataTest_Scal = dataset[0:len(DataTest)]
    X_test = DataTest_Scal
    X_test = tensorflow.expand_dims(X_test, axis=1)
    # y_test=DataTest_Scal[:,-1]

    # Load the trained model
    model = load_model(resource_path(r'Prediction Models\Electricity\trainedLSTM') + tag + '.h5')
    # predict with the loaded model
    yhat_te = model.predict(X_test)
    # Unscale with loaded scale metrics
    sc3 = joblib.load(resource_path(r'Prediction Models\Electricity\Scalar_Load_') + tag + '.save')
    Y_Predicted = sc3.inverse_transform(yhat_te)

    # Load the statistics of the prediction and pass it to output
    stats = joblib.load(resource_path(r'Prediction Models\Electricity\stats_') + tag + '.save')
    mean_temp = stats['mean'].loc[0]
    std_temp = stats['standard deviation'].loc[0]

    # Create dataframe with the predicted values and put the statitic metrics as well
    predicted = pd.DataFrame(Y_Predicted, columns=['predictedValue'])
    predicted['timestamp'] = pd.date_range(start=dt_start, end=dt_end, freq='H')

    predicted.set_index('timestamp', inplace=True)

    for i in predicted.index:
        predicted.loc[i, 'mean'] = mean_temp
        predicted.loc[i, 'standardDeviation'] = std_temp
    predicted.to_csv(resource_path(r'Prediction Models\Electricity\Predicted_Load_') + tag + '.csv')


    return predicted
# ---------------------------------------------------------------------------------

# tag = 'HSBLL_Load'

# startdate = "2023-08-03 00:00+0000"
# enddate =   "2023-08-04 23:00+0000"


# PredictedLoad= Predict_Load(tag,startdate,enddate)
# print(PredictedLoad)
# output_file = 'PredictedLoad.csv'
# PredictedLoad.to_csv(output_file, index=True)





