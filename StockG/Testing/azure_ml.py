import urllib.request
import json
import os
import ssl
import pandas as pd
import csv
from collections import OrderedDict
import numpy as np

from StockG.Managers.Model import Model
from StockG import params
from StockG.Managers.Dataset_Manager import Datasets_Manager

dataset_manager = Datasets_Manager()
model = Model(params.model_args)
dataset_dir = "E:/VS_Projects/StockG_data/S&P_500/full/SNP_test.csv"
target_column = 'Adj Close'

with open(dataset_dir) as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    test=pd.read_csv(dataset_dir)
    history = []
    i=0
    for row in csv_reader:

        prediction = model.predict(input=row)

        prediction = json.loads(json.loads(prediction.content))

        history.append({
            "prediction": prediction['forecast'], "truth": row[target_column],
            "diff": abs(float(prediction['forecast'][0])-float(row[target_column])),
            "Date": row['Date']
        })

        forecast = prediction['forecast'][0]
        #ew_row=row.copy()
        #new_row[target_column] = forecast
        #train.append(new_row, ignore_index=True)

        if(i%100==0 and i != 0):
            diff_lst = [d['diff'] for d in history if 'diff' in d]
            prediction_lst = [d['prediction'] for d in history if 'prediction' in d]
            dates_lst = [d['Date'] for d in history if 'Date' in d]
            close_lst = list(test['Close'].values[:i+1])
            dataset_manager.visualise_testing_close_diff(close=close_lst, prediction=prediction_lst, date=dates_lst)
            print("{i}: average difference: {avg_diff}".format(i=i, avg_diff=np.average(diff_lst[i-10:i])))


        i+=1