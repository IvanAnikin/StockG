import urllib.request
import json
import os
import ssl
import pandas as pd
import csv

from StockG.Managers.Model import Model
from StockG import params

model = Model(params.model_args)

with open("E:/VS_Projects/StockG_data/S&P_500/full/SNP_test.csv") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    for row in csv_reader:

        prediction = model.predict(input=row)


