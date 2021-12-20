import urllib.request
import json
import os
import ssl
import pandas as pd

from StockG.Managers.Model import Model

model = Model({'url':'http://21ed1e07-2233-486a-af57-47fe5585ce4c.westeurope.azurecontainer.io/score'})

with open("E:/VS_Projects/StockG_data/S&P_500/full/SNP_test.csv") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    for row in csv_reader:

        prediction = model.predict(input=row)


