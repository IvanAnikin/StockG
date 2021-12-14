import yfinance as yf
import matplotlib.pyplot as plt
import csv
import pandas as pd
import json
import os

class Datasets_Manager():
    def __init__(self, args):
        self.args=args

        if 'dataset_info' in self.args: self.dataset=self.load_dataset(dataset_info=self.args['dataset_info'])

    def load_dataset(self, dataset_info):
        try:
            if 'start' not in dataset_info or 'start' not in dataset_info:
                dataset = yf.download(dataset_info['name'])
            else:
                dataset = yf.download(dataset_info['name'],
                                        start=dataset_info['start'],
                                        end=dataset_info['end'],
                                        progress=dataset_info['progress'])
        except Exception as e:
            return e

        return dataset

    def load_datasets_csv(self, file_name, folder_name):
        path = "../Train/data/{folder}".format(folder=folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(file_name) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                dataset_info = {'name':row['Symbol']}
                dataset = self.load_dataset(dataset_info)

                dataset.to_csv("{path}/{name}.csv".format(path=path, name=dataset_info['name']), index=False) #, header=True

    def load_company_info(self, name): # !!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!! !!!!!  constituents-financials_csv.csv
        return name

    def visualise_dataset_close(self, **dataset):
        if not dataset:
            if self.dataset is None: dataset = self.load_dataset(self.args)
            else: dataset = self.dataset
            if dataset is None: return False


        close = dataset['Close']

        # Calculate the 20 and 100 days moving averages of the closing prices
        short_rolling = close.rolling(window=20).mean()
        long_rolling = close.rolling(window=100).mean()

        # Plot everything by leveraging the very powerful matplotlib package
        fig, ax = plt.subplots(figsize=(16, 9))

        ax.plot(close.index, close, label=self.args['dataset_info']['name'])
        ax.plot(short_rolling.index, short_rolling, label='20 days rolling')
        ax.plot(long_rolling.index, long_rolling, label='100 days rolling')

        ax.set_xlabel('Date')
        ax.set_ylabel('Adjusted closing price ($)')
        ax.legend()
        plt.show()

    def convert_to_json(self, dataset):
        data = []
        key_list = dataset['Close'].keys()
        for i in range(len(dataset)):
            data.append({
                "date": str(key_list[i].strftime("%Y-%m-%d")),
                "open": dataset['Open'][i],
                "high": dataset['High'][i],
                "low": dataset['Low'][i],
                "close": dataset['Close'][i]
            })
        return data