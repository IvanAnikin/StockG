import yfinance as yf
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
import datetime
import numpy as np

class Datasets_Manager():
    def __init__(self, args=[]):
        self.args=args

        if 'dataset_info' in self.args or 'dataset_path' in self.args or 'dataset_url' in self.args:
            self.dataset=self.load_dataset()
            if 'preprocess' in self.args: self.preprocessed_dataset=self.preprocess_dataset(self.dataset, self.args['preprocess'])

    # LOADING
    def load_dataset(self):
        try:
            if self.args['dataset_path'] != "":
                dataset = pd.read_csv(self.args['dataset_path'])

            # elif self.args['dataset_url'] != "":
            else:
                if 'start' not in self.args['dataset_info'] or 'start' not in self.args:
                    dataset = yf.download(self.args['dataset_info']['name'])
                else:
                    dataset = yf.download(self.args['dataset_info']['name'],
                                            start=self.args['dataset_info']['start'],
                                            end=self.args['dataset_info']['end'],
                                            progress=self.args['dataset_info']['progress'])
        except Exception as e:
            return e

        return dataset

    def load_datasets_csv_same_interval(self, dir, start, end):
        # FINDING LIMITS

        #if not os.path.exists(destination):
        #    os.makedirs(destination)
        #with open(source) as csv_file:
        #    csv_reader = csv.DictReader(csv_file, delimiter=',')
        #    max_start = datetime.datetime(1000, 1, 1)
        #    min_end = datetime.datetime.now()
        #    limits = []
        #    for row in csv_reader:
        #        dataset_info = {'name':row['Symbol']}
        #        dataset = self.load_dataset(dataset_info)

        #        keys = dataset['Close'].keys()
        #        if len(keys)==0:
        #            continue
        #        first = keys[0]
        #        last = keys[len(keys)-1]
        #        limits.append([row['Symbol'], first, last])

        #        if(first > max_start):  max_start = first
        #        if(last < min_end):     min_end = last
        #    np.save("{path}/{name}.npy".format(path=destination, name="limits"), limits)
        #    print("max_start: '{max_start}' | min_end: '{min_end}'".format(max_start=max_start, min_end=min_end))

        start_string = start.strftime("%Y-%m-%d")
        end_string = end.strftime("%Y-%m-%d")
        limits = np.load("{source}/limits.npy".format(source=dir), allow_pickle=True)
        with open("{source}/s&p_largest_50.csv".format(source=dir)) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                for limit in limits:
                    if limit[0] == row['Symbol'] and limit[1] < start:
                        with open(
                                "{source}/full/{symbol}.csv".format(source=dir, symbol=row['Symbol'])) as stock_file:
                            stock_reader = csv.DictReader(stock_file, delimiter=',')
                            new_stock = []
                            save = False
                            for stock_row in stock_reader:
                                if stock_row['Date'] == start_string:
                                    save = True
                                if save:
                                    new_stock.append(stock_row)
                            if not save:
                                print("Not save '{symbol}'".format(symbol=row['Symbol']))
                            else:
                                destination = "{source}/{start_string}_{end_string}".format(source=dir, start_string=start_string,
                                                                                              end_string=end_string)
                                if not os.path.exists(destination):
                                    os.makedirs(destination)
                                pd.DataFrame(new_stock).to_csv(
                                    "{destination}/{name}.csv".format(destination=destination, name=row['Symbol']), index=False)  # , header=

                            # new_stock=stock_reader[i:]
                            # print(i)
                            print("symbol: '{symbol}' | limit: '{limit}'".format(symbol=row['Symbol'], limit=limit[1]))
        self.load_dataset(dataset_info={"name":"SNP", "start":start_string, "end":end_string,'progress': False}).to_csv("{destination}/{name}.csv".format(destination=destination, name="SNP"))

    def save_datasets_csv(self, source, destination, start, end):
        if not os.path.exists(destination):
            os.makedirs(destination)
        with open(source) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                dataset_info = {'name':row['Symbol']}
                if start: dataset_info['start'] = start
                if end: dataset_info['end'] = end
                dataset = self.load_dataset(dataset_info)

                keys = dataset['Close'].keys()
                keys_list = keys.values
                i = 0
                for key in keys_list:
                    key2 = pd.to_datetime(key)
                    keys_list[i] = str(key2.strftime("%Y-%m-%d"))
                    i+=1
                dataset["Date"] = keys_list
                dataset.to_csv("{path}/{name}.csv".format(path=destination, name=dataset_info['name']), index=False) #, header=

    # PREPROCESSING
    def preprocess_dataset(self, dataset, args):
        preprocessed = pd.DataFrame()

        #for arg in args:
            # preprocessed.append( ***** )

        return preprocessed

    # COMBINING/SPLITTING
    # not done
    def combine_datasets(self, dir):
        merged = pd.DataFrame()
        with open("{dir}/{filename}".format(dir=dir, filename="SNP.csv")) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            i=0
            for row in csv_reader:
                columns = {}
                for column_name in pd.read_csv("{dir}/{filename}".format(dir=dir, filename="SNP.csv")):
                    if(column_name!="Date"):
                        column = []
                        for filename in os.listdir(dir):
                            #file_row = pd.read_csv("{dir}/{filename}".format(dir=dir, filename=filename)).iloc[[i]]
                            column.append(pd.read_csv("{dir}/{filename}".format(dir=dir, filename=filename)).iloc[[i]][column_name])
                        columns[column_name] = column
                i+=1
                merged.append(columns, ignore_index = True)
        print(merged)

    def split_dataset(self, dataset=[], test_percentage=0.1):
        if dataset==[]: dataset=self.load_dataset()
        breakpoint = int((1-test_percentage)*len(dataset))
        train = dataset[:breakpoint]
        test = dataset[breakpoint:]
        return train, test

    def split_dataset_files(self, path = "C:/Users/ivana/source/repos/InvestMaster/InvestMaster/InvestMaster/Train/data/S&P_500/full",
                            name = "SNP", test_percentage=0.1):
        dataset = pd.read_csv("{path}/{name}.csv".format(path=path, name=name))
        train, test = self.split_dataset(dataset=dataset, test_percentage=test_percentage)
        train.to_csv("{path}/{name}_train.csv".format(path=path, name=name), index=False)  # , header=
        test.to_csv("{path}/{name}_test.csv".format(path=path, name=name), index=False)

    # CONVERTING
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

    # VISUALISATION
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

    def visualise_testing_close_diff(self, close, prediction, date):

        fig, ax = plt.subplots(figsize=(16, 9))

        ax.plot(date, close, label="Close")
        ax.plot(date, prediction, label='Prediction')

        ax.set_xlabel('Date')
        ax.set_ylabel('Adjusted closing price ($)')
        ax.legend()
        plt.show()
