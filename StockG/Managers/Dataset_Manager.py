import yfinance as yf
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
import numpy as np
from StockG import params


class Datasets_Manager():
    def __init__(self, args):
        self.args=args

        # if 'dataset_info' in self.args or 'dataset_path' in self.args or 'dataset_url' in self.args:
        #     print(self.args)
        #     self.load_dataset()
            # if 'preprocess' in self.args: self.preprocess_dataset()

    # LOADING
    def load_dataset(self):
        try:
            if 'dataset_path' in self.args:
                self.dataset = pd.read_csv(self.args['dataset_path'])

            # elif self.args['dataset_url'] in self.args:
            elif 'dataset_info' in self.args:
                print("Reading data for " + self.args['dataset_info']['name'])
                if 'start' not in self.args['dataset_info']:
                    self.dataset = yf.download(self.args['dataset_info']['name'])
                else:
                    self.dataset = yf.download(self.args['dataset_info']['name'],
                                            start=self.args['dataset_info']['start'],
                                            end=self.args['dataset_info']['end'],
                                            progress=self.args['dataset_info']['progress'])
        except Exception as e:
            return e

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
    def preprocess_dataset(self):
        preprocessed = pd.DataFrame()
        n = self.args["n"]
        r = self.args["r"]

        date_list = self.dataset['Close'].keys()
        l = len(date_list)

        # adj_close_list = self.dataset['Adj Close']

        columns = ['Date', 'Index', 'Close', 'Future']#, 'Window']

        for prop in self.args['X']['windowed_data']:
            # to_call = f'add_{prop}_means'
            # print("Calling {}".format(to_call))
            # getattr(self, to_call)()
            self.add_means(prop)
            columns.append(prop)

        for index in range(int(l - (l / r))):
            date = date_list[index]
            # step = int(index / n)
            round = int(index / r)
            left = index % r
            # print("Round:" + str(round) + "\tLeftover:" + str(left) + "\tStep:" + str(step))

            if index >= n:
                for s in range(round):
                    # print(s)
                    predicted_index = index + s + 1
                    row = self.dataset.iloc[predicted_index]
                    close = row['Close']
                    list_of_data = [date, index, close, predicted_index]

                    for prop in self.args['X']['windowed_data']:
                        window = []
                        if s == 0:
                            expected_list = self.dataset[params.expected_mapping[prop]]
                            window = expected_list[date_list[index - n]:date_list[index-1]].tolist()
                        else:
                            avg_koeff = self.dataset[f'{prop}_{s + 1}']
                            for z in range(n):
                                curr = index - (s + 1)*(n - z) + s
                                window.append(avg_koeff.iloc[curr])
                        list_of_data.append(window)

                    df1 = pd.DataFrame([list_of_data], columns=columns)
                    preprocessed = preprocessed.append(df1)

        preprocessed.to_csv(self.args['store_path'], index=False)


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


    # Add SMA means
    def add_means(self, prop):
        r = self.args["r"]
        for i in range(2, int(len(self.dataset) / r) + 1):
            self.dataset[f'{prop}_{i}'] = self.dataset[params.expected_mapping[prop]].transform(lambda x: x.rolling(window=i).mean())
        print(self.dataset.keys())

        # start = self.args['dataset_info']['start']
        # end = self.args['dataset_info']['end']
        # fig = plt.figure(facecolor='white', figsize=(20, 10))
        #
        # ax0 = plt.subplot2grid((6, 4), (1, 0), rowspan=4, colspan=4)
        # ax0.plot(self.dataset.loc[start:end, ['Close', 'SMA_2', 'SMA_12', 'SMA_25']])
        # ax0.set_facecolor('ghostwhite')
        # ax0.legend(['Close', 'SMA_2', 'SMA_12', 'SMA_25'], ncol=3, loc='upper left', fontsize=15)
        # plt.title("Stock Price, Slow and Fast Moving Average", fontsize=20)
        #
        # plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
        # plt.show()


    # CONVERTING
    def convert_to_json(self):
        data = []
        key_list = self.dataset['Close'].keys()
        l = len(self.dataset)
        n = self.args["n"]
        max_koeff = int(l / n)
        # print(max_koeff)
        for i in range(l):
            item = {"date": str(key_list[i].strftime("%Y-%m-%d")),
                "open": self.dataset['Open'][i],
                "high": self.dataset['High'][i],
                "low": self.dataset['Low'][i],
                "close": self.dataset['Close'][i]}
            for j in range(2, max_koeff + 1):
                item[f'SMA_{j}'] = self.dataset[f'SMA_{j}'][i]
            data.append(item)
            # data.append({
            #     "date": str(key_list[i].strftime("%Y-%m-%d")),
            #     "open": self.dataset['Open'][i],
            #     "high": self.dataset['High'][i],
            #     "low": self.dataset['Low'][i],
            #     "close": self.dataset['Close'][i]
            # })
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
