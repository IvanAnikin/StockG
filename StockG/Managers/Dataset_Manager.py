import yfinance as yf
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os, re
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
            # elif self.args['dataset_url'] in self.args:
            if 'dataset_path' in self.args:
                self.dataset = pd.read_csv(self.args['dataset_path'])
            elif 'dataset_info' in self.args:
                print("Reading data for " + self.args['dataset_info']['name'])
                if 'start' not in self.args['dataset_info']:
                    self.dataset = yf.download(self.args['dataset_info']['name'])
                else:
                    self.dataset = yf.download(self.args['dataset_info']['name'],
                                            start=self.args['dataset_info']['start'],
                                            end=self.args['dataset_info']['end'],
                                            progress=self.args['dataset_info']['progress'])
            elif 'dataset_path' in self.args:
                self.dataset = pd.read_csv(self.args['dataset_path'])

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
            if prop == 'SMA' or prop == 'SMA_Volume':
                self.add_means(prop)
            elif prop == 'ATR':
                self.add_tr()
            elif prop == 'ST':
                self.add_stochastic()
            elif prop == 'RSI':
                self.add_diff()
            elif prop in ['ADX_15', 'ADX_25']:
                ad_value = int(re.search('(\d+)', prop).group(0))
                self.add_dx(ad_value)
            columns.append(prop)

        for index in range(int(l - (l / r))):
            date = date_list[index]
            # step = int(index / n)
            round = int(index / r)
            # left = index % r
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
                        if prop in ['SMA', 'SMA_Volume', 'ATR', 'ST', 'RSI']:
                            if s == 0:
                                expected_list = self.dataset[params.expected_mapping[prop]]
                                window = expected_list[date_list[index - n]:date_list[index - 1]].tolist()
                            else:
                                initial_data = self.dataset[f'{prop}_{s + 1}']
                                for z in range(n):
                                    curr = index - s * (n - z - 1)
                                    window.append(initial_data.iloc[curr])
                        elif prop in ['ADX_15', 'ADX_25']:
                            ad_value = int(re.search('(\d+)', prop).group(0))
                            if index < 15 * r + ad_value:
                                expected_list = self.dataset[params.expected_mapping[prop]]
                                window = expected_list[date_list[index - n]:date_list[index - 1]].tolist()
                            else:
                                initial_data = self.dataset[prop]
                                for z in range(n):
                                    curr = index - s * (n - z - 1)
                                    window.append(initial_data.iloc[curr])
                        # else:
                        #     initial_data = pd.DataFrame()

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


    def wilder(self, data, periods):
        start = np.where(~np.isnan(data))[0][0]  # Check if nans present in beginning
        Wilder = np.array([np.nan] * len(data))
        Wilder[start + periods - 1] = data[start:(start + periods)].mean()  # Simple Moving Average
        for i in range(start + periods, len(data)):
            Wilder[i] = (Wilder[i - 1] * (periods - 1) + data[i]) / periods  # Wilder Smoothing
        return (Wilder)


    def add_stochastic(self):
        r = self.args["r"]
        self.dataset['ST'] = ((self.dataset['Close'] - self.dataset['Low']) / (
                    self.dataset['High'] - self.dataset['Low'])) * 100
        for i in range(2, int(len(self.dataset) / r) + 1):
            self.dataset[f'Lowest_{i}D'] = self.dataset['Low'].transform(lambda x: x.rolling(window=i).min())
            self.dataset[f'High_{i}D'] = self.dataset['High'].transform(lambda x: x.rolling(window=i).max())

            self.dataset[f'Stochastic_{i}'] = ((self.dataset['Close'] - self.dataset[f'Lowest_{i}D']) / (
                    self.dataset[f'High_{i}D'] - self.dataset[f'Lowest_{i}D'])) * 100

            self.dataset[f'ST_{i}'] = self.dataset[f'Stochastic_{i}'].rolling(window=i).mean()
        print(self.dataset.keys())


    def add_diff(self):
        RSI_data = self.dataset.copy()
        r = self.args["r"]
        self.dataset['Diff'] = RSI_data['Close'].transform(lambda x: x.diff())
        self.dataset['Diff'] = np.where(np.isnan(self.dataset['Diff']),0, self.dataset['Diff'])

        RSI_data['Up'] = self.dataset['Diff']
        RSI_data.loc[(RSI_data['Up'] < 0), 'Up'] = 0

        RSI_data['Down'] = self.dataset['Diff']
        RSI_data.loc[(RSI_data['Down'] > 0), 'Down'] = 0
        RSI_data['Down'] = abs(RSI_data['Down'])

        self.dataset['Diff'] = abs(self.dataset['Diff'])

        for i in range(2, int(len(self.dataset) / r) + 1):
            RSI_data[f'avg_{i}up'] = RSI_data['Up'].transform(lambda x: x.rolling(window=5).mean())
            RSI_data[f'avg_{i}down'] = RSI_data['Down'].transform(lambda x: x.rolling(window=5).mean())
            RSI_data[f'RS_{i}'] = RSI_data[f'avg_{i}up'] / RSI_data[f'avg_{i}down']
            self.dataset[f'RSI_{i}'] = 100 - (100 / (1 + RSI_data[f'RS_{i}']))


    def add_tr(self):
        self.dataset['prev_close'] = self.dataset['Close'].shift(1)
        self.dataset['prev_close'].iloc[0] = self.dataset['Close'].iloc[0]
        self.dataset['TR'] = np.maximum((self.dataset['High'] - self.dataset['Low']),
                                        np.maximum(abs(self.dataset['High'] - self.dataset['prev_close']),
                                                   abs(self.dataset['prev_close'] - self.dataset['Low'])))
        TR_data = self.dataset.copy()
        r = self.args["r"]
        for i in range(2, int(len(self.dataset) / r) + 1):
            self.dataset[f'ATR_{i}'] = self.wilder(TR_data['TR'], i)
        print(self.dataset.keys())


    def add_dx(self, i):
        ADX_data = self.dataset.copy()
        ADX_data['prev_high'] = self.dataset['High'].shift(1)
        ADX_data['prev_low'] = self.dataset['Low'].shift(1)

        ADX_data['pDM'] = np.where(~np.isnan(ADX_data.prev_high),
                                   np.where((self.dataset['High'] > ADX_data['prev_high']) &
                                            (((self.dataset['High'] - ADX_data['prev_high']) > (
                                                    ADX_data['prev_low'] - self.dataset['Low']))),
                                            self.dataset['High'] - ADX_data['prev_high'],0), np.nan)

        ADX_data['mDM'] = np.where(~np.isnan(ADX_data.prev_low),
                                   np.where((ADX_data['prev_low'] > self.dataset['Low']) &
                                            (((ADX_data['prev_low'] - self.dataset['Low']) > (
                                                    self.dataset['High'] - ADX_data['prev_high']))),
                                             ADX_data['prev_low'] - self.dataset['Low'],0), np.nan)

        ADX_data['pDI'] = (ADX_data['pDM'] / ADX_data['TR']) * 100
        ADX_data['mDI'] = (ADX_data['mDM'] / ADX_data['TR']) * 100
        self.dataset['DX'] = np.where(ADX_data['pDI'] == 0, 0, 0)

        ADX_data[f'pDM_{i}'] = self.wilder(ADX_data['pDM'], i)
        ADX_data[f'mDM_{i}'] = self.wilder(ADX_data['mDM'], i)

        ADX_data[f'pDI_{i}'] = (ADX_data[f'pDM_{i}'] / ADX_data[f'ATR_{i}']) * 100
        ADX_data[f'mDI_{i}'] = (ADX_data[f'mDM_{i}'] / ADX_data[f'ATR_{i}']) * 100

        ADX_data[f'DX_{i}'] = (
            np.round(abs(ADX_data[f'pDI_{i}'] - ADX_data[f'mDI_{i}']) / (ADX_data[f'pDI_{i}'] + ADX_data[f'mDI_{i}']) * 100))

        self.dataset[f'ADX_{i}'] = self.wilder(ADX_data[f'DX_{i}'], i)

        print(self.dataset.keys())

        # ADX_data.to_csv('ADX.csv', index=False)
        # start = self.args['dataset_info']['start']
        # end = self.args['dataset_info']['end']
        # fig = plt.figure(facecolor='white', figsize=(20, 10))
        #
        # ax0 = plt.subplot2grid((6, 4), (1, 0), rowspan=4, colspan=4)
        # ax0.plot(self.dataset.loc[start:end, [f'ADX_{i}']])
        # ax0.set_facecolor('ghostwhite')
        # ax0.legend([f'ADX_{i}'], ncol=3, loc='upper left', fontsize=15)
        # plt.title("Average directional Index", fontsize=20)
        #
        # plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
        # plt.show()


    # Add SMA means
    def add_means(self, prop):
        r = self.args["r"]
        for i in range(2, int(len(self.dataset) / r) + 1):
            self.dataset[f'{prop}_{i}'] = self.dataset[params.expected_mapping[prop]].transform(lambda x: x.rolling(window=i).mean())
        print(self.dataset.keys())


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
