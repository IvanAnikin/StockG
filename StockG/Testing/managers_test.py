import datetime as dt

from StockG.Managers.Dataset_Manager import Datasets_Manager
from StockG import params

from csv import reader
company_list = []
category = "S&P500"

dataset_args = params.dataset_args
manager = Datasets_Manager(dataset_args)

# with open('s&p_largest_50.csv', 'r') as read_obj:
#     csv_reader = reader(read_obj)
#     header = next(csv_reader)
#     if header != None:
#         for row in csv_reader:
#             name = row[0]
#             dataset_args['dataset_info']['name'] = name
#             dataset_args['store_path'] = "{dir}/{category}/{name}.csv".format(dir = dataset_args['dir'], category = category, name=name)
#             manager.args = dataset_args
#             manager.load_dataset()
#             manager.preprocess_dataset()

# company_list = ['AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'FB', 'JPM', 'JNJ', 'XOM', 'BAC', 'WMT', 'WFC', 'V', 'BRK.B', 'T', 'HD', 'CVX', 'UNH', 'INTC', 'PFE', 'VZ', 'PG', 'BA', 'ORCL', 'CSCO', 'C', 'KO', 'MA', 'CMCSA', 'ABBV', 'DWDP', 'PEP', 'DIS', 'PM', 'MRK', 'IBM', 'MMM', 'NVDA', 'GE', 'MCD', 'AMGN', 'MO', 'NFLX', 'HON', 'MDT', 'GILD', 'NKE']

name = dataset_args['dataset_info']['name']
dataset_args['store_path'] = "{dir}/{category}/{name}.csv".format(dir = dataset_args['dir'], category = category, name=name)
manager.args = dataset_args
manager.load_dataset()
manager.preprocess_dataset()

