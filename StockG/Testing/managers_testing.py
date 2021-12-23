import pandas as pd
from StockG.Managers.Dataset_Manager import Datasets_Manager
from StockG import params

datasets_manager = Datasets_Manager(args=params.dataset_args)

#print(datasets_manager.dataset)

dataset = datasets_manager.dataset

keys = dataset['Close'].keys()
keys_list = keys.values
i = 0
for key in keys_list:
    key2 = pd.to_datetime(key)
    keys_list[i] = str(key2.strftime("%Y-%m-%d"))
    i += 1
dataset["Date"] = keys_list

print(dataset)
