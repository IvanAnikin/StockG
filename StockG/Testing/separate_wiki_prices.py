import csv
import pandas as pd

df = pd.read_csv('./data/WIKI_PRICES.csv')
with open("./data/wiki_stocks.csv") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    for row in csv_reader:
        dataset = df.loc[df['ticker'] == row["code"]]

        dataset.to_csv("./data/prices/"+row['code']+".csv", index=False)
