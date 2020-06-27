import os
import datetime
import time
import pandas as pd
from os.path import join
import numpy as np
from glob import glob


def generate_relative_price(all_stock_csv):
    all_relative_price_csv = all_stock_csv / all_stock_csv.shift(1)
    all_relative_price_csv = all_relative_price_csv.drop(all_relative_price_csv.index[0])
    all_relative_price_csv = all_relative_price_csv.dropna(axis=1)
    return all_relative_price_csv


class Factor:
    def __init__(self):
        csv_paths = glob('/Applications/python/quant/dataset/china_ic/*')
        all_csv = []
        for csv_path in csv_paths:
            csv = pd.read_csv(csv_path, index_col='date')['1D']
            csv = pd.DataFrame(csv)
            csv.rename(columns={'1D': csv_path.split('/')[-1].strip('.csv')}, inplace=True)
            all_csv.append(csv)
        self.csv = pd.concat(all_csv, axis=1)

    def generate_data_frame(self):
        csv = self.csv
        csv = abs(csv) + 1
        csv = generate_relative_price(csv)
        csv = csv.dropna(axis=0)
        return csv.values.tolist()


if __name__ == '__main__':
    factor = Factor()
    factor.generate_data_frame()
