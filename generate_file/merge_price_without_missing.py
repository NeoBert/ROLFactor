import pandas as pd
import os
from os.path import join
from tqdm import tqdm

# 生成日线数据 行为股票名称 列为时间
dir_path = '../data/stock'
stock_csv = pd.DataFrame()
count = 0
for csv_name in tqdm(os.listdir(dir_path)):
    csv_path = join(dir_path, csv_name)
    csv = pd.read_csv(csv_path, usecols=['ts_code', 'close'])
    csv = csv.dropna(axis=0, how='any')
    if len(csv) >= 3700:
        stock_csv[csv_name] = csv['close'][:3700]

# 生成前复权文件
stock_csv = stock_csv.iloc[::-1]
stock_csv.to_csv(join('../data', 'stock_qfq_price_number_equal_3700.csv'), index=None)
