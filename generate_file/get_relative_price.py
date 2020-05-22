import pandas as pd
import os
from os.path import join
from tqdm import tqdm

# 生成日线数据 行为股票名称 列为时间
csv = pd.read_csv("/Applications/python/quant/data/stock_qfq_price_number_equal_3700.csv")
for index, row in csv.iterrows():
    print(row)
exit()

# 生成前复权文件
# stock_csv.to_csv(join('../data', 'stock_qfq_price_number_equal_3700.csv'), index=None)
