import tushare as ts
from tqdm import tqdm
import time
from os.path import join

ts.set_token("35bdbecdaa5eca31d74117d4f44295c93076592f082278c32a2e3665")
pro = ts.pro_api()

# 获取所有股票名称
stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code')['ts_code'].values

'''
获取所有股票数据
列名可参考 https://tushare.pro/document/2?doc_id=27
'''

for ts_code in tqdm(list(stocks)[1861:]):
    while True:
        try:
            df = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date='20000101', end_date='20200517')
            break
        except:
            time.sleep(1)
    df.to_csv(join('../data/stock', ts_code))
