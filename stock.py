import tushare as ts
import os
import datetime
from tqdm import tqdm
import time


class Stock:
    def __init__(self, start_date, end_date):
        self.pro = ts.pro_api("35bdbecdaa5eca31d74117d4f44295c93076592f082278c32a2e3665")
        assert end_date > start_date
        self.start_date = start_date
        self.end_date = end_date
        self.dir_path = '/Applications/python/quant/dataset/daily'

    def update_stock_info(self):
        # 获得数据集中已经存储的数据
        stock_dates = os.listdir(self.dir_path)
        for day in tqdm(range((self.end_date - self.start_date).days + 1)):
            # 判断当前日期是否已经存在数据
            current_day = self.start_date + datetime.timedelta(days=day)
            if current_day.strftime("%Y-%m-%d") in stock_dates:
                continue

            # 获取当前日期所有stock的数据
            while True:
                try:
                    stock_day = self.pro.daily(trade_date=current_day.strftime("%Y%m%d"))
                    stock_day.to_csv(os.path.join(self.dir_path, current_day.strftime('%Y-%m-%d')), index=None)
                    break
                except:
                    time.sleep(1)

    def generate_relative_price(self):

