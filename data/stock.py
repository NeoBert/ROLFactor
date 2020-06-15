import tushare as ts
import os
import datetime
import time
import pandas as pd
from os.path import join
from tqdm import tqdm


def _generate_relative_price(all_stock_csv):
    all_relative_price_csv = all_stock_csv / all_stock_csv.shift(1) - 1
    all_relative_price_csv = all_relative_price_csv.drop(all_relative_price_csv.index[0])
    all_relative_price_csv = all_relative_price_csv.dropna(axis=1)
    return all_relative_price_csv


class Stock:
    def __init__(self, start_date, end_date):
        self.pro = ts.pro_api("35bdbecdaa5eca31d74117d4f44295c93076592f082278c32a2e3665")
        assert end_date > start_date
        self.start_date = start_date
        self.end_date = end_date
        self.dir_path = 'dataset/china_stock_daily'
        self.stock_dates = os.listdir(self.dir_path)

    def _update_stock_info(self):
        print(">>START UPDATE STOCK INFO IN DATABASE<<")
        for day in range((self.end_date - self.start_date).days + 1):
            # 判断当前日期是否已经存在数据
            current_day = self.start_date + datetime.timedelta(days=day)
            if current_day.strftime("%Y-%m-%d") in self.stock_dates:
                continue
            # 获取当前日期所有stock的数据
            while True:
                try:
                    stock_day = self.pro.daily(trade_date=current_day.strftime("%Y%m%d"))
                    stock_day.to_csv(os.path.join(self.dir_path, current_day.strftime('%Y-%m-%d')), index=None)
                    break
                except:
                    time.sleep(1)

    def _generate_financial_info(self, csv):
        print(">>START UPDATE STOCK FINANCIAL IN DATABASE<<")
        end_date = self.start_date
        start_date = end_date - datetime.timedelta(180)
        stock_finance = os.listdir(self.dir_path)

        all_stock_csv = pd.DataFrame()
        # 由于不同公司公告日不相同，可能可比性会适当降低
        for ts_code in tqdm(csv.columns.tolist()):
            while True:
                try:
                    stock_csv = self.pro.income(ts_code=ts_code, start_date=start_date.strftime("%Y%m%d"),
                                                end_date=end_date.strftime("%Y%m%d"),
                                                fields='ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,basic_eps,diluted_eps')
                    stock_csv = stock_csv.iloc[-1]
                    all_stock_csv = pd.concat((all_stock_csv, stock_csv), axis=0)
                    break
                except:
                    time.sleep(1)

    def _generate_all_stock_csv(self):
        print(">>START GENERATE ALL STOCK INFO IN DATABASE<<")
        all_stock_csv = pd.DataFrame()
        for day in range((self.end_date - self.start_date).days + 1):
            current_day = self.start_date + datetime.timedelta(days=day)
            stock_csv = pd.read_csv(join(self.dir_path, current_day.strftime('%Y-%m-%d')),
                                    usecols=['ts_code', 'close'], index_col=['ts_code'])
            if len(stock_csv) == 0:
                continue
            stock_csv.rename(columns={'close': current_day.strftime('%Y-%m-%d')}, inplace=True)
            stock_csv = pd.DataFrame(stock_csv.values.T, index=stock_csv.columns, columns=stock_csv.index)
            all_stock_csv = pd.concat((all_stock_csv, stock_csv), axis=0)
        return all_stock_csv

    def generate_data_frame(self, mode="random", finance=True):
        self._update_stock_info()
        csv = self._generate_all_stock_csv()
        csv = _generate_relative_price(csv)
        csv = csv.dropna(axis=0)
        if mode == "random":
            csv = csv.sample(frac=0.1, axis=1)
        ntime, nstock = csv.shape
        print("Duration:" + str(ntime))
        print("Number of N stock:" + str(nstock))
        return csv.values.tolist()
