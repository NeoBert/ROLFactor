import tushare as ts
import os
import datetime
import time
import pandas as pd
from os.path import join
from tqdm import tqdm


def _generate_relative_price(all_fund_csv):
    all_relative_price_csv = all_fund_csv / all_fund_csv.shift(1)
    all_relative_price_csv = all_relative_price_csv.drop(all_relative_price_csv.index[0])
    all_relative_price_csv = all_relative_price_csv.dropna(axis=1)
    return all_relative_price_csv


class Fund(object):
    def __init__(self, start_date, end_date):
        self.pro = ts.pro_api("35bdbecdaa5eca31d74117d4f44295c93076592f082278c32a2e3665")
        assert end_date > start_date
        self.start_date = start_date
        self.end_date = end_date
        self.dir_path = 'dataset/china_fund_daily'
        self.fund_dates = os.listdir(self.dir_path)

    def _update_fund_info(self):
        print(">>START UPDATE FUND INFO IN DATABASE<<")
        for day in range((self.end_date - self.start_date).days + 1):
            # 判断当前日期是否已经存在数据
            current_day = self.start_date + datetime.timedelta(days=day)
            if current_day.strftime("%Y-%m-%d") in self.fund_dates:
                continue
            # 获取当前日期所有基金的数据
            while True:
                try:
                    fund_day = self.pro.fund_daily(trade_date=current_day.strftime("%Y%m%d"))
                    fund_day.to_csv(os.path.join(self.dir_path, current_day.strftime('%Y-%m-%d')), index=None)
                    break
                except:
                    time.sleep(1)

    def _generate_all_fund_csv(self):
        print(">>START GENERATE ALL fund INFO IN DATABASE<<")
        all_fund_csv = pd.DataFrame()
        for day in range((self.end_date - self.start_date).days + 1):
            current_day = self.start_date + datetime.timedelta(days=day)
            fund_csv = pd.read_csv(join(self.dir_path, current_day.strftime('%Y-%m-%d')),
                                   usecols=['ts_code', 'close'], index_col=['ts_code'])
            if len(fund_csv) == 0:
                continue
            fund_csv.rename(columns={'close': current_day.strftime('%Y-%m-%d')}, inplace=True)
            fund_csv = pd.DataFrame(fund_csv.values.T, index=fund_csv.columns, columns=fund_csv.index)
            all_fund_csv = pd.concat((all_fund_csv, fund_csv), axis=0)
        return all_fund_csv

    def generate_data_frame(self, mode="all", finance=True, frequency="daily"):
        self._update_fund_info()
        csv = self._generate_all_fund_csv()
        csv = _generate_relative_price(csv)
        csv = csv.dropna(axis=0)
        if mode == "random":
            csv = csv.sample(frac=0.1, axis=1)
        if frequency == "weekly":
            csv = csv.iloc[0:len(csv):5]
        ntime, nfund = csv.shape
        print("Duration:" + str(ntime))
        print("Number of N fund:" + str(nfund))
        return csv.values.tolist()
