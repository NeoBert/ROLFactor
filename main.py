from stock import Stock
import datetime

s = Stock(start_date=datetime.date(2019, 1, 1), end_date=datetime.date(2020, 1, 1))
s.update_stock_info()