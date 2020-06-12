from stock import Stock
import datetime

s = Stock(start_date=datetime.date(2019, 1, 1), end_date=datetime.date(2019, 1, 5))
s.generate_data_frame()

