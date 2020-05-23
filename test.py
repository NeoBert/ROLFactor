import numpy as np
import pandas as pd
from pandas_datareader import DataReader
from datetime import datetime
import six
from algorithm import tools
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (16, 10)
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['lines.linewidth'] = 1

etfs = ['VTI', 'EFA', 'EEM', 'TLT', 'TIP', 'VNQ']
swensen_allocation = [0.3, 0.15, 0.1, 0.15, 0.15, 0.15]
benchmark = ['SPY']
train_start = datetime(2005, 1, 1)
train_end = datetime(2012, 12, 31)
test_start = datetime(2013, 1, 1)
test_end = datetime(2014, 12, 31)

train = DataReader(etfs, 'yahoo', start=train_start, end=train_end)['Adj Close']
test = DataReader(etfs, 'yahoo', start=test_start, end=test_end)['Adj Close']
train_b = DataReader(benchmark, 'yahoo', start=train_start, end=train_end)['Adj Close']
test_b = DataReader(benchmark, 'yahoo', start=test_start, end=test_end)['Adj Close']

from algorithm.corn import CORN
from algorithm.pamr import PAMR
corn = PAMR
result = corn.run(train)
print(result)