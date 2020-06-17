import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np

parent_dir_writer = abspath(join(dirname(__file__), '../result/statistic/weight'))


def simplex_proj(b):
    """
    Function:       project weight into delta-m field (w_i > 0, sigma w_i = 1) in linear time cost
    Input:          b: weight array (n_stock)
    """
    m = len(b)
    bget = False

    s = sorted(b, reverse=True)
    tmpsum = 0.

    for ii in range(m - 1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1) / (ii + 1)
        if tmax >= s[ii + 1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m - 1] - 1) / m

    return np.maximum(b - tmax, 0.)


class Olmar(object):
    # On-Line Portfolio Selection with Moving Average Reversion
    def __init__(self, n_stock):
        """
        Variable:   n_stock: number of stock
                    version: version of relative price
                    name: name of method
                    weights: store historical weights
                    window: window size
                    eps: Constraint on return for new weights on last price (average of prices)
        """
        self.n_stock = n_stock
        self.name = 'olmar'
        self.weights = []
        self.window = 5
        self.eps = 10
        self.variant = 0
        self.alpha = 0.25

    def compute_weight(self, relative_price):
        """
        Function:   compute portfolio weight
        Input:      relative_price: float-list (n_time, n_stock)
                                    relative_price from [0, n_time - 1]
                    stock_feature: float-list (n_time, n_stock, n_feature)
                                    stock_feature from [0, n_time - 1]
        """
        relative_price_array = np.array(relative_price)
        for t in range(len(relative_price_array)):
            if self.variant == 0:
                if t == 0:
                    b = np.ones(self.n_stock) / self.n_stock
                elif t < self.window:
                    pass
                else:
                    x = relative_price_array[t - 1, :]
                    history = relative_price_array[:t, :]
                    x_pred = (history[-self.window:] / x).mean(axis=0)
                    x_mean = np.mean(x_pred)
                    lam = max(0, (self.eps - np.dot(b, x_pred)) / np.linalg.norm(x_pred - x_mean) ** 2)
                    lam = min(100000, lam)
                    b = b + lam * (x_pred - x_mean)
                    b = simplex_proj(b)
            else:
                if t == 0:
                    b = np.ones(self.n_stock) / self.n_stock
                    x_pred = relative_price_array[0]
                else:
                    x = relative_price_array[t - 1, :]
                    x_pred = self.alpha + (1 - self.alpha) * x_pred / x
                    x_mean = np.mean(x_pred)
                    lam = max(0, (self.eps - np.dot(b, x_pred)) / np.linalg.norm(x_pred - x_mean) ** 2)
                    lam = min(100000, lam)
                    b = b + lam * (x_pred - x_mean)
                    b = simplex_proj(b)
            weight = list(w for w in b)
            self.weights.append(weight)

    def write_weight(self, file_name):
        """
        Function:   write weights
        Input       file_name: name of file
        """
        file_name = 'weight-' + file_name + '.csv'
        if exists(parent_dir_writer) == False:
            os.mkdir(parent_dir_writer)
        path = abspath(join(parent_dir_writer, file_name))
        pd_data = pd.DataFrame(self.weights)
        pd_data.to_csv(path, index=False, sep=',')
