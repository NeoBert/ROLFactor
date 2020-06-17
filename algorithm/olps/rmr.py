import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np

parent_dir_writer = abspath(join(dirname(__file__), '../result/statistic/weight'))


class Rmr(object):
    def __init__(self, n_stock,proj="simplex"):
        """
        Variable:   n_stock: number of stock
                    version: version of relative price
                    name: name of method
                    weights: store historical weights
                    window: window size
                    eps: Constraint on return for new weights on last price (average of prices)
        """
        self.n_stock = n_stock
        self.name = 'rmr'
        self.weights = []

        self.window = 5
        self.eps = 10
        self.tau = 0.001
        self.proj = proj

    def compute_weight(self, relative_price, stock_feature=None):
        """
        Function:   compute portfolio weight
        Input:      relative_price: float-list (n_time, n_stock)
                                    relative_price from [0, n_time - 1]
                    stock_feature: float-list (n_time, n_stock, n_feature)
                                    stock_feature from [0, n_time - 1]
        """
        relative_price_array = np.array(relative_price)

        l2_norm = lambda z: np.linalg.norm(z)
        l1_norm = lambda z: np.abs(z).sum()

        for t in range(len(relative_price_array)):
            if t == 0:
                b = np.ones(self.n_stock) / self.n_stock
            elif t < self.window:
                pass
            else:
                x = relative_price_array[t - 1, :]
                history = relative_price_array[:t, :][-self.window:]

                y = np.median(history, axis=0)

                y_last = None
                while y_last is None or l1_norm(y - y_last) / l1_norm(y_last) > self.tau:
                    y_last = y
                    d = l2_norm(history - y_last)
                    nu = (history == y_last).any(axis=0).astype(int)
                    ru = l2_norm(((history - y_last) / d).sum())
                    tabove = (history / l2_norm(history - y_last)).sum() / (1 / d).sum()
                    y = np.maximum(1 - nu / ru, 0) * tabove + np.minimum(nu / ru, 1) * y_last

                x_pred = y / x
                x_mean = np.mean(x_pred)
                lam = max(0, (self.eps - np.dot(b, x_pred)) / np.linalg.norm(x_pred - x_mean) ** 2)
                lam = min(100000, lam)
                b = b + lam * (x_pred - x_mean)
                b = self.simplex_proj(b, proj=self.proj)
            weight = list(w for w in b)
            self.weights.append(weight)

    def simplex_proj(self, b, proj):
        """
        Function:       project weight into delta-m field (w_i > 0, sigma w_i = 1) in linear time cost
        Input:          b: weight array (n_stock)
        """
        if proj == "simplex":
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
        elif proj == "softmax":
            power = np.power(np.e, b)
            return power / np.sum(power)

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
