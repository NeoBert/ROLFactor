import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np


class Pamr(object):
    def __init__(self, n_stock, version='ver1', param=dict(), dataset_name=''):
        """
        Variable:   n_stock: number of stock
                    version: version of relative price
                    name: name of method
                    weights: store historical weights
                    eps: Control parameter for variant 0
                    C: Control parameter for variant 1 and 2
                    variant: Variants 0, 1, 2 are available.
        """
        self.n_stock = n_stock
        self.version = version
        self.name = 'pamr'
        self.weights = []
        # method parameter
        self.variant = 0 if 'variant' not in param.keys() else param['variant']
        self.eps = 0.5 if 'epsilon' not in param.keys() else param['epsilon']
        self.c = 500 if 'c' not in param.keys() else param['c']

        assert self.variant in [0, 1, 2]

    def compute_weight(self, relative_price, stock_feature=None):
        """
        Function:   compute portfolio weight
        Input:      relative_price: float-list (n_time, n_stock)
                                    relative_price from [0, n_time - 1]
                    stock_feature: float-list (n_time, n_stock, n_feature)
                                    stock_feature from [0, n_time - 1]
        """
        if self.version == 'ver0':
            relative_price_array = np.array(relative_price) / 100 + 1
        else:
            relative_price_array = np.array(relative_price)

        for t in range(len(relative_price_array)):
            if t == 0:
                weight = [1 / self.n_stock] * self.n_stock
            else:
                x = relative_price_array[t - 1]
                x_mean = np.mean(x)
                b = np.array(self.weights[-1])
                le = max(0, np.dot(b, x) - self.eps)

                if self.variant == 0:
                    lam = le / np.linalg.norm(x - x_mean) ** 2
                elif self.variant == 1:
                    lam = min(self.c, le / np.linalg.norm(x - x_mean) ** 2)
                else:
                    lam = le / (np.linalg.norm(x - x_mean) ** 2 + 0.5 / self.c)

                lam = min(100000, lam)
                b = b - lam * (x - x_mean)
                weight = list(w for w in self.simplex_proj(b))
            self.weights.append(weight)

    def simplex_proj(self, b):
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
