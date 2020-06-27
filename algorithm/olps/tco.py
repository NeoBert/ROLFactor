import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np

parent_dir_writer = abspath('result/statistic/weight')


class TransCostOpt(object):
    # Transaction costs optimization for PAMR / OLMAR.
    def __init__(self, n_stock, version='ver1', param=dict(), dataset_name=''):
        """
        Variable:   n_stock: number of stock
                    version: version of relative price
                    name: name of method
                    weights: store historical weights
                    lam: trade-off parameter
                    eta: smoothing parameter
                    window: window size for variant 1
                    variant: Variants 0, 1 are available.
        """
        self.n_stock = n_stock
        self.version = version
        self.name = 'tco'
        self.weights = []
        # method parameter
        self.variant = 0 if 'variant' not in param.keys() else param['variant']
        self.lam = 0.05 if 'lam' not in param.keys() else param['lam']
        self.eta = 10 if 'eta' not in param.keys() else param['eta']
        self.window = 25 if 'window' not in param.keys() else param['window']

        assert self.variant in [0, 1]


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
                if self.variant == 0:
                    x = relative_price_array[t - 1]
                    f = 1/x
                    b_adjust = (relative_price_array[t - 1] * self.weights[-1]) / (relative_price_array[t - 1] @ self.weights[-1])
                    v = f / (b_adjust @ f)
                    v_mean = np.mean(v)
                    b_half = self.eta * (v - v_mean)
                    b_diff = np.maximum(0, b_half - self.lam)
                    b = b_adjust + np.sign(b_half) * b_diff
                    weight = list(w for w in self.__simplex_proj(b))
                else:
                    if t < self.window:
                        weight = [1 / self.n_stock] * self.n_stock
                    else:
                        history = relative_price_array[t - self.window:t]
                        f = np.ones(self.n_stock)
                        x_multi = np.ones(self.n_stock)
                        for i in range(self.window-1, -1, -1):
                            x_multi *= history[i]
                            f += 1 / x_multi
                        f /= self.window
                        b_adjust = (relative_price_array[t - 1] * self.weights[-1]) / (relative_price_array[t - 1] @ self.weights[-1])
                        v = f / (b_adjust @ f)
                        v_mean = np.mean(v)
                        b_half = self.eta * (v - v_mean)
                        b_diff = np.maximum(0, b_half - self.lam)
                        b = b_adjust + np.sign(b_half) * b_diff
                        weight = list(w for w in self.__simplex_proj(b))
            self.weights.append(weight)


    def __simplex_proj(self, b):
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