import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np

parent_dir_writer = abspath('result/statistic/weight')


def simplex_proj(b, proj):
    """
    Function:       project weight into delta-m field (w_i > 0, sigma w_i = 1) in linear time cost
    Input:          b: weight array (n_factor)
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
        w = np.exp(b)
        return w / np.sum(w)


class FaOlmar(object):
    # On-Line Portfolio Selection with Moving Average Reversion
    def __init__(self, n_factor, proj="softmax"):
        """
        Variable:   n_factor: number of stock
                    version: version of relative price
                    name: name of method
                    weights: store historical weights
                    window: window size
                    eps: Constraint on return for new weights on last price (average of prices)
        """
        self.n_factor = n_factor
        self.name = 'OLMAR'
        self.weights = []
        self.window = 5
        self.eps = 10
        self.variant = 1
        self.alpha = 0.25
        self.proj = proj
        self.__b = [1 / self.n_factor] * self.n_factor

    def compute_weight(self, abs_ic):
        """
        Function:   compute portfolio weight
        Input:      abs_ic: float-list (n_time, n_factor)
                                    abs_ic from [0, n_time - 1]
                    stock_feature: float-list (n_time, n_factor, n_feature)
                                    stock_feature from [0, n_time - 1]
        """
        abs_ic_array = np.array(abs_ic)
        for t in range(len(abs_ic_array)):
            if self.variant == 0:
                if t == 0:
                    b = np.zeros(self.n_factor)
                    b[np.random.randint(self.n_factor)] = 1
                elif t < self.window:
                    b = np.zeros(self.n_factor)
                    b[np.random.randint(self.n_factor)] = 1
                    pass
                else:
                    x = abs_ic_array[t - 1, :]
                    history = abs_ic_array[t - self.window:t]
                    x_pred = np.ones(self.n_factor)
                    x_multi = np.ones(self.n_factor)
                    for i in range(self.window-1, -1, -1):
                        x_multi *= history[i]
                        x_pred += 1 / x_multi
                    x_pred /= self.window
                    x_mean = np.mean(x_pred)
                    b = np.array(self.__b)
                    lam = max(0, (self.eps - np.dot(b, x_pred)) / np.linalg.norm(x_pred - x_mean) ** 2)
                    lam = min(100000, lam)
                    b = b + lam * (x_pred - x_mean)
                    b = simplex_proj(b, self.proj)
                    self.__b = [w for w in b]
            elif self.variant == 1:
                if t == 0:
                    b = np.zeros(self.n_factor)
                    b[np.random.randint(self.n_factor)] = 1
                    x_pred = np.zeros(self.n_factor)
                else:
                    x = abs_ic_array[t - 1]
                    # x_pred = self.alpha + (1 - self.alpha) * x_pred / x
                    x_pred = self.alpha * x_pred + (1 - self.alpha) * x
                    x_mean = np.mean(x_pred)
                    b = np.array(self.__b)
                    lam = max(0, (np.dot(b, x_pred) - self.eps) / np.linalg.norm(x_pred - x_mean) ** 2)
                    lam = min(100000, lam)
                    b = b + lam * (x_pred - x_mean)
                    b = simplex_proj(b, self.proj)
                    self.__b = [w for w in b]
            max_idx = np.argmax(b)
            weight = [0] * self.n_factor
            weight[max_idx] = 1
            self.weights.append(weight)

    def write_weight(self, file_name):
        """
        Function:   write weights
        Input       file_name: name of file
        """
        file_name = 'fa-weight-' + file_name + '.csv'
        if exists(parent_dir_writer) == False:
            os.mkdir(parent_dir_writer)
        path = abspath(join(parent_dir_writer, file_name))
        pd_data = pd.DataFrame(self.weights)
        pd_data.to_csv(path, index=False, sep=',')
