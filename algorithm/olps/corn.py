# Exponential gradient Portfolio

import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np
import scipy.optimize as optimize

parent_dir_writer = abspath(join(dirname(__file__), '../result/statistic/weight'))


class CORN(object):
    #  Correlation-driven nonparametric learning approach.
    def __init__(self, n_stock):
        """
        Variable:   n_stock: number of stock
                    version: version of relative price
                    name: name of method
                    weights: store historical weights
                    eta: learning rate
        """
        self.n_stock = n_stock
        self.weights = []
        self.window = 5
        self.rho = 0.1

    def compute_weight(self, relative_price):
        relative_price_array = np.array(relative_price)
        for t in range(len(relative_price_array)):
            if t == 0:
                weight = [1 / self.n_stock] * self.n_stock
            elif t < self.window:
                pass
            else:
                indices = []
                x = relative_price_array[t - 1, :]
                history = relative_price_array[:t, :]
                m = len(x)
                x_t = history[-self.window:].flatten()

                for i in range(self.window, len(history)):
                    x_i = history[i - self.window:i, :].flatten()
                    if np.corrcoef(x_t, x_i)[0, 1] >= self.rho:
                        indices.append(i)

                c = history[indices, :]
                if c.shape[0] == 0:
                    weight = np.ones(m) / float(m)
                else:
                    weight = self.opt_weights(c).tolist()
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

    def opt_weights(self, x, max_leverage=1):
        x_0 = max_leverage * np.ones(x.shape[1]) / float(x.shape[1])
        objective = lambda b: -np.sum(np.log(np.maximum(np.dot(x - 1, b) + 1, 0.0001)))
        cons = ({'type': 'ineq', 'fun': lambda b: max_leverage - sum(b)},)

        while True:
            res = optimize.minimize(objective, x_0, bounds=[(0., max_leverage)] * len(x_0), constraints=cons,
                                    method='slsqp')
            EPS = 1E-7
            if (res.x < 0. - EPS).any() or (res.x > max_leverage + EPS).any():
                x = x + np.random.randn(1)[0] * 1E-5
                continue
            elif res.success:
                break
            else:
                if np.isnan(res.x).any():
                    res.x = np.zeros(x.shape[1])
                else:
                    break
        return res.x
