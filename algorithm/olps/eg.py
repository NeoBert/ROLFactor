# Exponential gradient Portfolio

import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np

parent_dir_writer = abspath(join(dirname(__file__), '../result/statistic/weight'))


class EG(object):
    # Exponential gradient portfolio method
    def __init__(self, n_stock):
        """
        Variable:   n_stock: number of stock
                    version: version of relative price
                    name: name of method
                    weights: store historical weights
                    eta: learning rate
        """
        self.n_stock = n_stock
        self.name = 'eg'
        self.weights = []
        self.eta = 0.05

    def compute_weight(self, relative_price):
        """
        Function:   compute portfolio weight
        Input:      relative_price: float-list (n_time, n_stock)
                                    relative_price from [0, n_time - 1]
                    stock_feature: float-list (n_time, n_stock, n_feature)
                                    stock_feature from [0, n_time - 1]
        """
        for t in range(len(relative_price)):
            # equal weight in the first round
            if t == 0:
                weight = [1 / self.n_stock] * self.n_stock
            # following rounds
            else:
                weight_array = np.array(self.weights[-1])
                price_array = np.array(relative_price[t - 1])
                value_weighted_return = price_array / np.dot(weight_array, price_array)
                exp_weighted_return = np.exp(self.eta * value_weighted_return)
                weight_array = np.multiply(weight_array, exp_weighted_return) / np.dot(weight_array,
                                                                                       exp_weighted_return)
                weight = list(w for w in weight_array)
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
