import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np

parent_dir_writer = abspath('../result/statistic/weight')


class Anticor(object):
    def __init__(self, n_stock):
        """
        Variable:   n_stock: number of stock
                    version: version of relative price
                    name: name of method
                    weights: store historical weights
                    min_window: minimum window size
                    max_window: maximum window size
                    epsilon: a small value to keep divisor not being zero
                    window_weight_array: save weight of last period
        """
        self.n_stock = n_stock
        self.name = 'ubah_anticor'
        self.weights = []

        self.min_window = 2
        self.max_window = 30
        self.epsilon = 1e-8

        self.__window_weight = [[1 / self.n_stock] * self.n_stock] * (self.max_window - self.min_window + 1)
        self.__window_cw = [1.] * (self.max_window - self.min_window + 1)

    def compute_weight(self, relative_price, stock_feature=None):
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
                relative_price_array = np.array(relative_price)
                self.__window_cw = (np.array(self.__window_cw)
                                    * np.dot(np.array(self.__window_weight), relative_price[t - 1])).tolist()
                for window in range(self.min_window, self.max_window + 1):
                    self.__aggregate_anticor_weight(t, window, relative_price_array)

                current_weight = np.dot(self.__window_cw, self.__window_weight)
                weight = list(w / current_weight.sum() for w in current_weight)
            self.weights.append(weight)

    def __aggregate_anticor_weight(self, t, window, relative_price):
        """
        Function:   aggregate weights of anticor
        Input:      t: number of period
                    window: window size
                    relative_price: float-array [0, n_time - 1]
        """
        win_idx = window - self.min_window
        weight_array = np.array(self.__window_weight[win_idx])
        price_array = relative_price[t - 1]
        weight_array = np.multiply(weight_array, price_array) / np.dot(weight_array, price_array)
        if t < 2 * window:
            self.__window_weight[win_idx] = weight_array.tolist()
        else:
            lx1 = np.log(relative_price[t - 2 * window:t - window])  # (window, n_stock)
            lx2 = np.log(relative_price[t - window:t])  # (window, n_stock)
            self.__window_weight[win_idx] = self.__compute_anticor_weight(window, lx1, lx2, weight_array)

    def __compute_anticor_weight(self, window, lx1, lx2, weight):
        """
        Function:   compute weight of anticor with fixed window size
        Input:      lx1: logarithmic relative price of second last window
                    lx2: logarithmic relative price of last window
                    weight: float-array (n_stock)
        Output:     weight: float-array (n_stock)
        """
        mu1, std1 = np.mean(lx1, axis=0), np.std(lx1, axis=0)  # (n_stock)
        mu2, std2 = np.mean(lx2, axis=0), np.std(lx2, axis=0)  # (n_stock)

        cor = 1 / (window - 1) * np.dot((lx1 - mu1).T, (lx2 - mu2))  # (n_stock, n_stock), actually covariance
        for i in range(self.n_stock):
            for j in range(self.n_stock):
                if std1[i] != 0 and std2[j] != 0:
                    cor[i][j] /= std1[i] * std2[j]
                else:
                    cor[i][j] = 0

        claim = np.zeros((self.n_stock, self.n_stock))
        for i in range(self.n_stock):
            for j in range(self.n_stock):
                claim[i][j] += self.epsilon
                if cor[i][j] > 0 and mu2[i] >= mu2[j]:
                    claim[i][j] += cor[i][j]
                    claim[i][j] -= cor[i][i] if cor[i][i] < 0 else 0
                    claim[i][j] -= cor[j][j] if cor[j][j] < 0 else 0

        transfer = (np.array(self.weights[-1]) * claim.T / claim.sum(axis=1)).T
        for i in range(self.n_stock):
            for j in range(self.n_stock):
                weight[i] -= transfer[i][j]
                weight[i] += transfer[j][i]
            if weight[i] < 0:
                weight[i] = 0
        weight /= weight.sum()
        return weight

    def write_weight(self, file_name):
        """
        Function:   write weights
        Input       file_name: name of file
        """
        file_name = 'weight-' + file_name + '.csv'
        if not exists(parent_dir_writer):
            os.makedirs(parent_dir_writer)
        path = abspath(join(parent_dir_writer, file_name))
        pd_data = pd.DataFrame(self.weights)
        pd_data.to_csv(path, index=False, sep=',')
