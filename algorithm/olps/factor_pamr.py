import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np

parent_dir_writer = abspath('result/statistic/weight')

def simplex_proj(b):
    """
    Function:       project weight into delta-m field (w_i > 0, sigma w_i = 1) in linear time cost
    Input:          b: weight array (n_factor)
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

class FaPamr(object):
    def __init__(self, n_factor):
        """
        Variable:   n_factor: number of stock
                    name: name of method
                    weights: store historical weights
                    eps: Control parameter for variant 0
                    C: Control parameter for variant 1 and 2
                    variant: Variants 0, 1, 2 are available.
        """
        self.n_factor = n_factor
        self.name = 'PAMR'
        self.weights = []
        # method parameter
        self.variant = 0
        self.eps = 0.05
        self.c = 500
        # method update parameter
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
            if t == 0:
                weight = list(0 for i in range(self.n_factor))
                weight[np.random.randint(self.n_factor)] = 1
                # b = np.ones(self.n_factor) / self.n_factor
            else:
                x = abs_ic_array[t - 1]
                x_mean = np.mean(x)
                # b = np.array(self.weights[-1])
                b = np.array(self.__b)
                le = max(0, np.dot(b, x) - self.eps)

                if self.variant == 0:
                    lam = le / np.linalg.norm(x - x_mean) ** 2
                elif self.variant == 1:
                    lam = min(self.c, le / np.linalg.norm(x - x_mean) ** 2)
                else:
                    lam = le / (np.linalg.norm(x - x_mean) ** 2 + 0.5 / self.c)

                lam = min(100000, lam)
                b = b - lam * (x - x_mean)
                b = np.exp(b)
                b /= sum(b)
                self.__b = [w for w in b]
                # self.__b = [w for w in simplex_proj(b)]
                max_idx = np.argmax(b)
                weight = [0] * self.n_factor
                weight[max_idx] = 1
            self.weights.append(weight)

    def write_weight(self, file_name):
        file_name = 'fa-weight-' + file_name + '.csv'
        if not exists(parent_dir_writer):
            os.mkdir(parent_dir_writer)
        path = abspath(join(parent_dir_writer, file_name))
        pd_data = pd.DataFrame(self.weights)
        pd_data.to_csv(path, index=False, sep=',')
