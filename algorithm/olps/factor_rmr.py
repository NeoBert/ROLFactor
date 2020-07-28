import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np
from itertools import combinations

parent_dir_writer = abspath('result/statistic/weight')


class FaRmr(object):
    def __init__(self, n_factor, n_choose=1, proj="simplex"):
        """
        Variable:   n_factor: number of factor
                    n_choose: number of chosen factor
                    name: name of method
                    weights: store historical weights
                    window: window size
                    eps: Constraint on return for new weights on last price (average of prices)
        """
        self.n_factor = n_factor
        self.n_choose = n_choose
        self.name = 'RMR'
        self.weights = []
        self.proj = proj
        self.mask = list(combinations(range(self.n_factor), n_choose))
        self.n_comb = len(self.mask)
        # method parameter
        self.window = 5
        self.eps = 0.5
        self.tau = 0.001
        # method update parameter
        self.__b = [1 / self.n_comb] * self.n_comb

    def compute_weight(self, abs_ic):
        """
        Function:   compute portfolio weight
        Input:      abs_ic: float-array (n_time, n_factor)
                                    abs_ic from [0, n_time - 1]
        """
        for t in range(len(abs_ic)):
            weight = self.weigh_update(t, abs_ic[t-self.window:t])
            self.weights.append(weight)
        
    def weigh_update(self, t, per_ic):
        """
        Input:  per_ic: periodic absolute ic - float-array
        """
        l2_norm = lambda z: np.linalg.norm(z)
        l1_norm = lambda z: np.abs(z).sum()
        
        if t < self.window:
            b = np.zeros(self.n_factor)
            b[np.random.randint(self.n_factor)] = 1
        else:
            x = per_ic[-1]
            history = per_ic

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
            b = np.array(self.__b)
            lam = max(0, (self.eps - np.dot(b, x_pred)) / np.linalg.norm(x_pred - x_mean) ** 2)
            lam = min(100000, lam)
            b = b + lam * (x_pred - x_mean)
            b = self.simplex_proj(b, proj=self.proj).tolist()
            self.__b = b
        chosen_idx = np.argmax(b)
        weight = [0] * self.n_factor
        for j in range(self.n_choose):
            weight[self.mask[chosen_idx][j]] = 1
        return weight

    def simplex_proj(self, b, proj):
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
            power = np.exp(b)
            return power / np.sum(power)

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
