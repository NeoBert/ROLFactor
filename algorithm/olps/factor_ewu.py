import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np
import random
from itertools import combinations

parent_dir_writer = abspath('result/statistic/weight')

class FaEwu(object):
    def __init__(self, n_factor, n_choose=1):
        self.n_factor = n_factor
        self.n_choose = n_choose
        self.name = 'EWU'  # Exponential Weight Update
        self.weights = []
        self.variant = 0
        if self.variant in [0, 1]:
            self.mask = list(combinations(range(self.n_factor), n_choose))
            self.n_comb = len(self.mask)
        elif self.variant == 2:
            self.n_comb = self.n_factor
        # method parameter
        self.eta = 1
        self.gamma = 0.01  # EE rate
        # method update parameter
        self.__w = [1] * self.n_comb

    def compute_weight(self, abs_ic):
        """
        Input:  abs_ic: float-array (n_time, n_factor)
        """
        for t in range(len(abs_ic)):
            per_ic = abs_ic[t]
            weight = self.weight_update(t, per_ic)
            self.weights.append(weight)

    def weight_update(self, t, per_ic):
        """
        Input:  per_ic: periodic absolute ic - float-array (n_factor)
        """
        # choose
        if self.variant == 0:
            chosen_idx = self.__draw(self.__w)
        elif self.variant == 1:
            p = [(1 - self.gamma) * w / sum(self.__w) + self.gamma / self.n_comb for w in self.__w]
            chosen_idx = self.__draw(p)
        elif self.variant == 2:
            chosen_idx = self.__top(self.__w)

        # update
        for i in range(self.n_comb):
            if self.variant in [0, 1]:
                reward = 0
                for j in range(self.n_choose):
                    reward += per_ic[self.mask[i][j]]
                self.__w[i] *= np.exp(self.eta * reward)
            elif self.variant == 2:
                self.__w[i] *= np.exp(self.eta * per_ic[i])

        weight = [0] * self.n_factor
        for j in range(self.n_choose):
            if self.variant in [0, 1]:
                weight[self.mask[chosen_idx][j]] = 1
            elif self.variant == 2:
                weight[chosen_idx[j]] = 1
        return weight

    def __draw(self, weights):
        """
        Function:   draw form uniform distribution
        Input:      weights: float-list (n_stock)
        Output:     index: chosen index
        """
        choice = random.uniform(0, sum(weights))
        index = 0
        for weight in weights:
            choice -= weight
            if choice <= 0:
                return index
            index += 1
        return len(weights) - 1

    def __top(self, weights):
        """
        Function:   select top n_choose weight
        Input:      weights: float-list (n_stock)
        Output:     index: int-list (n_choose)
        """
        choice = np.argsort(weights)[::-1]
        index = []
        for i in range(self.n_choose):
            index.append(choice[i])
        return index

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
