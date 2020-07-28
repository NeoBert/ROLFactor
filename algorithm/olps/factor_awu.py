import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np
import random
from itertools import combinations

parent_dir_writer = abspath('result/statistic/weight')

class FaAwu(object):
    def __init__(self, n_factor, n_choose=1):
        self.n_factor = n_factor
        self.n_choose = n_choose
        self.name = 'AWU'  # Additive Weight Update
        self.weights = []
        # method parameter
        self.eta = 1
        self.mask = list(combinations(range(self.n_factor), n_choose))
        self.n_comb = len(self.mask)
        self.gamma = 0.01  # EE rate
        self.variant = 0
        # method update parameter
        self.__w = [1] * self.n_comb

    def compute_weight(self, abs_ic):
        """
        Input:  abs_ic: float-array (n_time, n_factor)
        """
        for t in range(len(abs_ic)):
            per_ic = abs_ic[t]
            weight = self.weight_update(per_ic)
            self.weights.append(weight)

    def weight_update(self, per_ic):
        """
        Input:  per_ic: periodic absolute ic - float-array (n_factor)
        """
        # choose
        if self.variant == 0:
            chosen_idx = self.__draw(self.__w)
        elif self.variant == 1:
            p = [(1 - self.gamma) * w / sum(self.__w) + self.gamma / self.n_comb for w in self.__w]
            chosen_idx = self.__draw(p)

        # update
        for i in range(self.n_comb):
            reward = 0
            for j in range(self.n_choose):
                reward += per_ic[self.mask[i][j]]
            self.__w[i] += self.eta * reward
            
        weight = [0] * self.n_factor
        for j in range(self.n_choose):
            weight[self.mask[chosen_idx][j]] = 1
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
