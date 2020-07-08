import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np
import random
from itertools import combinations

parent_dir_writer = abspath('result/statistic/weight')

class FaMwu(object):
    def __init__(self, n_factor, n_choose=1):
        self.n_factor = n_factor
        self.n_choose = n_choose
        self.name = 'MWU'  # Multiplicative Weight Update
        self.weights = []
        # method parameter
        self.eta = 0.1
        self.mask = list(combinations(range(self.n_factor), n_choose))
        self.n_comb = len(self.mask)
        # method update parameter
        self.__w = [1] * self.n_comb

    def compute_weight(self, abs_ic):
        for t in range(len(abs_ic)):
            # choose
            chosen_idx = self.__draw(self.__w)

            # update
            for i in range(self.n_comb):
                reward = 0
                for j in range(self.n_choose):
                    reward += abs_ic[t][self.mask[i][j]]
                self.__w[i] *= (1 + self.eta * reward)
            
            weight = [0] * self.n_factor
            for j in range(self.n_choose):
                weight[self.mask[chosen_idx][j]] = 1
            self.weights.append(weight)

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
