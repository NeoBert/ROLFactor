import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np
from itertools import combinations

parent_dir_writer = abspath('result/statistic/weight')

class FaUcb(object):
    def __init__(self, n_factor, n_choose=1):
        self.n_factor = n_factor
        self.n_choose = n_choose
        self.name = 'UCB'
        self.weights = []
        self.mask = list(combinations(range(self.n_factor), n_choose))
        self.n_comb = len(self.mask)
        # method update parameter
        self.__w = [1] * self.n_factor

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
        mean = np.array(self.__w)
        bound = np.sqrt(2 * np.log(t+1) / mean)
        chosen_idx = np.argmax(mean / (t + 1) + bound)

        # update
        # full feedback
        # for i in range(self.n_comb):
        #     reward = 0
        #     for j in range(self.n_choose):
        #         reward += per_ic[self.mask[i][j]]
        #     self.__w[i] += reward
        # bandit feedback
        # if chosen_idx == abs_ic[t].index(max(abs_ic[t])):
        comb_ic = np.zeros(self.n_comb)
        for i in range(self.n_comb):
            for j in range(self.n_choose):
                comb_ic[i] += per_ic[self.mask[i][j]]
        if chosen_idx == np.argmax(comb_ic):
            reward = 0
            for j in range(self.n_choose):
                reward += per_ic[self.mask[chosen_idx][j]]
            self.__w[chosen_idx] += reward

        weight = [0] * self.n_factor
        for j in range(self.n_choose):
            weight[self.mask[chosen_idx][j]] = 1
        return weight

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
