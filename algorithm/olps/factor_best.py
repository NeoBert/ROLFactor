import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np
from itertools import combinations

parent_dir_writer = abspath('result/statistic/weight')

class FaBest(object):
    # Best portfolio method
    def __init__(self, n_factor, n_choose=1):
        """
        Variable:   n_factor: number of factor
                    n_choose: number of chosen factor
                    name: name of method
                    weights: store historical weights
        """
        self.n_factor = n_factor
        self.n_choose = n_choose
        self.name = 'Best'
        self.weights = []
        self.mask = list(combinations(range(self.n_factor), n_choose))
        self.n_comb = len(self.mask)


    def compute_weight(self, abs_ic):
        """
        Function:   compute portfolio weight
        Input:      abs_ic: float-array (n_time, n_factor)
                                    abs_ic from [0, n_time - 1]
        """
        for t in range(len(abs_ic)):
            per_ic = abs_ic[t]
            weight = self.weight_update(t, per_ic)
            self.weights.append(weight)


    def weight_update(self, t, per_ic):
        """
        Input:  per_ic: periodic absolute ic - float-array (n_factor)
        """
        reward = np.zeros(self.n_comb)
        for i in range(self.n_comb):
            for j in range(self.n_choose):
                reward[i] += per_ic[self.mask[i][j]]
        chosen_idx = np.argmax(reward)
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