import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np
import cvxpy as cp

parent_dir_writer = abspath('result/statistic/weight')

class FaBcrp(object):
    # Best constant rebalanced portfolio method
    def __init__(self, n_factor):
        """
        Variable:   n_factor: number of stock
                    version: version of relative price
                    name: name of method
                    weights: store historical weight
        """
        self.n_factor = n_factor
        self.name = 'BCRP'
        self.weights = []


    def compute_weight(self, abs_ic):
        """
        Function:   compute portfolio weight
        Input:      abs_ic: float-list (n_time, n_factor)
                                    abs_ic from [0, n_time - 1]
        """
        idx = self.__linear_opt(abs_ic)
        weight = [0] * self.n_factor
        weight[idx] = 1
        for t in range(len(abs_ic)):
            self.weights.append(weight)


    def __linear_opt(self, abs_ic):
        """
        Function:   argmax(p): (sum(log(projection_space * p)))
        Input:      abs_ic: float-list (n_time, n_factor)
        Output:     idx
        """
        reward = list(0 for i in range(self.n_factor))
        n_time = len(abs_ic)
        for t in range(n_time):
            max_idx = abs_ic[t].index(max(abs_ic[t]))
            reward[max_idx] += abs_ic[t][max_idx]
        return reward.index(max(reward))


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