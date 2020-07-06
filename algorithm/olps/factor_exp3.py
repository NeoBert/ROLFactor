import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np
import random

parent_dir_writer = abspath('result/statistic/weight')

class FaExp3(object):
    def __init__(self, n_factor):
        self.n_factor = n_factor
        self.name = 'Exp3'
        self.weights = []
        # method parameter
        self.gamma = 0.1
        # method update parameter
        self.__w = [1] * self.n_factor
        self.__p =[0] * self.n_factor

    def compute_weight(self, abs_ic):
        for t in range(len(abs_ic)):
            # choose
            self.__p = list((1 - self.gamma) * w / sum(self.__w) + self.gamma / self.n_factor for w in self.__w)
            chosen_idx = self.__draw(self.__p)

            # update
            # full feedback
            # for i in range(self.n_factor):
            #     self.__w[i] *= np.exp(self.gamma * abs_ic[t][i] / self.n_factor / self.__p[i])
            # bandit feedback
            if chosen_idx == abs_ic[t].index(max(abs_ic[t])):
                self.__w[chosen_idx] *= np.exp(self.gamma * abs_ic[t][chosen_idx] / self.n_factor / self.__p[chosen_idx])

            weight = [0] * self.n_factor
            weight[chosen_idx] = 1
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