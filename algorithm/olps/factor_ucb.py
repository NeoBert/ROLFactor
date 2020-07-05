import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np

parent_dir_writer = abspath('result/statistic/weight')

class FaUcb(object):
    def __init__(self, n_factor):
        self.n_factor = n_factor
        self.name = 'UCB'
        self.weights = []
        # method update parameter
        self.__w = [1] * self.n_factor

    def compute_weight(self, abs_ic):
        for t in range(len(abs_ic)):
            # choose
            mean = np.array(self.__w)
            bound = np.sqrt(2 * np.log(t+1) / mean)
            chosen_idx = np.argmax(mean / (t + 1) + bound)

            # update
            # full feedback
            # for i in range(self.n_factor):
            #     self.__w[i] += abs_ic[t][i]
            # bandit feedback
            if chosen_idx == abs_ic[t].index(max(abs_ic[t])):
                self.__w[chosen_idx] += abs_ic[t][chosen_idx]

            weight = [0] * self.n_factor
            weight[chosen_idx] = 1
            self.weights.append(weight)

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
