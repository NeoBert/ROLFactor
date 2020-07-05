import pandas as pd
from os.path import abspath, join, dirname, exists
import os

parent_dir_writer = abspath('result/statistic/weight')

class FaBest(object):
    # Best portfolio method
    def __init__(self, n_factor):
        """
        Variable:   n_factor: number of stock
                    name: name of method
                    weights: store historical weights
        """
        self.n_factor = n_factor
        self.name = 'Best'
        self.weights = []


    def compute_weight(self, abs_ic):
        """
        Function:   compute portfolio weight
        Input:      abs_ic: float-list (n_time, n_factor)
                                    abs_ic from [0, n_time - 1]
        """
        for ic in abs_ic:
            max_idx = ic.index(max(ic))
            weight = [0] * self.n_factor
            weight[max_idx] = 1
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