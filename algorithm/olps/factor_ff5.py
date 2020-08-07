import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np
from itertools import combinations

parent_dir_writer = abspath('result/statistic/weight')

class FaFf5(object):
    # FF5
    # chosen factor: 0 beta
    #                9 market_cap
    #                19 book_to_price_ratio -> 1 / pb_ratio (pd_ratio_reverse)
    #                24 roe_ttm
    #                27 total_asset
    def __init__(self, n_factor, n_choose=5):
        """
        Variable:   n_factor: number of factor
                    n_choose: number of chosen factor
                    name: name of method
                    weights: store historical weights
        """
        self.n_factor = n_factor
        self.n_choose = n_choose
        assert(self.n_choose == 5)
        self.name = 'FF5'
        self.weights = []


    def compute_weight(self, abs_ic):
        """
        Function:   compute portfolio weight
        Input:      abs_ic: float-array (n_time, n_factor)
                                    abs_ic from [0, n_time - 1]
        """
        for t in range(len(abs_ic)):
            weight = [0] * self.n_factor
            weight[0] = 1
            weight[9] = 1
            weight[19] = 1
            weight[24] = 1
            weight[27] = 1
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