# Best Portfolio

import pandas as pd
from os.path import abspath, join, dirname, exists
import os

parent_dir_writer = abspath('result/statistic/weight')

class Best(object):
    # Best portfolio method
    def __init__(self, n_stock):
        """
        Variable:   n_stock: number of stock
                    name: name of method
                    weights: store historical weights
        """
        self.n_stock = n_stock
        self.name = 'best'
        self.weights = []


    def compute_weight(self, relative_price):
        """
        Function:   compute portfolio weight
        Input:      relative_price: float-list (n_time, n_stock)
                                    relative_price from [0, n_time - 1]
        """
        for price in relative_price:
            max_idx = price.index(max(price))
            weight = list(0 for i in range(self.n_stock))
            weight[max_idx] = 1
            self.weights.append(weight)

    
    def write_weight(self, file_name):
        """
        Function:   write weights
        Input       file_name: name of file
        """
        file_name = 'weight-' + file_name + '.csv'
        if exists(parent_dir_writer) == False:
            os.mkdir(parent_dir_writer)
        path = abspath(join(parent_dir_writer, file_name))
        pd_data = pd.DataFrame(self.weights)
        pd_data.to_csv(path, index=False, sep=',')