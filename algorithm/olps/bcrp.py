# Best Constant Rebalanced Portfolio

import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np
import cvxpy as cp

parent_dir_writer = abspath('result/statistic/weight')

class Bcrp(object):
    # Best constant rebalanced portfolio method
    def __init__(self, n_stock):
        """
        Variable:   n_stock: number of stock
                    version: version of relative price
                    name: name of method
                    weights: store historical weight
        """
        self.n_stock = n_stock
        self.name = 'bcrp'
        self.weights = []


    def compute_weight(self, relative_price):
        """
        Function:   compute portfolio weight
        Input:      relative_price: float-list (n_time, n_stock)
                                    relative_price from [0, n_time - 1]
        """
        relative_price = np.array(relative_price)
        weight_array = self.__linear_opt(relative_price)
        weight = list(w for w in weight_array)
        for t in range(len(relative_price)):
            self.weights.append(weight)


    def __linear_opt(self, projection):
        """
        Function:   argmax(p): (sum(log(projection_space * p)))
        Input:      projection: float-list (n_time, n_stock)
        Output:     result: optimization result, np.array(n_stock)
        """
        p = cp.Variable(self.n_stock)
        projection = np.array(projection)
        objective = cp.Maximize(cp.sum(cp.log(projection * p)))
        constraints = [p >= 0, cp.sum(p) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        # print("status: ", prob.status)
        result = np.array(p.value).squeeze()
        return result


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