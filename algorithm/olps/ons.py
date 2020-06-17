# Online newton step

import pandas as pd
from os.path import abspath, join, dirname, exists
import os
import numpy as np
from cvxopt import solvers, matrix
import cvxpy as cp

parent_dir_writer = abspath(join(dirname(__file__), '../result/statistic/weight'))


class ONS(object):
    # Online newton step method
    def __init__(self, n_stock):
        """
        Variable:   n_stock: number of stock
                    version: version of relative price
                    name: name of method
                    weights: store historical weights
                    eta: learning rate
                    beta: for theoretical analysis
                    delta: market variability parameter (price fluctuate under delta)
                    A: np.mat (n_stock, n_stock)
                    b: np.mat (n_stock, 1)
        """
        self.n_stock = n_stock
        self.name = 'ons'
        self.weights = []
        # method parameter
        self.eta = 0.1
        self.__beta = 0.1 / 8
        self.delta = 1
        # method update parameter
        self.__A = np.mat(np.eye(self.n_stock))
        self.__b = np.mat(np.zeros(self.n_stock)).T

    def compute_weight(self, relative_price, stock_feature=None):
        """
        Function:   compute portfolio weight
        Input:      relative_price: float-list (n_time, n_stock)
                                    relative_price from [0, n_time - 1]
                    stock_feature: float-list (n_time, n_stock, n_feature)
                                    stock_feature from [0, n_time - 1]
        """
        relative_price = np.array(relative_price) / 100 + 1 if self.version == 'ver0' \
            else np.array(relative_price)
        for t in range(len(relative_price)):
            # equal weight in the first round
            if t == 0:
                weight = [1 / self.n_stock] * self.n_stock
            # following rounds
            else:
                log_return = np.dot(np.array(self.weights[-1]), relative_price[t - 1])
                grad = np.mat(np.array(relative_price[t - 1]) / log_return).T
                self.__A += grad * grad.T
                self.__b += (1 + 1 / self.__beta) * grad
                objective = self.delta * self.__A.I * self.__b
                origin_weight = self.__projection_opt(objective, self.__A)
                weight = list(
                    ((1 - self.eta) * w + self.eta / self.n_stock) for w in origin_weight)  # w = (1 - eta)w + eta*ucrp
            self.weights.append(weight)

    def __projection_opt(self, objective, projection_space):
        """
        Function:   projection in the norm induced by projection_space
                    argmin(p):(objective - p)^T * projection_space * (objective - p)
        Input:      objective: np.mat (n_stock, 1)
                    projection_space: np.mat (n_stock, n_stock)
        Output:     result: optimization result, np.array(n_stock)
        """
        p = cp.Variable(self.n_stock)
        G = -np.eye(self.n_stock)
        h = np.zeros(self.n_stock)
        A = projection_space
        b = objective

        obj = cp.Minimize(0.5 * cp.quad_form(p, 2 * A) - (2 * b.T) @ p)
        cons = [G @ p <= h, cp.sum(p) == 1]
        prob = cp.Problem(obj, cons)
        prob.solve(cp.CVXOPT)
        return np.array(p.value)

    def write_weight(self, file_name):
        file_name = 'weight-' + file_name + '.csv'
        if exists(parent_dir_writer) == False:
            os.mkdir(parent_dir_writer)
        path = abspath(join(parent_dir_writer, file_name))
        pd_data = pd.DataFrame(self.weights)
        pd_data.to_csv(path, index=False, sep=',')
