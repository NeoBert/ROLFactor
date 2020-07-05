# Evaluation of result

from os.path import abspath, join, dirname, exists
import os
import pandas  as pd
import numpy as np

parent_dir_writer = abspath(join(dirname(__file__), 'result/statistic'))


class EvalFactor(object):
    # Evaluation of result
    def __init__(self, abs_ic=None, weight=None):
        if weight is None:
            weight = []
        if abs_ic is None:
            abs_ic = []
        self.n_time = len(abs_ic)
        self.reward = self.__get_reward(abs_ic, weight)

    def __get_reward(self, abs_ic, weight):
        reward = 0
        for i in range(self.n_time):
            reward += np.array(abs_ic[i]) @ np.array(weight[i])
        return reward


    def print_info(self):
        """
        Function:   print evaluation results
        """
        print('============================')
        print('Reward:    %.3f' % self.reward)