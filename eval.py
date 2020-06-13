# Evaluation of result (1. cumulative wealth, 2. sharp ratio, 3. max drawdown, 4. volatility)

from os.path import abspath, join, dirname, exists
import os
import pandas  as pd
import numpy as np

parent_dir_writer = abspath(join(dirname(__file__), 'result/statistic'))


class Eval(object):
    # Evaluation of result
    def __init__(self, relative_price=None, version='ver1', weight=None, frequency='daily', transaction_cost=0):
        """
        Input:      relative_price: list (n_time, n_stock)
                    version: string, version of relative_price
                            'ver0': (relative price - 1) * 100
                            'ver1': relative price
                    weight: list (n_time, n_stock)
                    frequency: string (monthly, weekly, daily)
                    transaction_cost: float
        Variable:   n_time: int
                    frequency: frequency of dataset
                    transaction_cost: float
                    metrics: string-list
                    gross_returns: list (n_time)
                    cumulative_returns: list (n_time)
                    cumulative_wealth: float
                    apy: float
                    sharp_ratio: float
                    calmar_ratio: float
                    max_drawdown: float
                    volatility: float
                    turnover: float
                    turnover_list: float-list (n_time)
        """
        if weight is None:
            weight = []
        if relative_price is None:
            relative_price = []
        self.n_time = len(relative_price)
        self.frequency = frequency
        self.transaction_cost = transaction_cost
        self.metrics = ['cumulative_wealth',
                        'apy',
                        'sharp_ratio',
                        'calmar_ratio',
                        'max_drawdown',
                        'volatility',
                        'turnover']
        if version == 'ver0':
            relative_price = np.array(relative_price) / 100 + 1
        self.turnover_list = []

        self.gross_returns = []
        rp_w = relative_price * np.array(weight)
        rp_w_sum = rp_w.sum(axis=1)
        weight_next_array = np.roll(np.array(weight), shift=-1, axis=0)
        weight_nor_array = rp_w / rp_w_sum.reshape(rp_w.shape[0], 1)
        weight_nor_array[-1] = weight_next_array[-1]  # no transaction cost at end of last period
        delta = np.abs(weight_next_array - weight_nor_array)
        for i in range(rp_w.shape[0]):
            l1_norm = delta[i].sum()
            self.turnover_list.append(l1_norm)
            gross_return = rp_w_sum[i] * (1 - self.transaction_cost * l1_norm)
            self.gross_returns.append(gross_return)
        # fake turnover
        # self.test_list = []
        # weight_test_array = np.array(weight)
        # weight_test_array[-1] = weight_next_array[-1]
        # delta = np.abs(weight_next_array - weight_test_array)
        # for i in range(rp_w.shape[0]):
        #     self.test_list.append(delta[i].sum())

        self.cumulative_returns = []
        cumulative_return = 1
        for gross_return in self.gross_returns:
            cumulative_return = cumulative_return * gross_return
            self.cumulative_returns.append(cumulative_return)

        self.cumulative_wealth = self.__get_cumulative_wealth()
        self.apy = self.__get_apy()
        self.sharp_ratio = self.__get_sharp_ratio()
        self.max_drawdown = self.__get_max_drawdown()
        self.calmar_ratio = self.__get_calmar_ratio()
        self.volatility = self.__get_volatility()
        self.turnover = self.__get_turnover()

    def __get_cumulative_wealth(self):
        return self.cumulative_returns[-1]

    def __get_apy(self):
        if self.frequency == 'monthly':
            year = self.n_time / 11.45  # 252/22
        elif self.frequency == 'weekly':
            year = self.n_time / 50.4  # 252/5
        elif self.frequency == 'daily':
            year = self.n_time / 252
        return np.power(self.cumulative_wealth, 1 / year) - 1

    def __get_sharp_ratio(self):
        rf = 0.04  # risk free return = treasury bills
        mean = self.apy - rf
        stdv = np.std(np.array(self.gross_returns))  # format into array to accelerate
        return mean / stdv

    def __get_calmar_ratio(self):
        return self.apy / self.max_drawdown

    def __get_max_drawdown(self):
        max_drawdown = 0
        for i in range(len(self.cumulative_returns)):
            current_max = max(self.cumulative_returns[: (i + 1)])
            current_drawdown = (current_max - self.cumulative_returns[i]) / current_max
            max_drawdown = max(max_drawdown, current_drawdown)
        return max_drawdown

    def __get_volatility(self):
        if self.frequency == 'monthly':
            H = 11.45  # 252/22
        elif self.frequency == 'weekly':
            H = 50.4  # 252/5
        elif self.frequency == 'daily':
            H = 252
        volatility = np.sqrt(H) * np.std(self.gross_returns)
        return volatility

    def __get_turnover(self):
        turnover = sum(self.turnover_list) / len(self.turnover_list)
        return turnover

    def print_info(self):
        '''
        Function:   print evaluation results
        '''
        print('============================')
        print('Transaction cost:    %.3f' % self.transaction_cost)
        print('Cumulative wealth:   %.3f' % self.cumulative_wealth)
        print('APY:                 %.3f' % self.apy)
        print('Sharp ratio:         %.3f' % self.sharp_ratio)
        print('Calmar ratio:        %.3f' % self.calmar_ratio)
        print('Maximum drawdown:    %.3f' % (100 * self.max_drawdown))
        print('Volatility:          %.3f' % (100 * self.volatility))
        print('Turnover:            %.3f' % (100 * self.turnover))

    def gen_df(self, data=[]):
        '''
        Function:   generate DataFrame
        Input:      data: list (n_data)
        Output:     res: DataFrame
        '''
        res = pd.DataFrame()
        if len(data) == 0:
            res['cumulative_wealth'] = [self.cumulative_wealth]
            res['apy'] = [self.apy]
            res['sharp_ratio'] = [self.sharp_ratio]
            res['calmar_ratio'] = [self.calmar_ratio]
            res['max_drawdown'] = [self.max_drawdown]
            res['volatility'] = [self.volatility]
            res['turnover'] = [self.turnover]
        else:
            for index, metric in enumerate(self.metrics):
                res[metric] = [data[index]]
        return res

    def write_info(self, file_name):
        '''
        Function:   write evaluation results
        Input:      file_name: name of file
        '''
        file_name = 'eval-' + file_name + '-c=' + str(self.transaction_cost) + '.csv'
        path = join(parent_dir_writer, 'eval')
        if exists(path) == False:
            os.mkdir(path)
        path = abspath(join(path, file_name))
        info = self.gen_df()
        info.to_csv(path, index=False, sep=',')

    def dup_write_info(self, file_name, dup=0, is_last=False):
        '''
        Function:   write evaluation results with duplication
        Input:      file_name: name of file
                    dup: int
                    is_last: bool
        '''
        if dup == 0:
            self.write_info('dup-' + file_name)
        else:
            file_name = 'eval-dup-' + file_name + '-c=' + str(self.transaction_cost) + '.csv'
            path = abspath(join(parent_dir_writer, 'eval', file_name))
            old = pd.read_csv(path, ',')
            new = self.gen_df()
            info = old.append(new, ignore_index=True)
            if is_last == True:
                data = []
                for metric in self.metrics:
                    data.append(np.mean(info[metric].values))  # compute mean of each metrics
                info = info.append(self.gen_df(data), ignore_index=True)
            info.to_csv(path, index=False, sep=',')

    def write_cumulative_wealth(self, file_name):
        '''
        Function:   write cumulative wealth
        Input:      file_name: name of file
        '''
        file_name = 'cw-' + file_name + '-c=' + str(self.transaction_cost) + '.csv'
        path = join(parent_dir_writer, 'cw')
        if exists(path) == False:
            os.mkdir(path)
        path = abspath(join(path, file_name))
        pd_data = pd.DataFrame(self.cumulative_returns)
        pd_data.to_csv(path, index=False, sep=',')

    def write_periodic_return(self, file_name):
        '''
        Function:   write gross return of each period
        Input:      file_name: name of file
        '''
        file_name = 'return-' + file_name + '-c=' + str(self.transaction_cost) + '.csv'
        path = join(parent_dir_writer, 'return')
        if exists(path) == False:
            os.mkdir(path)
        path = abspath(join(path, file_name))
        pd_data = pd.DataFrame(self.gross_returns)
        pd_data.to_csv(path, index=False, sep=',')
