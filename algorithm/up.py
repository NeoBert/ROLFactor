# Universal Portfolio
import numpy as np
import copy


class UP(object):
    def __init__(self, n_stock):
        '''
        Variable:   n_stock: number of stock
                    version: version of relative price
                    name: name of method
                    weights: store historical weights
                    del0: minimum coordinate
                    delta: spacing of grid
                    n_sample: number of samples
                    n_step: number of steps in the random walk
        '''
        self.n_stock = n_stock
        self.name = 'up'
        self.weights = []
        self.del0 = 4e-3
        self.delta = 5e-3
        self.n_sample = 10
        self.n_step = 5

    def compute_weight(self, relative_price):
        '''
        Function:   compute portfolio weight
        Input:      relative_price: float-list (n_time, n_stock)
                                    relative_price from [0, n_time - 1]
                    stock_feature: float-list (n_time, n_stock, n_feature)
                                    stock_feature from [0, n_time - 1]
        '''
        for t in range(len(relative_price)):
            # print(t)
            # equal weight in the first round
            if t == 0:
                weight = [1 / self.n_stock] * self.n_stock
            else:
                relative_price = np.array(relative_price)
                samples = np.zeros((self.n_sample, self.n_stock))
                for s in range(self.n_sample):
                    w = [1 / self.n_stock] * self.n_stock
                    for i in range(self.n_step):
                        w_new = copy.deepcopy(w)
                        j = np.random.randint(self.n_stock - 1)
                        a = -1 if np.random.randint(2) == 0 else 1
                        # If 
                        w_new[j] = w[j] + a * self.delta
                        w_new[-1] = w[-1] - a * self.delta
                        if w_new[j] >= self.del0 and w_new[-1] >= self.del0:
                            x = self.__find_q(w, relative_price)
                            y = self.__find_q(w_new, relative_price)
                            pr = 1 if y >= x else y / x
                            temp = np.random.uniform()
                            if temp < pr:
                                w = w_new
                    samples[s] = np.array(w)

                weight_array = samples.sum(axis=0) / self.n_sample
                weight_array /= weight_array.sum(axis=0)
                weight = list(w for w in weight_array)
            self.weights.append(weight)

    def __find_q(self, weight, relative_price):
        '''
        Function:   find q
        Input:      weight: float-list (n_stock)
                    relative_price: float-array (n_time, n_stock)
        Output:     q
        '''
        p = np.prod(np.dot(relative_price, np.array(weight)), axis=0)
        q = p * min(1, np.exp(weight[-1] - 2 * self.del0 / (self.n_stock * self.delta)))
        return q
