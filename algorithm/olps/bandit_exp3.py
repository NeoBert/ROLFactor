import numpy as np
import random
import random


class BanditExp3(object):
    def __init__(self, n_dataset, n_method):
        """
        Variable:   n_comb: number of combination
                    gamma: EE rate
        """
        self.n_comb = n_dataset * n_method
        self.gamma = 0.1
        # method update parameter
        self.__w = [1 for i in range(self.n_comb)]
        self.__p = [(1 - self.gamma) * w / sum(self.__w) + self.gamma / self.n_comb for w in self.__w]

    def choose(self, context):
        """
        Function:   choose an arm
        Output:     chosen_idx: int
        """
        self.__p = list((1 - self.gamma) * w / sum(self.__w) + self.gamma / self.n_comb for w in self.__w)
        chosen_idx = self.__draw(self.__p)
        return chosen_idx

    def update(self, score):
        """
        Function:   update w
        Input:      score: float-list (n_dataset * n_method)
        """
        # full feedback
        # score = (score - np.min(score)) / (np.max(score) - np.min(score))
        for i in range(len(self.__w)):
            self.__w[i] *= np.exp(self.gamma * score[i] / self.n_comb / (self.__p[i]))

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


class BanditUcb(object):
    def __init__(self, n_dataset, n_method):
        """
        Variable:   n_comb: number of combination
        """
        self.n_comb = n_dataset * n_method
        # method update parameter
        self.__w = [1 for i in range(self.n_comb)]

    def choose(self, n_round):
        """
        Function:   choose an arm
        Input:      n_round: index of round
        Output:     chosen_idx: int
        """
        mean = np.array(self.__w)
        bound = np.sqrt(2 * np.log(n_round) / mean)
        chosen_idx = np.argmax(mean / n_round + bound)
        return chosen_idx

    def update(self, score):
        """
        Function:   update w
        Input:      score: float-list (n_dataset * n_method)
        """
        # full feedback
        score = (score - np.min(score)) / (np.max(score) - np.min(score))
        for i in range(len(self.__w)):
            self.__w[i] += score[i]


class BanditRegression(object):
    def __init__(self, n_dataset, n_method):
        """
        Variable:   n_comb: number of combination
                    gamma: EE rate
        """
        self.n_comb = n_dataset * n_method
        self.gamma = 0.1
        # method update parameter
        self.__w = [1e-5 for _ in range(self.n_comb)]
        self.__p = [(1 - self.gamma) * w / sum(self.__w) + self.gamma / self.n_comb for w in self.__w]

    def choose(self, context):
        context = np.average(context) - 1
        kernel = [w * context + random.uniform(-1e-4, 1e-4) for w in self.__w]
        return kernel.index(max(kernel))

    def update(self, score):
        for i in range(len(self.__w)):
            self.__w[i] += np.sign(self.__w[i]) * score[i]


class BanditLinucb(object):
    def __init__(self, n_dataset, n_method):
        """
        Variable:   n_comb: number of combination
        """
        self.n_comb = n_dataset * n_method
        self.n_context = 15
        self.alpha = 1
        # method update parameter
        self.__a = [np.eye(self.n_context)] * self.n_comb
        self.__b = [np.zeros(self.n_context)] * self.n_comb
        self.__context = np.empty((self.n_comb, self.n_context))

    def choose(self, context):
        """
        Input:  context: array (n_comb, c_context)
        """
        p = np.empty(self.n_comb)
        self.__context = context

        for i in range(self.n_comb):
            theta = np.linalg.inv(self.__a) @ self.__b
            mean = theta.T @ context[i]
            bound = self.alpha * np.sqrt(context[i].T @ np.linalg.inv(self.__a) @ context[i])
            p[i] = mean + bound
        chosen_idx = np.argmax(p)
        return chosen_idx

    def update(self, score):
        for i in range(self.n_comb):
            self.__a[i] += self.__context[i] @ self.__context[i].T
            self.__b[i] += score[i] * self.__context[i]

