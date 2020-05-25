from algorithm.algo import Algo
import numpy as np
import pandas as pd
from algorithm import tools


class OLMAR(Algo):
    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLMAR, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps

    def init_weights(self, m):
        return np.ones(m) / m

    def step(self, x, last_b, history):
        # calculate return prediction
        x_pred = self.predict(x, history.iloc[-self.window:])
        b = self.update(last_b, x_pred, self.eps)
        return b

    def predict(self, x, history):
        """ Predict returns on next day. """
        return (history / x).mean()

    def update(self, b, x, eps):
        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)
        lam = max(0., (eps - np.dot(b, x)) / np.linalg.norm(x - x_mean) ** 2)

        lam = min(100000, lam)

        b = b + lam * (x - x_mean)

        # project it onto simplex
        return tools.simplex_proj(b)


if __name__ == '__main__':
    tools.quickrun(OLMAR())
