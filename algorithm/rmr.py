from algorithm.algo import Algo
from algorithm import tools
import numpy as np
from .olmar import OLMAR
import pandas as pd


def norm(x):
    if isinstance(x, pd.Series):
        axis = 0
    else:
        axis = 1
    return np.sqrt((x ** 2).sum(axis=axis))


class RMR(OLMAR):
    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10., tau=0.001):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        :param tau: Precision for finding median. Recommended value is around 0.001. Strongly
                    affects algo speed.
        """
        super(RMR, self).__init__(window, eps)
        self.tau = tau

    def predict(self, x, history):
        y = history.mean()
        y_last = None
        while y_last is None or norm(y - y_last) / norm(y_last) > self.tau:
            y_last = y
            d = norm(history - y)
            y = history.div(d, axis=0).sum() / (1. / d).sum()
        return y / x


if __name__ == '__main__':
    tools.quickrun(RMR())
