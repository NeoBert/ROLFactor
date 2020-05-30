from algorithm.algo import Algo
import numpy as np
import pandas as pd
from algorithm import tools
import scipy.stats
from numpy.linalg import inv
from numpy import diag, sqrt, log, trace


class CWMR(Algo):
    PRICE_TYPE = 'ratio'
    REPLACE_MISSING = True

    def __init__(self, eps=-0.5, confidence=0.95):
        """
        :param eps: Mean reversion threshold
        :param confidence: confidence
        """
        super(CWMR, self).__init__()

        if not (0 <= confidence <= 1):
            raise ValueError('confidence must be from interval [0,1]')
        self.eps = eps
        self.theta = scipy.stats.norm.ppf(confidence)

    def init_weights(self, m):
        return np.ones(m) / m

    def init_step(self, X):
        m = X.shape[1]
        self.sigma = np.eye(m) / m ** 2

    def step(self, x, last_b):
        m = len(x)
        mu = last_b.T
        sigma = self.sigma
        theta = self.theta
        eps = self.eps
        x = x.T

        M = mu.T * x
        V = x.T * sigma * x
        x_upper = sum(diag(sigma) * x) / trace(sigma)

        mu, sigma = self.update(x, x_upper, mu, sigma, M, V, theta, eps)
        mu = tools.simplex_proj(mu)
        sigma = sigma / (m ** 2 * trace(sigma))
        self.sigma = sigma
        return mu

    def update(self, x, x_upper, mu, sigma, M, V, theta, eps):
        foo = (V - x_upper * x.T * np.sum(sigma, axis=1)) / M ** 2 + V * theta ** 2 / 2
        a = foo ** 2 - V ** 2 * theta ** 4 / 4
        b = 2(eps - log(M)) * foo
        c = (eps - log(M)) ** 2 - V * theta ** 2

        a, b, c = a[0, 0], b[0, 0], c[0, 0]

        lam = max(0, (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a), (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a))

        lam = min(lam, 1E+7)
        U_sqroot = 0.5 * (-lam * theta * V + sqrt(lam ** 2 * theta ** 2 * V ** 2 + 4 * V))
        mu = mu - lam * sigma * (x - x_upper) / M
        sigma = inv(inv(sigma) + theta * lam / U_sqroot * diag(x) ** 2)
        return mu, sigma
