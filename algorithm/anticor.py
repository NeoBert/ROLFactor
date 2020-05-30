from algorithm.algo import Algo
from algorithm import tools
import numpy as np
import warnings


class Anticor(Algo):
    def __init__(self, window=30, c_version=True):
        super(Anticor, self).__init__()
        self.window = window
        self.c_version = c_version

    def weights(self, X):
        window = self.window
        port = X
        n, m = port.shape
        weights = 1 / m * np.ones(port.shape)

        CORR, EX = tools.rolling_corr(port, port.shift(window), window=window)
        for t in range(n - 1):
            M = CORR[t, :, :]
            mu = EX[t, :]

            claim = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    if mu[i] > mu[j] and M[i, j] > 0:
                        claim[i, j] += abs(M[i, j])
                        if M[i, i] < 0:
                            claim[i, j] += abs(M[i, i])
                        if M[j, j] < 0:
                            claim[i, j] += abs(M[j, j])
            transfer = claim * 0.0
            for i in range(m):
                total_claim = sum(claim[i, :])
                if total_claim != 0:
                    transfer[i, :] = weights[t, i] * claim[i, :] / total_claim
            weights[t + 1, :] = weights[t, :] + np.sum(transfer, axis=0) - np.sum(transfer, axis=1)
        return weights
