import copy
import pandas as pd
import numpy as np
import datetime


def init(context):
    context.industry = get_industry('银行')
    context.weight = np.ones(len(context.industry)) / len(context.industry)
    context.n_stock = len(context.industry)
    context.del0 = 4e-3
    context.delta = 5e-3
    context.n_sample = 10
    context.n_step = 5


def before_trading(context):
    now = context.now
    yesterday = context.now - datetime.timedelta(days=1)
    x = get_price_change_rate(context.industry, 1).values
    x = (x + 1).tolist()[0]

    relative_price = np.array(relative_price)
    samples = np.zeros((context.n_sample, context.n_stock))
    for s in range(self.n_sample):
        w = [1 / context.n_stock] * context.n_stock
        for i in range(context.n_step):
            w_new = copy.deepcopy(w)
            j = np.random.randint(context.n_stock - 1)
            a = -1 if np.random.randint(2) == 0 else 1
            w_new[j] = w[j] + a * context.delta
            w_new[-1] = w[-1] - a * context.delta
            if w_newå[j] >= context.del0 and w_new[-1] >= context.del0:
                x = find_q(context, w, relative_price)
                y = find_q(context, w_new, relative_price)
                pr = 1 if y >= x else y / x
                temp = np.random.uniform()
                if temp < pr:
                    w = w_new
        samples[s] = np.array(w)

    weight_array = samples.sum(axis=0) / self.n_sample
    weight_array /= weight_array.sum(axis=0)
    context.weight = list(w for w in weight_array)


def find_q(self, context, weight, relative_price):
    p = np.prod(np.dot(relative_price, np.array(weight)), axis=0)
    q = p * min(1, np.exp(weight[-1] - 2 * self.del0 / (self.n_stock * self.delta)))
    return q


def handle_bar(context, bar_dict):
    for index in range(len(context.weight)):
        order_target_percent(context.industry[index], context.weight[index])


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass
