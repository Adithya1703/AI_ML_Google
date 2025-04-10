import numpy as np
import matplotlib.pyplot as plt

plt.style.use('deeplearning.mplstyle')
dlblue = '#0096ff';
dlorange = '#FF9300';
dldarkred = '#C00000';
dlmagenta = '#FF40FF';
dlpurple = '#7030A0';
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
dlc = dict(dlblue='#0096ff', dlorange='#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')


# Regression Routines

def compute_cost_matrix(X, y, w, b, verbose=False):
    m = X.shape[0]  # gives no.of rows in X
    f_wb = X @ w + b  # matrix multiplication

    '''
    we have X(featues) like [1, 2, 3](as transposed in single column and 3 rows)
    we have to multiple each with weight and add bias right so instead of using
    .dot in each iteration we can directly use @ to do this
    what it does is as follows
    [1].200 + 100 = [300]
    [2].200 + 100 = [500]
    [3].200 + 100 = [700]
    as single matrix with 1 column three rows
    '''
    total_cost = ((1 / 2) * m) * np.sum((f_wb - y) ** 2)  # MSE final error cost which we need to decrease
    return total_cost


def compute_gradient_descent(X, y, w, b):
    m, n = X.shape
    f_wb = X @ w + b
    e = f_wb - y
    dj_dw = (1 / m) * (X.T @ e)
    dj_db = (1 / 2) * np.sum(e)
    return dj_db, dj_dw


# using loop instead of @

def compute_cost(X, y, w, b):
    m = X.shape[0]  # number of training examples
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  # prediction for i-th example
        cost = cost + (f_wb_i - y[i]) ** 2  # squared error
    cost = cost / (2 * m)
    return cost
