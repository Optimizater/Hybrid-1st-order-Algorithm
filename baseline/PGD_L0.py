# -*- coding: utf-8 -*-
# @Time    : 2021/1/8 1:29
# @Author  : Xin Deng
# @FileName: test.py


import numpy as np
from numpy import linalg as LA
from timeit import default_timer as timer


def PGD_L0(x, mu, radius, loss, gradf):
    """L1 ball exact projection
    :param tau: L1 ball radius.
    :param x: parameter vector of dimension n by 1 to be projected
    :param g: gradient g to be projected
    :param beta: stepsize of outer algorithm
    :return: projection of theta onto l1 ball ||x||_1 <= tau, and the iteration k
    """
    n = len(x)
    tol = 1e-8
    t0 = timer()
    for iter in range(int(1e4)):
        # print(loss(x))
        grad = gradf(x)
        x_pre = x
        z = x - mu * grad

        if len(np.nonzero(z)[0]) <= radius:
            x = z
        else:
            # Find the largest s elements (magnitude), i.e., find the smallest n-s elements
            ind_ord = np.argsort(abs(z))
            ind_ord = ind_ord[0 : n - int(radius)]
            x = z
            x[ind_ord] = 0.0

        res = LA.norm(x - x_pre)
        # res = LA.norm(A.dot(x) - y)
        # print('{}   obj = {}   res = {}'.format(iter, loss(A, x, y), res))
        # if res < tol:
        # print(loss(x))
        if res < tol:
            break

    return x, timer() - t0, iter
