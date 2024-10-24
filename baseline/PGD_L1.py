# -*- coding: utf-8 -*-
# @Time    : 2021/1/8 1:29
# @Author  : Xin Deng
# @FileName: test.py


import numpy as np
from numpy import linalg as LA
from timeit import default_timer as timer


def hyperplane(x, tau):
    """
    Projection onto the hyperplane.
    :param x: being projected point
    :param tau: radius of L1 ball
    :return: the projection z
    """
    dim = len(x)
    w = x - (sum(x) - tau) / dim
    # print w
    return w


def GPM(x, beta, tau, loss, gradf):
    """L1 ball exact projection
    :param tau: L1 ball radius.
    :param x: parameter vector of dimension n by 1 to be projected
    :param g: gradient g to be projected
    :param beta: stepsize of outer algorithm
    :return: projection of theta onto l1 ball ||x||_1 <= tau, and the iteration k
    """
    t0 = timer()
    for iter in range(int(5e5)):
        g = gradf(x)
        v = x - beta * g  # being projected point
        # print LA.norm(v, 1)
        n = len(x)
        # exact projection
        if LA.norm(v, 1) <= tau:
            z = v
            # print('It is already in the L1 ball.')
        else:
            # print('It is not in the L1 ball.')
            signum = np.sign(v)  # record the signum of v
            max_iter = 1e3
            v_temp = signum * v  # project v onto R+, named v_temp
            act_ind = range(n)  # record active index
            z = v
            for i in range(int(max_iter)):

                # calculate v_temp_act
                # v_temp_act = np.zeros((len(act_ind), 1))
                v_temp_act = v_temp[act_ind]

                w = hyperplane(v_temp_act, tau)  # projection onto hyperplane

                # update v_temp_act
                v_temp_act = np.maximum(w, 0.0)
                # update v_temp
                v_temp[act_ind] = v_temp_act

                # update act_ind
                # act_ind = []
                # for ind in range(n):
                #     if v_temp[ind] > 0:
                #         act_ind.append(ind)
                act_ind = v_temp > 0
                # print act_ind

                # termination criterion
                if sum(w < 0) == 0:
                    z = signum * v_temp
                    break

        norm_d = LA.norm(z - x)
        # print(loss(x),len(x[np.abs(x)>1e-5])/x.shape[0],timer() - t0)
        # print('||z-x||: %f' % norm_d)
        # termination criterion
        if norm_d <= 1e-8:
            # print("x")
            break
        x = z

    return x, timer() - t0, iter
