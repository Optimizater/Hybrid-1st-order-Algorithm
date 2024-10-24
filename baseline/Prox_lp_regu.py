import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from timeit import default_timer as timer


def ProximityLp(x_proj, mu):
    n = x_proj.shape[0]
    x_opt = np.zeros(n)
    act_ind = np.where(abs(x_proj) > (1.5 * mu ** (2 / 3)))[0]
    if any(act_ind):
        mask = np.zeros(n, dtype=bool)
        mask[[act_ind]] = True
        x_opt[mask] = (
            (2 / 3)
            * x_proj[mask]
            * (
                1
                + np.cos(
                    (2 / 3)
                    * np.arccos(
                        -(3 ** (3 / 2) / 4) * mu * abs(x_proj[mask]) ** (-3 / 2)
                    )
                )
            )
        )

    return x_opt


def lpRegu(x, step_size, loss, gradf, mu=1e-4, p=0.5):
    tol = 1e-10
    obj = [loss(x)]
    iter_ = 0
    t0 = timer()
    while True:
        grad = gradf(x)
        z = x - step_size * grad  # the resulting points

        # Proximity operation in vector manner
        x = ProximityLp(z, mu)

        iter_ += 1
        obj.append(loss(x) + mu * LA.norm(x, p) ** p)
        # print(obj[-1])
        if abs(obj[-1] - obj[-2]) < tol or iter_ > 1e4:
            break
    x_opt = x.copy()
    return x_opt, timer() - t0, iter_
