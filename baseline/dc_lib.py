import warnings

warnings.filterwarnings("ignore")
import numpy as np
import time
from numpy.linalg import norm


def computing_x(v, u, y_i):
    """BY QIDENG"""
    """Used to compute vector x for every y_i"""
    l1 = v - (u + 1) * y_i
    l2 = (u - 1) * y_i - v
    l1 = np.maximum(l1, 0)
    l2 = np.maximum(l2, 0)
    return l1 - l2


def projection(v, u, tau):
    """BY QIDENG"""
    """
    Projects the sub_problem solution onto constraint set
            min ||x-v||^2  s.t.  ||x||_1 + <u,x> <= tau

    :param v: the solution of k-th sub_problem
    :param u, tau: l_1 norm constraint
    :return: the projected solution over the constraint
    """

    level = norm(v, ord=1) + u.dot(v)
    # if strict feasible, then return the point
    if level <= tau:
        return v

    ind = np.where((u != 1) & (u != -1))
    vind, uind = v[ind], u[ind]
    y1 = vind / (1 + uind)
    y2 = vind / (uind - 1)
    y = np.append(y1, y2)
    y.sort()

    y_opt = 0
    x_left = computing_x(v, u, y[0])
    x_right = computing_x(v, u, y[-1])

    result_left = np.dot(u, x_left) + np.linalg.norm(x_left, 1)
    result_right = np.dot(u, x_right) + np.linalg.norm(x_right, 1)
    if result_left < tau:
        x_0 = computing_x(v, u, 0)
        result_0 = np.dot(u, x_0) + np.linalg.norm(x_0, 1)
        y_opt = (result_left - tau) * y[0] / (result_0 - result_left) + y[0]

    if result_right > tau:
        x_some = computing_x(v, u, y[-1] + 1)
        result_some = np.dot(u, x_some) + np.linalg.norm(x_some, 1)
        y_opt = (tau - result_right) * (y[-1] + 1 - y[-1]) / (
            result_some - result_right
        ) + y[-1]

    else:
        # binary search
        lo, hi = 0, len(y) - 1
        while lo <= hi:
            half = (lo + hi) // 2
            x = computing_x(v, u, y[half])
            result = np.dot(u, x) + np.linalg.norm(x, 1)
            if result > tau:
                x_tune = computing_x(v, u, y[half + 1])
                result_tune = np.dot(u, x_tune) + np.linalg.norm(x_tune, 1)
                if result_tune < tau:
                    y_opt = (tau - result_tune) * (y[half] - y[half + 1]) / (
                        result - result_tune
                    ) + y[half + 1]
                    break
                lo = half + 1
            elif result < tau:
                x_tune = computing_x(v, u, y[half - 1])
                result_tune = np.dot(u, x_tune) + np.linalg.norm(x_tune, 1)
                if result_tune > tau:
                    y_opt = (tau - result) * (y[half - 1] - y[half]) / (
                        result_tune - result
                    ) + y[half]
                    break
                hi = half - 1
            else:
                y_opt = y[half]
                break
    return computing_x(v, u, y_opt)


def ACSA(
    grad,
    x0,
    u,
    gamma,
    tau,
    L0=1,
    gamma_u=2,
    gamma_d=2,
    line_search=True,
    maxiter=10,
    tol=1e-10,
):
    """The correspounding proj_nag function in QIDENG"""
    """
    Solve subproblem of LCPP
    min w^Tx + ||x-bar{x}||^2_2 + I_{g_k(x) <= eta_k}(x)
    """
    # gamma = 1/2L
    Lk = L0
    xk = x0.copy()
    Ak, ak = 0, 0
    sum_grad = np.zeros(len(xk))

    for t in range(maxiter):
        grad_count = 0
        sum_grad += ak * grad(xk)
        grad_count += 1
        v_k = projection((x0 - sum_grad / (gamma * Ak + 1)), u, tau)
        L = Lk

        while True:
            tem = (1 + gamma * Ak) / L
            a = tem + np.sqrt(tem**2 + 2 * Ak * tem)
            y = (Ak * xk + a * v_k) / (Ak + a)
            grad_y = grad(y)
            y_next = (gamma * x0 + L * y - grad_y) / (L + gamma)
            T_L = projection(y_next, u, tau)
            phi_prime = L * (y - T_L) + grad(T_L) - grad_y
            grad_count += 1
            if not line_search:
                break
            if phi_prime.dot(y - T_L) >= (norm(phi_prime, ord=2) ** 2 / L):
                break
            L *= gamma_u
        ak = a
        Ak += ak
        if line_search:
            Lk = L / gamma_d
        rel = norm(T_L - xk) / (1e-10 + norm(xk))
        xk, xk_proj = y, T_L
        xk = xk_proj
        if rel < tol:
            break
    return xk, Lk


def LCPP(
    obj,
    nabla_f,
    x0,
    eta_max,
    lambda_,
    p,
    epsilon,
    theta,
    trunc_thres=0.3,
    update_rule="IRLS2",
    gamma=0.5,
    verbose=False,
    tol=1e-8,
    maxiter=int(1e6),
    maxtime=100,
):
    """
    min phi{x}
    s.t. g(x) <= eta
         g(x) = lamdba ||x||_1 - h(x)
    """

    def h(x, lambda_, epsilon, theta):
        return lambda_ * norm(x, 1) - norm(np.abs(x) + epsilon, 1 / theta) ** (
            1 / theta
        )

    def nabla_h(x, lambda_, epsilon, theta):
        return (lambda_) * np.sign(x) - 1 / theta * np.sign(x) * (
            np.abs(x) + epsilon
        ) ** (1 / theta - 1)

    t0 = time.time()
    x = x0
    eta0 = lambda_ * norm(x0, ord=1) - h(x0, lambda_, epsilon, theta)
    eta = eta0
    L = 1

    for t in range(maxiter):
        # show processing
        if time.time() - t0 > maxtime:
            break
        delta = (eta_max - eta0) / ((t + 1) * (t + 2))
        eta += delta

        # truncate
        if eta_max - eta < trunc_thres:
            eta = eta_max - 1e-15
            epsilon = 1e-35

        tau = (
            eta
            + h(x, lambda_, epsilon, theta)
            - nabla_h(x, lambda_, epsilon, theta).T @ x
        ) / lambda_
        u = -nabla_h(x, lambda_, epsilon, theta) / lambda_
        """
        Subproblem 
        min phi(x) + gamma/2*||x-x^{k-1}||^2_2
        s.t. g_k(x) <= eta_k
        """
        x_sub, L = ACSA(nabla_f, x, u, gamma, tau, L0=L, line_search=True)
        # To show the sub problem constraints
        # gkx = lambda_*norm(x_sub,1)-h(x,lambda_,epsilon,theta)-nabla_h(x,lambda_,epsilon,theta).T @ (x_sub - x)

        rel = norm(x_sub - x) / (1 + norm(x))

        # epsilon updating rules
        # modify here with lambda_ = (epsilon**(1/theta-1)/theta)
        if update_rule == "IRLS1":
            if norm(x - x_sub) < np.sqrt(epsilon) / 100:
                epsilon *= 0.1
                lambda_ *= 0.1 ** (1 / theta - 1)
        elif update_rule == "IRLS2":
            if norm(x - x_sub) < np.sqrt(epsilon):
                epsilon *= 0.1
                lambda_ *= 0.1 ** (1 / theta - 1)
        elif update_rule == "naive":
            if (t + 1) % 10 == 0 and epsilon > 1e-30:
                epsilon *= 0.1
                lambda_ *= 0.1 ** (1 / theta - 1)
        x = x_sub

        if verbose:
            print(
                f"iter: {t}, time:{time.time() - t0:.2f} obj: {obj(x):.3f} epsilon: {epsilon}"
            )
            # print("constraint: ",eta_max,eta,norm(x,p)**p,norm(np.abs(x)+epsilon,p)**p)
        if rel < tol:
            break
    return x, time.time() - t0, t


if __name__ == "__main__":
    import least_square
    import sys
    from numpy import linalg as LA

    sys.path.append("./")
    # sys.stdout = utils.Logger("./log/dc_cs.log")
    np.random.seed(2023)
    dataSize = int(1e5)
    p = 0.5
    obj, grad, x, L = least_square.squared_loss()
    radius = norm(x, p) ** p
    # v = np.random.uniform(0,1,size = x.shape)
    # x_ini = 0.1*(radius/LA.norm(v,1)*v)**(1/p)
    x_ini = np.zeros_like(x)
    epsilon = 0.8 * ((radius - norm(x_ini, p) ** p) / dataSize) ** (1 / p)
    theta = 1 / p
    lambda_ = epsilon ** (1 / theta - 1) / theta
    x_DC, t_DC, iter_DC = LCPP(
        obj,
        grad,
        x_ini,
        radius,
        lambda_,
        p,
        epsilon,
        theta,
        tol=1e-6,
        gamma=0.5 * L,
        maxtime=5000,
        verbose=True,
    )
    print(t_DC, norm(x_DC - x) / norm(x))
