import numpy as np
from numpy.linalg import norm
from timeit import default_timer as timer
import warnings

warnings.filterwarnings("ignore")


def fwstep(grad, xi, sigma):
    Sgn = np.sign(grad)
    Sgn[Sgn == 0] = 1
    b = -np.abs(grad) / (1 + xi * Sgn)
    idx = np.argmin(b)
    ufw = np.zeros_like(xi)
    ai_sgn = Sgn[idx]
    ufw[idx] = -sigma * ai_sgn / (1 + xi[idx] * ai_sgn)
    return ufw


def awstep(a, x, sigma, xi_I, zeta):
    I = np.where(x != 0)
    a_I = a[I]
    x_I = x[I]
    xi_I = xi_I[I]
    u_I = sigma * (np.sign(xi_I) / (1 - np.abs(xi_I)))
    c_I = x_I / u_I
    b = a_I * u_I
    id_I = np.argmax(b)
    bmax = b[id_I]
    u_id = u_I[id_I]
    x_id = x_I[id_I]
    id = I[0][id_I]
    d_aw = x
    d_aw[id] = x_id - u_id
    c_id = c_I[id_I]
    alpha_aw = c_id / (1 - c_id)
    # print(np.sum(c_I))
    if np.sum(c_I) < 1:
        xi_id = xi_I[id_I]
        uprime = -sigma * np.sign(xi_id) / (1 + np.abs(xi_id))
        cprime = (1 - np.sum(c_I)) * (1 + np.abs(xi_id)) / 2
        c_id = c_id + (1 - np.sum(c_I)) * (1 - np.abs(xi_id)) / 2
        bprime = a_I[id_I] * uprime
        if bmax <= bprime:
            d_aw = x
            d_aw[id] = x_id - uprime
            alpha_aw = cprime / (1 - cprime)
        else:
            alpha_aw = c_id / (1 - c_id)
    alpha_aw = min(alpha_aw, zeta)
    return d_aw, alpha_aw


def FW_proj(
    obj,
    nabla_f,
    x_ini,
    sigma0,
    lambda_,
    epsilon,
    theta,
    update_rule="IRLS2",
    maxtime=100,
    verbose=False,
):
    tstart = timer()
    x = x_ini
    tol = 1e-5
    p = 1 / theta
    maxiter = 1e6
    isbdryb = 1
    isafw = 0
    c = 1e-6
    epi = 1e-2
    zeta = 1e5
    iter = 1
    alpha_init = 1
    alpha_fw = alpha_init
    traceTime = []
    traceFval = []

    grad = nabla_f(x)
    fval = obj(x)

    def h(x, lambda_, epsilon, theta):
        return lambda_ * norm(x, 1) - norm(np.abs(x) + epsilon, 1 / theta) ** (
            1 / theta
        )

    def nabla_h(x, lambda_, epsilon, theta):
        return (lambda_) * np.sign(x) - 1 / theta * np.sign(x) * (
            np.abs(x) + epsilon
        ) ** (1 / theta - 1)

    while True:
        x_last = x
        sigma = (
            sigma0
            + h(x, lambda_, epsilon, theta)
            - nabla_h(x, lambda_, epsilon, theta).T @ x
        ) / lambda_
        xi = nabla_h(x, lambda_, epsilon, theta) / lambda_
        u_fw = fwstep(grad.copy(), xi.copy(), sigma)
        d_fw = u_fw - x
        dderiv_fw = grad.T @ d_fw
        timeNow = timer() - tstart
        traceTime = [traceTime, timeNow]
        traceFval = [traceFval, fval]

        if iter % 1 == 0 and verbose:
            print(
                f"iter: {iter}, time: {timeNow}, obj:{obj(x)}, epsilon: {epsilon}, x_p: {norm(x,p)**p}, x_p_eps: {norm(np.abs(x)+epsilon,p)**p}"
            )

        if isafw:
            if norm(x) < 1e-8:
                d_aw = np.zeros_like(x_ini)
                dderiv_aw = 0
                alpha_aw = 0
            else:
                d_aw, alpha_aw = awstep(grad.copy(), x.copy(), sigma, xi.copy(), zeta)
                dderiv_aw = grad.T @ d_aw
            if dderiv_fw > dderiv_aw and alpha_aw > epi:
                is_aw = 1
                dir = d_aw
                dderiv = dderiv_aw
                alpha_init = alpha_aw
            else:
                is_aw = 0
                dir = d_fw
                dderiv = dderiv_fw
        else:
            is_aw = 0
            dir = d_fw
            dderiv = dderiv_fw
        iter_ls = 0
        alpha = alpha_init
        while True:
            xtilde = x + alpha * dir
            ftilde = obj(xtilde)
            if ftilde <= fval + c * alpha * dderiv:
                break
            else:
                alpha = alpha / 2
                iter_ls = iter_ls + 1

        if isbdryb and iter % 1 == 0:
            ratio = sigma0 / norm(np.abs(xtilde) + epsilon, p) ** p
            x = ratio**theta * xtilde
            fval = obj(x)
            if fval > ftilde:
                x = xtilde
                grad = nabla_f(x)
                fval = ftilde
            else:
                grad = nabla_f(x)
        else:
            x = xtilde
            grad = nabla_f(x)
            fval = ftilde

        if (
            np.abs(dderiv_fw) / max(np.abs(fval), 1) < tol
            or iter > maxiter
            or timeNow > maxtime
            or norm(x_last - x) < tol
        ):
            break
        iter = iter + 1

        if ~is_aw:
            alpha_fw = alpha
        if (~is_aw) and iter_ls == 0:
            alpha_init = min(max(alpha_fw * 2, 1e-8), 1)
        else:
            alpha_init = min(max(alpha_fw, 1e-8), 1)
        if update_rule == "IRLS1":
            if norm(x - x_last) < np.sqrt(epsilon) / 100:
                epsilon *= 0.1
                lambda_ *= 0.1 ** (1 / theta - 1)
        elif update_rule == "IRLS2":
            if norm(x - x_last) < np.sqrt(epsilon):
                epsilon *= 0.1
                lambda_ *= 0.1 ** (1 / theta - 1)
        elif update_rule == "naive":
            if iter % 10 == 0:
                epsilon *= 0.1
                lambda_ *= 0.1 ** (1 / theta - 1)
    trace = [traceTime, traceFval]
    return x, iter, fval, trace, timeNow


if __name__ == "__main__":
    import least_square

    # sys.stdout = utils.Logger("./log/dc_cs.log")
    np.random.seed(2023)
    dataSize = int(1e5)
    p = 0.7
    obj, grad, x, L = least_square.squared_loss()
    radius = norm(x, p) ** p
    x_ini = np.zeros_like(x)
    epsilon = 0.8 * ((radius - norm(x_ini, p) ** p) / x.shape[0]) ** (1 / p)
    theta = 1 / p
    lambda_ = epsilon ** (1 / theta - 1) / theta
    x_, iter, fval, trace, time = FW_proj(
        obj,
        grad,
        x_ini.copy(),
        radius,
        lambda_,
        epsilon,
        theta,
        maxtime=100,
        verbose=True,
    )
    print(time, norm(x - x_) / norm(x))
