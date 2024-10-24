import numpy as np
from numpy import linalg as LA
import warnings
import sys

# Import the baseline file
sys.path.append("..")
warnings.filterwarnings("ignore")


def squared_loss(
    m=800,
    n=1000,
    s=100,
    loss_type="convex",
    A_type="Gaussian",
    x_type="discrete",
    noise_type="normal",
    noise_level=1e-4,
):
    if A_type == "Gaussian":
        A = np.random.standard_normal((m, n))

    elif A_type == "DCT":
        A = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                A[i, j] = np.cos(2 * np.pi * (j - 1) * np.random.uniform(0, 1))
        col_norms = np.linalg.norm(A, axis=0)
        A = A / col_norms

    if x_type == "discrete":
        x = np.zeros(n)
        for i in np.random.choice(n, s, replace=False):
            x[i] = 1 if np.random.uniform() > 0.5 else -1

    elif x_type == "normal":
        x = np.zeros(n)
        x[np.random.choice(n, size=s, replace=False)] = np.random.standard_normal(s)

    if noise_type == "normal":
        noise = np.random.normal(0, noise_level, m)  # * noise_level

    b = A @ x + noise

    if loss_type == "convex":

        def obj(z):
            return 0.5 * LA.norm(A @ z - b) ** 2

        def grad(z):
            return A.T @ (A @ z - b)

    elif loss_type == "nonconvex":

        def obj(z):
            return np.sum(np.log(0.5 * (A @ z - b) ** 2 + 1))

        def grad(z):
            axb = A @ z - b
            return A.T @ (axb / (0.5 * axb**2 + 1))

    return obj, grad, x, np.max(LA.eigvals(A.T @ A).real), A, b


def test(method, obj, grad, x_star, L, p=0.9):
    if method == "FW":
        from fw_lib import FW_LP

        radius = LA.norm(x_star, p) ** p
        v = np.random.uniform(0.0, 1.0, size=x_star.shape)
        x_ini = 0.9 * (radius / LA.norm(v, 1) * v) ** (1 / p)
        solver = FW_LP(x_ini, p, radius, obj, grad, Lf=L)
        x, t, iter = solver.solve(mu=1 / L, verbose=False, maxtime=3000)

    if method == "PGD_L1":
        from PGD_L1 import GPM

        radius = LA.norm(x_star, 1)
        v = np.random.uniform(0, 1, size=x_star.shape)
        x_ini = 0.9 * (radius / LA.norm(v, 1) * v)
        x, t, iter = GPM(x_ini, 1 / L, radius, obj, grad)

    if method == "PGD_L0":
        from PGD_L0 import PGD_L0

        radius = LA.norm(x_star, 0)
        v = np.random.uniform(0, 1, size=x_star.shape)
        x_ini = np.zeros_like(x_star)  # 0.9*(radius/LA.norm(v,1)*v)
        x, t, iter = PGD_L0(x_ini, 1 / L, radius, obj, grad)

    if method == "PGP_Lp":
        from PGP_Lp import PGD_Lp

        radius = LA.norm(x_star, p) ** p
        v = np.random.uniform(0, 1, size=x_star.shape)
        x_ini = 0.9 * (radius / LA.norm(v, 1) * v) ** (1 / p)
        x, t, iter = PGD_Lp(x_ini, 1 / L, p, radius, obj, grad, verbose=False)

    if method == "Prox_Lp":
        from Prox_lp_regu import lpRegu

        x_ini = np.zeros_like(x_star)
        x, t, iter = lpRegu(x_ini, 1 / L, obj, grad)

    if method == "DC":
        from dc_lib import LCPP

        radius = LA.norm(x_star, p) ** p
        x_ini = np.zeros_like(x_star)
        epsilon = 0.8 * ((radius - LA.norm(x_ini, p) ** p) / x_ini.shape[0] / 10) ** (1 / p)
        theta = 1 / p
        lambda_ = epsilon ** (1 / theta - 1) / theta
        x, t, iter = LCPP(
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
            maxtime=300,
            verbose=True,
            update_rule="naive",
        )

    if method == "fwzeng":
        from fwzeng_lib import FW_proj

        radius = LA.norm(x_star, p) ** p
        x_ini = np.zeros_like(x_star)
        epsilon = 0.8 * ((radius - LA.norm(x_ini, p) ** p) / x_ini.shape[0]) ** (1 / p)
        theta = 1 / p
        lambda_ = epsilon ** (1 / theta - 1) / theta
        x, iter, _, _, t = FW_proj(
            obj,
            grad,
            x_ini.copy(),
            radius,
            lambda_,
            epsilon,
            theta,
            maxtime=300,
            verbose=False,
        )

    return x, t, iter


if __name__ == "__main__":
    m = 500
    n = 1000
    s = int(n * 0.05)
    p = 0.9
    num_exps = 5

    loss_type = "convex"
    A_type = "Gaussian"
    x_type = "discrete"

    methods = ["FW", "PGD_L1", "PGD_L0", "PGP_Lp", "Prox_Lp", "DC"]
    results = {}
    output_template = {
        "(m,n,s,p)": (m, n, s, p),
        "loss_type": loss_type,
        "A_type": A_type,
        "x_type": x_type,
        "num_exps": num_exps,
        "num_succ": 0.0,
        "succ_time": 0.0,
        "avg_recover_time": 0.0,
        "recover_rate": 0.0,
    }
    for method in methods:
        results[method] = output_template.copy()
    for i in range(num_exps):
        obj, grad, x_star, L, A, b = squared_loss(
            m=m,
            n=n,
            s=s,
            loss_type=loss_type,
            A_type=A_type,
            x_type=x_type,
            noise_level=0,
        )
        for method in methods:
            x, t, iter = test(method, obj, grad, x_star.copy(), L, p=p)
            if LA.norm(x - x_star) / LA.norm(x_star) < 1e-3:
                results[method]["num_succ"] += 1.0
                results[method]["succ_time"] += t
    for method in methods:
        if results[method]["num_succ"] > 0:
            results[method]["recover_rate"] = (
                results[method]["num_succ"] / results[method]["num_exps"]
            )
            results[method]["avg_recover_time"] = (
                results[method]["succ_time"] / results[method]["num_succ"]
            )
    print(results[method])
