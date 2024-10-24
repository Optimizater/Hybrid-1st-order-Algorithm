# -*- coding: utf-8 -*-
# @Time    : 2021/1/11 16:29
# @Author  : Xin Deng
# @FileName: PGP_Lp.py


import numpy as np
from numpy import linalg as LA
from timeit import default_timer as timer
from fw_proj_2 import fw_lp


# import irbp_lib
# %% the start of the inner loop
def hyperplane(y_proj_act, weights_act, gamma_k):
    # %% Projection onto hyperplane
    """
    Parameters
    ----------
    y_proj_act : being projected point with active index

    weights_act : the corresponding weights with the same active index as y_proj

    gamma_k : radius of weighted L1 ball of the k-th subproblem

    Returns
    -------
    w : the solution to the weighted l1 projection

    """

    scalar1 = np.dot(weights_act, y_proj_act) - gamma_k
    scalar2 = LA.norm(weights_act, ord=2) ** 2
    try:
        scalar3 = np.divide(scalar1, scalar2)
    except ZeroDivisionError:
        print("Error! - derivation zero for scalar2 =", scalar2)
    x_sub = y_proj_act - scalar3 * weights_act
    return x_sub, scalar3


def WeightSimplexProjection(n, y, signum, gamma_k, weights):
    """
    Parameters
    ----------
    n: dimension
    y: the point to be projected
    signum: An array that record the sign of the y
    gamma_k: the radius of the weighted l1 ball for the k-th subproblem
    weights: the fixed weights for the k-th subproblem

    Returns
    -------
    x_opt: the solution to the k-th subproblem

    """
    y_proj = (
        signum * y
    )  # elementwise multiplication of two ndarray : the datapoint to be projected
    act_ind = list(range(n))  # record active index
    while True:  # Because this algorithm terminates in finite steps
        # calculate y_bar_act
        y_proj_act = y_proj[act_ind]
        weights_act = weights[act_ind]

        x_sol_hyper, lamb = hyperplane(
            y_proj_act, weights_act, gamma_k
        )  # projection onto the hyperplane
        # %% dimension reduction and now the y_bar_act is the next projection point. Projection onto the non-negative orthont.
        # Once the non-negative components are detected, then elemiate them. These components are kept as 0 during next projection.
        y_proj_act = np.maximum(x_sol_hyper, 0.0)
        y_proj[act_ind] = y_proj_act  # back to the initial index

        # %% update the active set
        act_ind = []

        # %% We only need to find the nonzeros and then extract its index.
        arr_nonzero_y_proj = np.nonzero(y_proj > 0)
        arr_nonzero_y_proj = arr_nonzero_y_proj[0]
        act_ind = arr_nonzero_y_proj.tolist()

        signum_x_inner = np.sign(x_sol_hyper)
        inact_ind_cardinality = sum(
            elements < 0 for elements in signum_x_inner
        )  # inact_ind

        if inact_ind_cardinality == 0:
            x_opt = y_proj
            break

    return x_opt, lamb


def WeightLpBallProjection(n, x, y, p, radius, epsilon):
    """
    Lp ball exact projection
    param radius: Lp ball radius.
    param y_bar: parameter vector of dimension n by 1 to be projected
    return: projection of theta onto lp ball ||x||_p <= radius, and the iteration k

    """
    # %% Input parameters
    Tau, tol = 1.1, 1e-8
    Iter_max = 1e3
    M = 1e4

    # record the signum of the point to be projected to restore the solution
    signum = np.sign(y)
    x_final = np.zeros(n, dtype=np.float64)

    bar_y = signum * y  # Point lives in the positive orthant

    # store the  values
    res_alpha = []  # residual_alpha
    res_beta = []  # residual_beta

    count = 0  # the iteration counter

    if (
        LA.norm(y, p) ** p <= radius
    ):  # Determine the current ball whether it falls into the lp-ball.
        # print('The current point falls into the Lp ball. Please choose another new point!')
        return y, 0.0
    else:
        Flag_gamma_pos = "Success"
        while True:
            count = count + 1
            weights = p * (np.abs(x) + epsilon) ** (p - 1)
            gamma_k = radius - LA.norm(np.abs(x) + epsilon, p) ** p + np.dot(weights, x)

            if gamma_k <= 0:
                Flag_gamma_pos = "Fail"
                print("The current Gamma is not positive!!!")
                break

            # %% Calling algorithm2: weighted l1 ball projection
            x_opt, lamb = WeightSimplexProjection(
                n, y, signum, gamma_k, weights
            )  # x_opt: R^n

            num_nonzeros = np.count_nonzero(x_opt)

            # %% Compute the Objective value
            obj_k = 0.5 * LA.norm(signum * x_opt - y) ** 2

            # %% whether the update condition is triggerd for epsilon

            local_reduction = x_opt - x  # in the limit, it should be zero

            # Adapted by our current paper.
            condition_left = (
                LA.norm(local_reduction, ord=2) * LA.norm(weights, ord=2) ** Tau
            )
            condition_right = M

            # %% Determine whether to trigger the update condition
            error_appro = np.abs(LA.norm(x_opt, p) ** p - radius)

            if condition_left <= condition_right:
                epsilon = np.minimum(error_appro, 1 / (np.sqrt(count))) * epsilon

            eps_norm = LA.norm(epsilon, np.inf)

            # %% Checking the termination conditon

            act_ind_outer = []  # collect the active index from the (k+1)-th solution
            for ind in range(len(x_opt)):
                if x_opt[ind] > 0:
                    act_ind_outer.append(ind)

            # Determine the inactive set whether remains unchanged
            # %% Determine whether this collection is empty
            if (
                act_ind_outer
            ):  # Nonempty inactive set I. Our lemma shows the I(x^k) is nonempty

                residual_alpha = (1 / n) * np.sum(
                    np.abs((bar_y - x_opt) * x_opt - p * lamb * x_opt**p)
                )
                residual_beta = (1 / n) * np.abs(LA.norm(x_opt, p) ** p - radius)

                res_alpha += [residual_alpha]
                res_beta += [residual_beta]

                # %% Step 6 of our algorithm: go to the (k+1)-th iterate.
                x = x_opt  # (k+1)-th solution

                # print(
                #     '{0:3d}: Obj = {1:3.3f}, alpha = {2:4.3e}, beta = {3:4.3e}, eps = {4:4.3e}, dual = {5:3.3f}, #nonzeros = {6:2d}'.format(
                #         count, obj_k, residual_alpha, residual_beta, eps_norm, lamb, num_nonzeros), end=' ')
                # print()
                # %% Check the stopping criteria
                if (
                    np.maximum(residual_alpha, residual_beta)
                    <= tol * np.maximum(np.maximum(res_alpha[0], res_beta[0]), 1)
                    or count >= Iter_max
                ):
                    if count >= Iter_max:
                        Flag_gamma_pos = "Fail"
                        print("The solution is not so good")
                        break
                    else:
                        Flag_gamma_pos = "Success"
                        x_final = signum * x_opt  # element-wise product
                        break
        return x_final, lamb


def get_hyperplane_projection(point_to_be_projected_act, weights_act, radius):
    """Gets the hyperplane projection of a given point.

    Args:
        point_to_be_projected_act: Point to be projected with positive components.
        weights: the weights vector with positive components
        radius: The radius of weighted l1-ball.
    Returns:
        x_sub : The projection point

    """

    EPS = np.finfo(np.float64).eps

    numerator = np.inner(weights_act, point_to_be_projected_act) - radius
    denominator = np.inner(weights_act, weights_act)

    dual = np.divide(
        numerator, denominator + EPS
    )  # compute the dual variable for the weighted l1-ball projection problem

    x_sub = point_to_be_projected_act - dual * weights_act

    return x_sub, dual


def get_weightedl1_ball_projection(point_to_be_projected, weights, radius):
    """Gets the weighted l1 ball projection of given point.

    Args:
        point_to_be_projected: Point to be projected.
        weights: the weights vector.
        radius: The radius of weighted l1-ball.
    Returns:
        x_opt : The projection point.

    """

    signum = np.sign(point_to_be_projected)
    point_to_be_projected_copy = signum * point_to_be_projected

    act_ind = [True] * point_to_be_projected.shape[0]

    # The loop of the weight l1-ball projection algorithm
    while True:
        # Discarding the zeros
        point_to_be_projected_copy_act = point_to_be_projected_copy[act_ind]
        weights_act = weights[act_ind]

        # Perform projections in a reduced space R^{|act_ind|}
        x_sol_hyper, dual = get_hyperplane_projection(
            point_to_be_projected_copy_act, weights_act, radius
        )

        # Update the active index set
        point_to_be_projected_copy_act = np.maximum(x_sol_hyper, 0.0)

        point_to_be_projected_copy[act_ind] = point_to_be_projected_copy_act.copy()

        act_ind = point_to_be_projected_copy > 0

        inact_ind_cardinality = sum(x_sol_hyper < 0)

        # Check the stopping criteria
        if inact_ind_cardinality == 0:
            x_opt = point_to_be_projected_copy * signum
            break

    return x_opt, dual


def get_lp_ball_projection(
    starting_point,
    point_to_be_projected,
    p,
    radius,
    epsilon,
    Tau=1.1,
    condition_right=100,
    tol=1e-8,
    MAX_ITER=1000,
    MAX_TIME=200,
    **kwargs
):
    """Gets the lp ball projection of given point.

    Args:
    ----------
    point_to_be_projected: Point to be projected.
    starting_point: Iterates of IRBP.
    p: p parameter for lp-ball.
    radius: The radius of lp-ball.
    epsilon: Initial value of the smoothing parameter epsilon
    Tau, condition_right: hyperparameters
    Returns
    -------
    x_final : The projection point
    dual : The multiplier
    Flag_gamma_pos : whether IRBP successfully returens a solution
    count : The number of iterations

    """
    if LA.norm(point_to_be_projected, p) ** p <= radius:
        return point_to_be_projected, 0.0, 0

    # Step 1 and 2 in IRBP.
    n = point_to_be_projected.shape[0]

    signum = np.sign(point_to_be_projected)
    yAbs = signum * point_to_be_projected  # yAbs lives in the positive orthant of R^n

    lamb = 0.0
    residual_alpha0 = (1.0 / n) * LA.norm(
        (yAbs - starting_point) * starting_point - p * lamb * starting_point**p, 1
    )
    residual_beta0 = abs(LA.norm(starting_point, p) ** p - radius)

    cnt = 0

    # The loop of IRBP
    timeStart = timer()
    while True:

        cnt += 1
        alpha_res = (1.0 / n) * LA.norm(
            (yAbs - starting_point) * starting_point - p * lamb * starting_point**p, 1
        )
        beta_res = abs(LA.norm(starting_point, p) ** p - radius)

        if (
            max(alpha_res, beta_res)
            < tol * max(max(residual_alpha0, residual_beta0), 1.0)
            or cnt > MAX_ITER
            or timer() - timeStart > MAX_TIME
        ):
            timeEnd = timer()
            x_final = signum * starting_point  # symmetric property of lp ball
            break

        # Step 3 in IRBP. Compute the weights
        weights = p * 1.0 / ((np.abs(starting_point) + epsilon) ** (1 - p) + 1e-12)

        # Step 4 in IRBP. Solve the subproblem for x^{k+1}
        gamma_k = (
            radius
            - LA.norm(abs(starting_point) + epsilon, p) ** p
            + np.inner(weights, np.abs(starting_point))
        )

        assert gamma_k > 0, "The current Gamma is non-positive"

        # Subproblem solver : The projection onto weighted l1-ball
        x_new, lamb = get_weightedl1_ball_projection(yAbs, weights, gamma_k)
        x_new[np.isnan(x_new)] = np.zeros_like(x_new[np.isnan(x_new)])

        # Step 5 in IRBP. Set the new relaxation vector epsilon according to the proposed condition
        condition_left = (
            LA.norm(x_new - starting_point, 2)
            * LA.norm(np.sign(x_new - starting_point) * weights, 2) ** Tau
        )

        if condition_left <= condition_right:
            theta = np.minimum(beta_res, 1.0 / np.sqrt(cnt)) ** (1.0 / p)
            epsilon = theta * epsilon

        # Step 6 in IRBP. Set k <--- (k+1)
        starting_point = x_new.copy()

    return x_final, timeEnd - timeStart, cnt


def PGD_Lp(x, mu, p, radius, loss, gradf, verbose=False, maxtime=100):
    """ """
    Iter_max = int(1e3)
    n = len(x)
    t0 = timer()
    for i in range(Iter_max):
        grad = gradf(x)
        z = x - mu * grad
        x_pre = x

        # rand_num = np.random.uniform(0, 1, n)
        # abs_norm = LA.norm(rand_num, 1)
        # epsilon = 0.9 * (rand_num * radius / abs_norm) ** (1/p)  # ensure that the point is feasible.
        # # x, lamb = WeightLpBallProjection(n, np.zeros(n), z, p, radius, epsilon)
        # x,_,_ = get_lp_ball_projection(np.zeros(n), z, p, radius, epsilon)
        y = np.random.normal(0, 1, n)
        x_ini = (
            0.3 * (radius ** (1 / p)) * (np.abs(y, dtype=np.float64) / LA.norm(y, p))
        )
        solver = fw_lp(
            x_ini, p, radius, lambda x: 0.5 * LA.norm(x - z) ** 2, lambda x: x - z, Lf=1
        )
        x, _, _ = solver.solve(mu=1, verbose=False)

        if verbose:
            print(
                "{:5d}   Obj = {:3.3f}   Res = {:4.3e}   #nonzero = {}".format(
                    i, loss(x), LA.norm(x - x_pre), len(np.nonzero(x)[0])
                )
            )
        if LA.norm(x - x_pre) < 1e-8 or timer() - t0 > maxtime:
            break
    return x, timer() - t0, i
