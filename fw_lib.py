#!/usr/bin/env ppoint_to_be_projectedthon3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 14:31:39 2021

@author: starting_pointiangpoint_to_be_projectedu point_to_be_projectedang
"""

from timeit import default_timer as timer

import numpy as np
from numpy import linalg as LA

import alternating_proj_lib


def Frank_Wolfe_Lp(starting_point,
                   point_to_be_projected, 
                   p, 
                   radius,
                   mu=0.3,
                   tol = 1e-10,**kwargs):  # sourcery skip: low-code-quality
    
    """Gets the solution of Lp ball constrained minimization problem, which reads
    
    min f(starting_point)  s.t. ||starting_point||_p^p <= r,
    where 0 < p < 1.
    
    Args:
    ----------
    starting_point: Iterates of FW.
    point_to_be_projected: Point to be projected.
    p: p parameter for lp-ball.
    radius: the radius of the Lp-ball
    mu: the step-size of the projected gradient descent method.
    tol: specifpoint_to_be_projecteding the tol for the determination of the boundarpoint_to_be_projected case.
    """
    
    assert p > 0 and p < 1 and radius > 0 and mu > 0 and tol > 0 
    
    EPS = np.finfo(np.float64).eps
    
    stopping_tol = 1e-5     # the tolerance of subproblem stopping criteria
    
    cnt = 0          # the number of total iteration
    cnt_proj = 0     # the number of the call of projected gradient descent method
    cnt_fw = 0       # the number of the call of Frank-Wolfe method
     
    
    timeStart = timer()
    while True:
        cnt += 1
        grad = starting_point - point_to_be_projected # gradient of objective
    
        # Step 8: Boudnary case
        if abs(radius - LA.norm(starting_point, p) ** p) <= tol:
            """
            If the current point is on the boundary of the Lp-ball, applied with gradient projection method.
            """

            cnt_proj += 1
            starting_point_pre = starting_point.copy()  # type: ignore
            z = starting_point - mu * grad # the point to be projected
            
            # different signs and zero components: constraints: sgn(starting_point_{Ik}^{k})\circ starting_point_{Ik} >= 0
            ind = np.where(starting_point * z <= 0)[0] 
            if any(ind):
                z[ind] = 0
            
            act_ind = np.where(abs(starting_point) > EPS)[0]
            if any(act_ind):
            
                mask = np.ones(len(z), dtype=bool)
                mask[[act_ind]] = False
                z[mask] = 0
    
                weights = np.zeros(starting_point.shape[0])  # type: ignore
                weights[act_ind] = p * abs(starting_point[act_ind] + EPS) ** (p - 1)  # type: ignore
                
                radius_L1 = radius - LA.norm(starting_point[act_ind], p) ** p + weights[act_ind].dot(abs(starting_point[act_ind]))   # type: ignore

            # Subproblem solution: Performe weighted L1 ball projection
            starting_point, lamb = alternating_proj_lib.get_weightedl1_ball_projection(z, weights, radius_L1) # alternating projection
           
            proj_res = LA.norm(starting_point - starting_point_pre, 2)
            if proj_res < stopping_tol:
                break
            
        # Step 3: Interior case
        elif LA.norm(starting_point, p) ** p < radius: 
            """
            If the current point is inside the Lp-ball, applied with Frank-Wolfe method.
            """
            cnt_fw += 1
            grad_abs = abs(grad)
            max_ind = np.argmax(grad_abs) # return the first occurrence 

            # Derive the transformed variable
            z = np.zeros(starting_point.shape[0])
            z[max_ind] = radius

            # Derive the original variable and the direction
            s_abs = z ** (1 / p)
            fw_sol = - s_abs * np.sign(grad) # solution of FW subproblem
            desect_dir = fw_sol - starting_point

            # Stopping criteria
            fw_res = grad.dot(-desect_dir)
            if fw_res < stopping_tol:
                
                break

            # Step 5: set step-size
            alpha_bar = fw_res / (LA.norm(desect_dir) ** 2 + EPS) # Lipschitz constant is 1 in this case
            # Step 6: Using the bisection procedure to find an appropriate step-size
            if LA.norm(starting_point + alpha_bar * desect_dir, p) > radius ** (1/p):
                """
                Apply the bisection method to find a root of equation over interval (0, alpha_bar)
                    ||starting_point^k + gamma * d||_p^p = radius
                """
                alpha_r = alpha_bar
                alpha_l = 0.0
                alpha_bi = alpha_l + 0.5 * (alpha_r - alpha_l)
                res =   LA.norm(starting_point + alpha_bi * desect_dir, p) ** p - radius
                
                while abs(res) > tol:
                    if res > 0:
                        alpha_r = alpha_bi
                    else:
                        alpha_l = alpha_bi

                    alpha_bi = alpha_l + 0.5 * (alpha_r - alpha_l)
                    res =  LA.norm(starting_point + alpha_bi * desect_dir, p) ** p  - radius
                    
            else:
                alpha_bi = 1.0
            
            # Step 7: Compute new interates
            alpha = min(1.0, alpha_bi, alpha_bar)
            starting_point += alpha * desect_dir
            
        else:
            break
    timeEnd = timer()
    return starting_point, timeEnd-timeStart, cnt








