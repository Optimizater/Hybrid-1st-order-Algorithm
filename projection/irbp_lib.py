#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:55:08 2021

@author: Xiangyu Yang
"""

from timeit import default_timer as timer

import numpy as np
import alternating_proj_lib
from numpy import linalg as LA


def get_lp_ball_projection(starting_point,
                    point_to_be_projected, 
                                        p,
                                   radius, 
                                  epsilon,
                                  Tau = 1.1,
                                  condition_right=100,
                                  tol=1e-8,
                                  MAX_ITER=1000,**kwargs):
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
        return point_to_be_projected, 0.0
    
    # Step 1 and 2 in IRBP.  
    n = point_to_be_projected.shape[0]
            
    signum = np.sign(point_to_be_projected) 
    yAbs = signum * point_to_be_projected  # yAbs lives in the positive orthant of R^n
    
    lamb = 0.0
    residual_alpha0 = (1. / n) * LA.norm((yAbs - starting_point) * starting_point - p * lamb * starting_point ** p, 1)
    residual_beta0 =  abs(LA.norm(starting_point, p) ** p - radius)
    
    cnt = 0
    
    # The loop of IRBP
    timeStart = timer()
    while True:
            
        cnt += 1
        alpha_res = (1. / n) * LA.norm((yAbs - starting_point) * starting_point - p * lamb * starting_point ** p, 1)
        beta_res = abs(LA.norm(starting_point, p) ** p - radius)
        
        if max(alpha_res, beta_res) < tol * max(max(residual_alpha0, residual_beta0),\
                                                              1.0) or cnt > MAX_ITER:
            timeEnd = timer()
            x_final = signum * starting_point # symmetric property of lp ball
            break
        
            
        # Step 3 in IRBP. Compute the weights
        weights = p * 1. / ((np.abs(starting_point) + epsilon) ** (1 - p) + 1e-12)
        
        # Step 4 in IRBP. Solve the subproblem for x^{k+1}
        gamma_k = radius - LA.norm(abs(starting_point) + epsilon, p) ** p + np.inner(weights, np.abs(starting_point))
            
        assert gamma_k > 0, "The current Gamma is non-positive"
         
        # Subproblem solver : The projection onto weighted l1-ball
        x_new, lamb = alternating_proj_lib.get_weightedl1_ball_projection(yAbs, weights, gamma_k)
        x_new[np.isnan(x_new)] = np.zeros_like(x_new[np.isnan(x_new)])
            
        # Step 5 in IRBP. Set the new relaxation vector epsilon according to the proposed condition
        condition_left = LA.norm(x_new - starting_point, 2) * LA.norm(np.sign(x_new - starting_point) * weights, 2) ** Tau
            
        if condition_left <= condition_right:
            theta = np.minimum(beta_res, 1. / np.sqrt(cnt)) ** (1. / p)
            epsilon = theta * epsilon
            
        # Step 6 in IRBP. Set k <--- (k+1)
        starting_point = x_new.copy()

        
    return  x_final, lamb, timeEnd-timeStart, cnt