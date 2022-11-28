#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 09:16:58 2021

@author: Xiangyu Yang

This lib contains utils to do l1-ball projection.
"""

import numpy as np

def get_hyperplane_projection(point_to_be_projected_act, 
                                            weights_act, 
                                                radius):
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
    
    dual = np.divide(numerator, denominator + EPS) # compute the dual variable for the weighted l1-ball projection problem
        
    x_sub = point_to_be_projected_act - dual * weights_act

    return x_sub,dual


def get_weightedl1_ball_projection(point_to_be_projected,
                                   weights, 
                                   radius):
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
        x_sol_hyper, dual = get_hyperplane_projection(point_to_be_projected_copy_act, weights_act, radius)
        
        # Update the active index set
        point_to_be_projected_copy_act = np.maximum(x_sol_hyper, 0.0)

        point_to_be_projected_copy[act_ind] = point_to_be_projected_copy_act.copy()
        
        act_ind = point_to_be_projected_copy > 0

        inact_ind_cardinality = sum(x_sol_hyper < 0)
        
        # Check the stopping criteria
        if inact_ind_cardinality == 0:
            x_opt = point_to_be_projected_copy * signum
            break

    # gap = radius -  np.inner(weights, abs(x_opt))
    # print(gap)
    return x_opt, dual


