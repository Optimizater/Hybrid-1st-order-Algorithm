#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 14:31:39 2021

@author: Xiangyu Yang
"""

import numpy as np
from numpy import linalg as LA

def point_projected(N:int):
    
    """Generates the point to be projected
    Args:
        N (int): [The length of the point to be projected.]
    Returns:
        data (float): [The point to be projected, following the standard Normal distribution.]
    """ 

    mu, sigma = 0., 1. # mean and standard deviation    
    
    return np.random.normal(mu,sigma,N)


def least_square_loss(projection_point, point_to_be_projected):
    """
    Args:
    
    projection_point : the current solution of dimension n by 1.
    point_to_be_projected : The given point to be projected

    Returns:s
    
    The Objective value of the lp-ball projection problem, i.e., 0.5 * || x-y ||_2^2
    """
    return 0.5 * LA.norm(point_to_be_projected - projection_point,2)