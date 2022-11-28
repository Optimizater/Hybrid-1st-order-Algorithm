import warnings

warnings.filterwarnings('ignore')

from msilib.schema import Error

warnings.filterwarnings('ignore')
# import cv2
import numpy as np
import scipy
import utils
from matplotlib import pyplot as plt
from numpy import linalg as LA
from scipy.optimize import line_search


def ly(u,v,y):
    '''
    u, v : R^d (d,)
    y : R^1 
    '''
    d = u.shape[0]
    pos = sum((u[i]-1) * max( v[i] - (u[i]+1) * y , 0) for i in range(d))
    neg = -sum((u[i]+1) * max( (u[i]-1) * y - v[i] , 0) for i in range(d))
    zero = 2 * sum((u[i]-1) * max( v[i] - (u[i]+1) * y ,(u[i]-1) * y - v[i], 0) for i in range(d))
    return pos + neg + zero

def find_y(tau,u,v):
    '''
    Solve dual problem of 
    min ||x-v||^2_2
    s.t. ||x||_1 + <u,x> <= tau
    '''
    d = u.shape[0]
    breakingpoints = []
    bp_obj = []
    
    # find breakingpoints
    for i in range(d):
        breakingpoints.append(v[i]/(u[i]+1))
        breakingpoints.append(v[i]/(u[i]-1))
        breakingpoints.append(v[i]/u[i])
    
    for i in breakingpoints:
        bp_obj.append(ly(u,v,i))
        
    bp_obj = np.array(bp_obj)
    breakingpoints = np.array(breakingpoints)
    
    # sort 
    sorted_indice = np.argsort(breakingpoints)
    bp_obj = bp_obj[sorted_indice]
    breakingpoints = breakingpoints[sorted_indice]
    
    # discard y < 0
    bp_obj = bp_obj[breakingpoints>=0]
    breakingpoints = breakingpoints[breakingpoints>=0]
    
    # add y = 0 and y = inf
    inf = 1e10
    bp_obj = np.insert(bp_obj,0,ly(u,v,0))
    breakingpoints = np.insert(breakingpoints,0,0)
    bp_obj = np.append(bp_obj,ly(u,v,inf))
    breakingpoints = np.append(breakingpoints,inf)
    
    # find interval
    for i in range(len(breakingpoints)-1):
        if (bp_obj[i]-tau)*(bp_obj[i+1]-rau) <= 0:
            x1,y1 = breakingpoints[i],bp_obj[i]
            x2,y2 = breakingpoints[i+1],bp_obj[i+1]
            break
    return (y1-tau)/(y1-y2)*(x2-x1)+x1

def projection(tau,u,v):
    '''
    min ||x-v||^2_2
    s.t. ||x||_1 + <u,x> <= tau
    '''
    x = np.zeros_like(v)
    y = find_y(tau,u,v)
    for i in range(len(x)):
        x[i] = max(v[i] - (u[i]+1)*y, 0) - max((u[i]-1)*y-v[i], 0)
    return x

def ACSA(G,x0,u,gamma,tau):
    '''
    Solve subproblem of LCPP
    min w^Tx + ||x-bar{x}||^2_2 + I_{g_k(x) <= eta_k}(x)
    '''
    # gamma = 1/2L
    maxiter = 1e3
    x = x0
    xag = x0
    for t in range(1,maxiter):
        xmd = 2/(t+1)*x + (t-1)/(t+1)*xag
        w = 4/(t+1)*gamma*G(xmd)
        v = x + 0.5*w
        x = projection(tau,u,v)
        xag = 2/(t+1)*x+(t-1)/(t+1)*xag
    return x

def LCPP(x0,eta0,delta,h,nabla_h,f,nabla_f,gamma):
    '''
    min phi{x} 
    s.t. g(x) <= eta
         g(x) = lamdba ||x||_1 - h(x)
    '''
    maxiter = 1e3
    x = x0
    eta = eta0
    for k in range(maxiter):
        eta = eta + delta[k]
        tau = eta + h(x) - nabla_h(x).T @ x
        u = -nabla_h(x)
        
        '''
        Subproblem 
        min phi(x) + gamma/2*||x-x^{k-1}||^2_2
        s.t. g_k(x) <= eta_k
        '''
        # x_ini need to be feasible (not gauranteed)
        x_ini = np.zeros_like(x) 
        x = ACSA(nabla_f,x_ini,u,gamma,tau)
    return x

    