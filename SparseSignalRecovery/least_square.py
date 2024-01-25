import sys
sys.path.append("./")
import numpy as np
from numpy import linalg as LA
import warnings
warnings.filterwarnings('ignore')

def squared_loss(m = 800, n = 1000, s = 100,loss_type = "convex", A_type = "Gaussian", x_type = "discrete",noise_type = "normal", noise_level = 1e-4):
    if A_type == "Gaussian":
        A = np.random.standard_normal((m,n))
    elif A_type == "DCT":
        A = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                A[i,j] = np.cos(2*np.pi*(j-1)*np.random.uniform(0,1))
        col_norms = np.linalg.norm(A, axis=0)
        A = A / col_norms
        
    if x_type == "discrete":
        x = np.zeros(n)
        for i in np.random.choice(n,s,replace=False):
            x[i] = 1 if np.random.uniform()>0.5 else -1
    elif x_type == "normal":
        x = np.zeros(n)
        x[np.random.choice(n,size = s,replace=False)] = np.random.standard_normal(s)
        
    if noise_type == "normal":
        noise = np.random.normal(0,noise_level,m) # * noise_level
        
    b = A@x + noise
    
    if loss_type == "convex":
        def obj(z):
            return 0.5*LA.norm(A@z - b)**2  
        def grad(z):
            return A.T@(A@z - b)
    elif loss_type == "nonconvex":        
        def obj(z):
            return np.sum(np.log(0.5*(A@z - b)**2 + 1))
        def grad(z):
            axb = A@z - b
            return A.T@(axb/(0.5*axb**2 + 1))
    
    return obj, grad, x, np.max(LA.eigvals(A.T@A).real),A,b
