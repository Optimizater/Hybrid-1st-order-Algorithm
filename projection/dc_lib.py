import warnings
warnings.filterwarnings('ignore')
import numpy as np
import time
from numpy.linalg import norm



def computing_x(v, u, y_i):
    """
    Computes the vector x for every y_i in the projection algorithm.
    Parameters:
    v, u: Input vectors for the computation
    y_i: A scalar value used in the computation
    Returns:
    A numpy array representing the computed x value
    """
    l1 = v - (u + 1) * y_i
    l2 = (u - 1) * y_i - v
    l1 = np.maximum(l1, 0)
    l2 = np.maximum(l2, 0)
    return l1 - l2


def projection(v, u, tau):
    """
    Projects the subproblem solution onto the constraint set.
    Parameters:
    v: Solution of k-th subproblem
    u, tau: Parameters for the l1 norm constraint
    Returns:
    The projected solution over the constraint
    """

    level = norm(v, ord=1) + u.dot(v)
    # if strict feasible, then return the point
    if level <= tau:
        return v

    # y1 = v / (1 + u)
    # y2 = v / (u - 1)

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
        y_opt = (result_left-tau)*y[0] / (result_0-result_left) + y[0]

    if result_right > tau:
        x_some = computing_x(v, u, y[-1]+1)
        result_some = np.dot(u, x_some) + np.linalg.norm(x_some, 1)
        y_opt = (tau-result_right)*(y[-1]+1-y[-1]) / (result_some-result_right) + y[-1]

    else:
        # binary search
        lo, hi = 0, len(y) - 1
        while lo <= hi:
            half = (lo + hi) // 2
            x = computing_x(v, u, y[half])
            result = np.dot(u, x) + np.linalg.norm(x, 1)
            if result > tau:
                x_tune = computing_x(v, u, y[half+1])
                result_tune = np.dot(u, x_tune) + np.linalg.norm(x_tune, 1)
                if result_tune < tau:
                    y_opt = (tau - result_tune) * (y[half] - y[half+1]) / (result - result_tune) + y[half+1]
                    break
                lo = half + 1
            elif result < tau:
                x_tune = computing_x(v, u, y[half-1])
                result_tune = np.dot(u, x_tune) + np.linalg.norm(x_tune, 1)
                if result_tune > tau:
                    y_opt = (tau - result) * (y[half-1] - y[half]) / (result_tune - result) + y[half]
                    break
                hi = half - 1
            else:
                y_opt = y[half]
                break
    return computing_x(v, u, y_opt)



def ACSA(grad,x0,u,gamma,tau,L0=1,gamma_u=2, gamma_d=2,line_search=True,maxiter=10,tol=1e-10):
    """
    Implements the ACSA algorithm for solving the subproblem of LCPP.
    Parameters:
    grad: Gradient of the function
    x0: Initial solution estimate
    u, gamma, tau: Algorithm parameters
    L0, gamma_u, gamma_d: Line search parameters
    line_search: Flag to enable/disable line search
    maxiter: Maximum number of iterations
    tol: Tolerance for convergence
    Returns:
    The computed solution and the last value of L
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
        v_k = projection((x0 - sum_grad / (gamma*Ak + 1)),u,tau)
        L = Lk

        while True:
            tem = (1 + gamma * Ak) / L
            a = tem + np.sqrt(tem ** 2 + 2 * Ak * tem)
            y = (Ak * xk + a * v_k) / (Ak + a)
            grad_y = grad(y)
            y_next = (gamma * x0 + L * y - grad_y) / (L + gamma)
            T_L = projection(y_next,u,tau)
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
    return xk,Lk

def LCPP(obj,nabla_f,x0,eta_max,lambda_,epsilon,theta,truncate_threshold = 0.3,Lf = 1,update_rule = "IRLS2",gamma=0.5,verbose=False,tol=1e-8,maxiter = int(1e6),maxtime = 100):
    """
    Solves the LCPP problem subject to a set of constraints.
    Parameters:
    obj (callable): Objective function to be minimized.
    grad (callable): Gradient of the objective function.
    x0 : Input vectors
    eta_max, lambda_, p, epsilon, theta: Parameters for the optimization problem
    truncate_threshold, update_rule: Parameters controlling the update mechanism
    Lf : Lipschitz constant of the gradient (optional).
    gamma: Parameter for the subproblem
    verbose: Flags for logging and verbose output
    tol: Tolerance for convergence
    maxiter, maxtime: Maximum iterations and time for the algorithm
    Returns:
    The solution vector, total time taken, and the number of iterations
    """
    def h(x,lambda_,epsilon,theta):
        return lambda_*norm(x,1) - norm(np.abs(x)+epsilon,1/theta)**(1/theta)
    def nabla_h(x,lambda_,epsilon,theta):
        return (lambda_) * np.sign(x) - 1/theta * np.sign(x) * (np.abs(x) + epsilon)**(1/theta-1)
    t0 = time.time()
    x = x0
    eta0 = lambda_ * norm(x0, ord=1) - h(x0,lambda_,epsilon,theta)
    eta = eta0
    L = Lf
    
    for t in range(maxiter):
        # show processing
        if time.time() - t0 > maxtime:
            break
        delta = (eta_max - eta0) / ((t + 1) * (t + 2))
        eta += delta

        # truncate
        if eta_max-eta<truncate_threshold:
            eta = eta_max-1e-15
            epsilon = 1e-35
        
        tau = (eta + h(x,lambda_,epsilon,theta) - nabla_h(x,lambda_,epsilon,theta).T @ x)/lambda_
        u = -nabla_h(x,lambda_,epsilon,theta)/lambda_
        '''
        Subproblem 
        min phi(x) + gamma/2*||x-x^{k-1}||^2_2
        s.t. g_k(x) <= eta_k
        '''
        x_sub,L = ACSA(nabla_f,x,u,gamma,tau,L0 = L,line_search=False) 

        rel = norm(x_sub - x) / (1 + norm(x))  
        # epsilon updating rulesW
        if update_rule == "IRLS1":              
            if norm(x-x_sub)<np.sqrt(epsilon)/100:
                epsilon*= 0.1
                lambda_*= 0.1**(1/theta-1)
        elif update_rule == "IRLS2": 
            if norm(x-x_sub)<np.sqrt(epsilon):
                epsilon*= 0.1
                lambda_*= 0.1**(1/theta-1)
        elif update_rule == "naive":
            if (t+1) % 10 == 0:
                epsilon*= 0.1
                lambda_*= 0.1**(1/theta-1)
        x = x_sub        
        
        if verbose:
            print(f"Iteration: {t:6}, Objective: {obj(x):.4e}, Time: {time.time() - t0:.2f}")

        if rel < tol:
            break
    return x,time.time() - t0,t


if __name__ == "__main__":
    np.random.seed(2023)
    
    # Projection
    p = 0.9
    y = np.random.normal(0.,1.,int(1e5))
    radius = 0.01 * norm(y, p) ** p
    x_ini = 0.3 *  (radius ** (1/p)) * (np.abs(y,dtype=np.float64) / norm(y,p))
    epsilon = 0.8*((radius - norm(x_ini,p)**p)/x_ini.shape[0])**(1/p)
    theta = 1/p
    lambda_ = (epsilon**(1/theta-1)/theta)
    x_DC, t_DC, iter_DC = LCPP(lambda x: 0.5*norm(x-y)**2,lambda x:x-y,x_ini.copy(),radius,lambda_,epsilon,theta,verbose=True)
