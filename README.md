# Hybrid 1st-order Algorithm for $\ell_p$ ball Minimization Problem
This Python repo is for a hybrid Frank-Wolfe method introduced in "Minimization Over the Nonconvex Sparsity Constraint Using A Hybrid First-order method with Guaranteed Feasibility"

## The Problem
We focus on the $\ell_p$ ball-constrained optimization problem, specifically
$f:\mathbb{R}^d \rightarrow \mathbb{R}$ via $\ell_p$ ball-constraint $\mathcal{B}_{\ell_p}:=  \lbrace x\in \mathbb{R}^n \mid 
||x||_p^p \leq \gamma \rbrace$.

$$
\min_{x\in \mathbb{R}^d} \qquad f(x) \qquad \qquad
\text{s. t. }  \qquad x  \in  \mathcal{B}_{\ell_p}
$$
## Usage
```
import fw_lib
solver = fw_lib.FW_LP(p, radius, obj, grad, Lf, maxiter)
x, time, iterations = solver.solve(x_ini, mu, stopping_tol, tol, verbose, maxtime)
```

## Input Description
For fw_lib.FW_LP function
1. p (float): The exponent 'p' in the lp-ball constraint.
2. radius (float): Radius of the lp-ball.
3. obj (callable): Objective function to be minimized.
4. grad (callable): Gradient of the objective function.
5. Lf (float): Lipschitz constant of the gradient.
6. maxiter (int): Maximum number of iterations for the solver with default setting to 5e4.
    
For solve function
1. x_ini (np.array): Initial point  for the optimization problem.
2. mu (float): Step size for projected gradient descent.
3. stopping_tol (float): Stopping tolerance for the optimization.
4. tol (float): Tolerance used for numerical comparisons.
5. verbose (bool): Enables detailed logging.
6. maxtime (int): Maximum time allowed for the optimization process.
## Output Description
1. x : estimated optimum point 
2. time :  duration of optimization
3. iterations : total iterations
## Demo
Run `fw_lib.py` for demo. `numpy` and `SparseSignalRecovery.least_square` are required
```
python3 fw_lib.py
```
Therein, we include the Euclidean Projection  problem and the Sparse Signal Recovery problem for our method. More details about the test problems can be found in the paper. 
