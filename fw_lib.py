import numpy as np
from numpy import linalg as LA
from timeit import default_timer as timer
import warnings
warnings.filterwarnings('ignore')

class FW_LP:
    """
    Solves the Lp ball constrained minimization problem:
    min_x f(x) subject to ||x||_p^p <= r, where 0 < p < 1.

    Attributes:
        p (float): The exponent 'p' in the lp-ball constraint.
        radius (float): Radius of the lp-ball.
        obj (callable): Objective function to be minimized.
        grad (callable): Gradient of the objective function.
        Lf (float): Lipschitz constant of the gradient (optional).
        maxiter (int): Maximum number of iterations for the solver.
        EPS (float): Small epsilon value for numerical stability.
    """

    def __init__(self, p, radius, obj, grad, Lf=None, maxiter=int(5e4)):
        assert 0 < p < 1 and radius > 0, "Invalid parameters: 0 < p < 1 and radius > 0 required."
        self.p = p
        self.radius = radius
        self.obj = obj
        self.grad = grad
        self.Lf = Lf 
        self.maxiter = maxiter
        self.EPS = np.finfo(np.float64).eps

    def search_stepsize(self, d, x, g, Lf, tau=1.1, eta=0.9):
        """
        Computes stepsize for the Frank-Wolfe subproblem using adaptive stepsize rule.

        Args:
            d (np.array): Direction of the step.
            x (np.array): Current iterate.
            g (float): Gradient at the current iterate.
            Lf (float): Current estimate of Lipschitz constant.
            tau (float): Increase factor for Lipschitz constant estimate.
            eta (float): Decrease factor for Lipschitz constant estimate.

        Returns:
            tuple: Calculated stepsize and updated Lipschitz constant.
        """
        M = g**2 / (2 * (self.objective_list[-2] - self.objective_list[-1] + self.EPS) * LA.norm(d)**2)
        M = np.clip(M, eta * Lf, Lf)
        alpha = min(g / (M * LA.norm(d)**2), 1)
        iter_count = 0
        while self.obj(x + alpha * d) > self.obj(x) - alpha * g + 0.5 * alpha**2 * M * LA.norm(d)**2:
            if iter_count > 100:
                break
            iter_count += 1
            M = tau * M
            alpha = min(g / (M * LA.norm(d)**2 + self.EPS), 1)
        if LA.norm(x + alpha * d, self.p)**self.p > self.radius:
            alpha = self.binary_search(0.0, alpha, d, x)
        return alpha, M

    def binary_search(self, alpha_l, alpha_r, desect_dir, x, tol=1e-12):
        """
        Bisection method to find a root of the equation over interval (0, alpha_bar).
        ||x^k + gamma * d||_p^p = radius.

        Args:
            alpha_l (float): Left endpoint of the interval.
            alpha_r (float): Right endpoint of the interval.
            desect_dir (np.array): Direction for the bisection search.
            x (np.array): Current iterate.
            tol (float): Tolerance for the bisection search.

        Returns:
            float: The root found within the interval.
        """
        alpha_bi = alpha_l + 0.5 * (alpha_r - alpha_l)
        res = LA.norm(x + alpha_bi * desect_dir, self.p) ** self.p - self.radius
        while abs(res) > tol:
            if res > 0:
                alpha_r = alpha_bi
            else:
                alpha_l = alpha_bi
            alpha_bi = alpha_l + 0.5 * (alpha_r - alpha_l)
            res = LA.norm(x + alpha_bi * desect_dir, self.p) ** self.p - self.radius
        return alpha_bi

    def base_weighted_sort(self,y,w,a):
        """
        Projects a point onto the weighted L1 ball.

        Args:
            y (np.array): Point to be projected.
            w (np.array): Weights for L1 ball.
            a (float): Radius of the L1 ball.

        Returns:
            np.array: Projected point.
        """
        signum = np.sign(y)
        y = y * signum
        d = len(y)
        z = y / w        
        z_perm = np.argsort(-z)
        z = z[z_perm]
        i = 0
        sumWY = w[z_perm[i]] * y[z_perm[i]]
        Ws = w[z_perm[i]] * w[z_perm[i]]
        tau = (sumWY - a) / Ws
        i += 1
        while ((i < d) and (z[i] > tau)):
            sumWY += w[z_perm[i]] * y[z_perm[i]]
            Ws += w[z_perm[i]] * w[z_perm[i]]
            tau = (sumWY - a) / Ws
            i += 1
        return np.maximum(y - w * tau, 0) * signum

    def weighted_l1ball_projection(self, mu, grad, x):
        """
        Project a vector onto the weighted l1 ball.

        Args:
            mu (float): Step size for the gradient descent step.
            grad (np.array): Gradient of the objective at the current iterate.
            x (np.array): Current iterate.

        Returns:
            np.array: Projected point.
        """
        # Step 1: Compute the point to be projected: u^k = x^k - mu * \nabla f(x^k)
        z = x - mu * grad  # Point to be projected

        # Step 2: Enforce sign consistency (implements sign extraction)
        # Mathematical: \tilde{u}_i = sign(x_i) * u_i
        # For nonnegative weighted l1 projection, if sign(x_i) \neq sign(u_i),
        # then \tilde{u}_i < 0, which would project to 0 on the nonnegative ball.
        # We pre-filter these to improve efficiency.
        
        z[x * z <= 0] = 0 

        # Step 3: Handle inactive components (near-zero in x^k)
        z[np.abs(x) <= self.EPS] = 0 

        # Step 4: Identify active indices I(x^k) = {i : |x_i^k| > EPS}
        active_indices = np.where(np.abs(x) > self.EPS)[0]

        # Step 5: Compute weights for the weighted l1 ball projection
        weights = np.zeros_like(x)
        weights[active_indices] = self.p * np.abs(x[active_indices] + self.EPS) ** (self.p - 1)

        # Step 6: Compute the radius for the weighted l1 ball
        radius_L1 = (self.radius - LA.norm(x[active_indices], self.p) ** self.p +
                     weights[active_indices].dot(np.abs(x[active_indices])))

        # Step 7: Initialize the projected point (inactive indices already zero)
        x_projected = np.zeros_like(z)

        # Step 8: Project onto nonnegative weighted l1 ball and map back
        x_projected[active_indices] = self.base_weighted_sort(z[active_indices], weights[active_indices], radius_L1)
        
        return x_projected

    def fw_subproblem(self, grad, x):
        """
        Solves the Frank-Wolfe subproblem.

        Args:
            grad (np.array): Gradient of the objective at the current iterate.
            x (np.array): Current iterate.

        Returns:
            tuple: Direction and the result of the Frank-Wolfe subproblem.
        """
        z = np.zeros_like(x)
        z[np.argmax(np.abs(grad))] = self.radius
        fw_solution = -np.power(z, 1 / self.p) * np.sign(grad)  # Solution of FW subproblem
        descent_direction = fw_solution - x
        fw_result = grad.dot(-descent_direction)
        return descent_direction, fw_result

    def solve(self, x_ini, mu=None, stopping_tol=1e-8, tol=1e-10, verbose=False, maxtime=200):
        """
        Solves the optimization problem using a hybrid approach of 
        Frank-Wolfe and projected gradient descent methods.

        Args:
            x_ini (np.array): Initial guess for the optimization problem.
            mu (float, optional): Step size for projected gradient descent.
            stopping_tol (float): Stopping tolerance for the optimization.
            tol (float): Tolerance used for numerical comparisons.
            verbose (bool): Enables detailed logging.
            maxtime (int): Maximum time allowed for the optimization process.

        Returns:
            tuple: Optimized variable, duration of optimization, total iterations.
        """
        x = np.copy(x_ini)
        self.objective_list = [self.obj(x), self.obj(x)]
        time_start = timer()

        for iteration in range(self.maxiter):
            if timer() - time_start > maxtime and verbose:
                print("Maximum time reached.")
                break

            grad = self.grad(x)  # Gradient computation
            if abs(self.radius - LA.norm(x, p) ** p) <= tol:
                # On the boundary, use projected gradient descent
                x_pre = x.copy()
                if mu is None:
                    beta = 2 / self.Lf
                    while self.obj(x - beta * grad) > self.obj(x) - beta / 2 * np.linalg.norm(grad) ** 2:
                        beta *= 0.9
                    x = self.weighted_l1ball_projection(beta, grad, x)
                    self.Lf = 1 / beta
                else:
                    x = self.weighted_l1ball_projection(mu, grad, x)
                projection_residual = LA.norm(x - x_pre, 2)
                if projection_residual < stopping_tol:
                    if verbose:
                        print("Projection residual reaches stopping tolerance.")
                    break
            elif LA.norm(x, p) ** p < radius:
                # Inside the Lp ball, use Frank-Wolfe method
                descent_direction, fw_result = self.fw_subproblem(grad, x)
                if fw_result < stopping_tol:
                    if verbose:
                        print("Frank-Wolfe residual below tolerance.")
                    break
                if self.Lf == None:
                    self.Lf = LA.norm(self.grad(x)-self.grad(x+1e-3*descent_direction))/LA.norm(1e-3*descent_direction)    
                alpha, self.Lf = self.search_stepsize(descent_direction, x, fw_result, self.Lf)
                x += alpha * descent_direction
            else:
                if verbose:
                    print("Iterative exceed constraint.")
                break
            self.objective_list.append(self.obj(x))
            if verbose:
                print(f"Iteration: {iteration:6}, Objective: {self.objective_list[-1]:.4e}, Time: {timer() - time_start:.2f}")

        time_end = timer()
        return x, time_end - time_start, iteration + 1

if __name__ == "__main__":
    np.random.seed(2023)
    
    # Projection
    p = 0.9
    y = np.random.normal(0.,1.,int(1e5))
    radius = 0.01 * LA.norm(y, p) ** p
    x_ini = 0.3 *  (radius ** (1/p)) * (np.abs(y,dtype=np.float64) / LA.norm(y,p))
    
    solver = FW_LP(p,radius,lambda x: 0.5*LA.norm(x-y)**2,lambda x:x-y, Lf = 1)
    x,_,_ = solver.solve(x_ini.copy(),mu=1,verbose=True)
    
    


