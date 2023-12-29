import numpy as np
import scipy.optimize
import scipy.integrate
from Exceptions import IncompatibleOptions, UnknownOption


# Helper class to count the number of iterative solver iterations
class gmres_counter:
    def __init__(self):
        # Store the number of iterations
        self.niter = 0

    def __call__(self, rk=None):
        # Increment the number of iterations at each call
        self.niter += 1


class ImplicitRK:
    """
    This class implements a general implicit Runge-Kutta method
    It handles several options for the nonlinear solver and the linear solver
    In particular:
    - Nonlinear solver: scipy (fsolve) or Newton
    - Linear solver: direct or iterative (GMRES)
    - Preconditioner: none or ILU
    - Matrix free: only for iterative solver

    After each step, the error is estimated in child classes.
    """

    def __init__(self, A, b, c, solver_opts, adaptivity_opts):
        """
        Initialize the Runge-Kutta method

        Args:
            A (np.array or list of lists): Runge-Kutta matrix in (s,s) shape array or list of lists
            b (np.array or list): Runge-Kutta weights b in shape (s,) array or list
            c (np.array or list): Runge-Kutta nodes c in shape (s,) array or list
            solver_opts (dict): dictionary containing the options for the nonlinear solver
            adaptivity_opts (dict): dictionary containing the options for time step adaptivity

        Raises:
            UnknownOption: When an unknown option is passed
            IncompatibleOptions: When incompatible options are passed
        """

        self.A = np.array(A)
        self.b = np.array(b)
        self.c = np.array(c)
        self.s = self.b.shape[0]

        # Compute the inverse of A, used to compute y1 from z
        self.invA = scipy.linalg.inv(self.A)

        # Set the nonlinear solver
        self.solver_opts = solver_opts
        if solver_opts["nonlinear_solver"] == "scipy":
            self.RK_solve = self.scipy_solver
        elif solver_opts["nonlinear_solver"] == "newton":
            self.RK_solve = self.Newton
            if solver_opts["linear_solver"] not in ["direct", "iterative"]:
                raise UnknownOption(f"Unknown linear_solver {solver_opts['linear_solver']}")
            if solver_opts["linear_solver"] == "direct" and solver_opts["matrix_free"]:
                raise IncompatibleOptions("direct linear solver", "matrix free")
        else:
            raise UnknownOption(f"Unknown nonlinear_solver {solver_opts['nonlinear_solver']}")

        # Set the quasi-Newton option
        self.quasi_newton = solver_opts["quasi_newton"]

        # Set the adaptivity options
        self.enable_adaptivity = adaptivity_opts["enable_adaptivity"]
        self.err_tol = adaptivity_opts["error_tolerance"]

        # Give a name to the integrator as name+order+options
        self.description = f"{self.__class__.__name__}{self.order} "
        # Add the options
        # Q for quasi-Newton
        # SP for scipy nonlinear solver
        # N for Newton nonlinear solver
        # D for direct linear solver
        # I for iterative  linear solver
        # MF for matrix free approach
        # ILU for ILU preconditioner (only for iterative linear solver)
        if self.quasi_newton:
            self.description = self.description + "Q"
        if solver_opts["nonlinear_solver"] == "scipy":
            self.description = self.description + "SP"
        else:
            self.description = self.description + "N"
            if solver_opts["linear_solver"] == "direct":
                self.description = self.description + " D"
            else:
                self.description = self.description + " I"
                if solver_opts["matrix_free"]:
                    self.description = self.description + " MF"
            if solver_opts["preconditioner"] != "none":
                self.description = self.description + " " + solver_opts["preconditioner"]

    def set_nonlinear_solver_parameters(self, dt):
        """
        Set the parameters of the nonlinear solver (Newton) and the linear solver (GMRES)
        As well as the finite difference step (used in the matrix free approach)

        Args:
            dt (float): the time step size
        """

        # Newton parameters
        if self.enable_adaptivity:
            # If we use time step adaptivitiy, Newton algorithm tolerance is the same as the one of time step adaptivity.
            self.Newton_tol = self.err_tol
        else:
            # Newton algorithm tolerance is proportional to the RK local error, but not overly small neither.
            self.Newton_tol = np.max([1e-2 * dt ** (self.order + 1), 1e-12])
        # Maximal number of iterations in the Newton algorithm
        self.Newton_max_iter = 100
        # The iterative linear solver error tolerance must be a bit smaller than Newton, since it is called a few times and errors will accumulate.
        # We set it to 1e-2 of the Newton tolerance. However, it depends on the problem and the method. You might want to change it to increase efficiency or accuracy.
        self.iter_solver_tol = 1e-2 * self.Newton_tol
        # Finite Difference step (Jacobian approximation when using the matrix-free approach)
        # Quite a delicate choice this one. If too large ==> bad approximation, if too small ==> large round off errors.
        # We set it to 1e-3. However, it depends on the problem and the method. You might want to change it to increase efficiency or accuracy.
        self.FD_eps = 1e-3

    def step(self, t0, y0, f, dt):
        """
        Performs one step of the Runge-Kutta method
        This function is called by the TimeIntegrator class and solves the
        ds x ds nonlinear system arising at each step of an implicit Runge-Kutta method.

        After step evaluation, the error is estimated in child classes.

        Args:
            t0 (float): the initial time
            y0 (np.array): the initial value
            f (function): the right-hand side of the ODE
            dt (float): the time step size

        Returns:
            y1 (np.array): the solution at time t0+dt
            err (float): the estimated error
            n_Newtion_iter (int): number of Newton iterations
            n_lin_solver_iter (int): number of linear solver iterations (over all Newton iterations)
        """

        # Fix the nonlinear solver paramters. It would be better to do so in the constructor
        # but there dt in unknown. So we do it here, but only once.
        if not hasattr(self, "Newton_tol"):
            self.set_nonlinear_solver_parameters(dt)
            self.n_vars = y0.size  # another useful variable

        # Solve the RK system Phi(z)=0 and return some statistics
        z, n_Newtion_iter, n_lin_solver_iter = self.RK_solve(t0, y0, f, dt)

        # Compute y1 given z
        y1 = self.get_y1(z, t0, y0, f, dt)

        # Estimate the error
        if self.enable_adaptivity:
            err = self.estimate_error(z, t0, y0, f, dt)
        else:
            err = self.err_tol

        return y1, err, n_Newtion_iter, n_lin_solver_iter

    def get_y1(self, z, t0, y0, f, dt):
        """
        Args:
            z (np.array): Array of shape (s*n_vars) containing the solution of the RK system Phi(z)=0
            t0 (float): initial time
            y0 (np.array): initial value, an array of shape (n_vars,)
            f (function): right-hand side of the ODE
            dt (float): time step size

        Returns:
            y1 (np.array): the solution at time t0+dt
        """
        # Method 1
        # Simplistic implementation, where we compute the stages k1,...,ks from z and then y1 from k1,...,ks
        #return y0 + dt * np.kron(self.b, np.identity(self.n_vars)) @ self.fz(z, t0, y0, f, dt)

        # Method 2
        # Compute y1 from z by using the inverse of A
        # This is more efficient than method 1, since we do not need to compute the stages k1,...,ks
        # Compute the matrix z-->F(z) only once!
        A_inv = np.linalg.inv(self.A)
        return y0 + np.kron(self.b, np.eye(self.n_vars)) @ np.kron(A_inv, np.eye(self.n_vars)) @ z

    def estimate_error(self, z, t0, y0, f, dt):
        """
        Estimate the error of the RK method at time t0+dt
        This method is overridden in child classes whenever they do have an error estimator

        Args:
            z (np.array): Array of shape (s*n_vars) containing the solution of the RK system Phi(z)=0
            t0 (float): initial time
            y0 (np.array): initial value, an array of shape (n_vars,)
            f (function): right-hand side of the ODE
            dt (float): time step size

        Raises:
            FeatureNotImplemented: When the error estimator is not implemented for the method

        """

        # TODO ImplicitRK Error estimator - page:35
        raise NotImplementedError("Error estimator not implemented for " + self.description)

    def scipy_solver(self, t0, y0, f, dt):
        """
        Solve the RK system Phi(z)=0 using a black box method from scipy (scipy.optimize.fsolve)

        Args:
            t0 (float): initial time
            y0 (np.array): initial value, an array of shape (n_vars,)
            f (function): right-hand side of the ODE
            dt (float): time step size

        Returns:
            z (np.array): Array of shape (s*n_vars) containing the solution of the RK system Phi(z)=0
            n_Newtion_iter (int): number of function evaluations, which is also the number of Newton iterations
            n_lin_solver_iter (int): number of linear solver iterations (over all Newton iterations). Here it is always 0, since we do not use a linear solver.
        """

        if not self.quasi_newton:
            z, info, ier, msg = scipy.optimize.fsolve(lambda z: self.Phi(z, t0, y0, f, dt),
                                                      x0=np.zeros(self.n_vars * self.s), xtol=self.Newton_tol,
                                                      maxfev=self.Newton_max_iter, full_output=True)
        else:
            # We use a quasi-Newton method, where we compute the Jacobian only once and reuse it at each iteration
            dPhi = self.dPhi(np.zeros(self.n_vars * self.s), t0, y0, f, dt)
            z, info, ier, msg = scipy.optimize.fsolve(
                lambda z: self.Phi(z, t0, y0, f, dt), x0=np.zeros(self.n_vars * self.s), xtol=self.Newton_tol,
                maxfev=self.Newton_max_iter, fprime=lambda x: dPhi, full_output=True
            )

        return z, info["nfev"], 0

    def Newton(self, t0, y0, f, dt):
        """
        Solve the RK system Phi(z)=0 using a Newton method

        Args:
            t0 (float): initial time
            y0 (np.array): initial value, an array of shape (n_vars,)
            f (function): right-hand side of the ODE
            dt (float): time step size

        Returns:
            z (np.array): Array of shape (s*n_vars) containing the solution of the RK system Phi(z)=0
            n_Newtion_iter (int): number of Newton iterations
            tot_lin_solver_iter (int): number of linear solver iterations (over all Newton iterations)
        """

        # Solve the RK system Phi(z)=0 using a Newton method
        # At each iteration we call a linear solver for Phi'(z)dz=-Phi(z) as:
        #       dz, dPhi_info, n_lin_solver_iter = self.call_linear_solver(z, t0, y0, f, dt, dPhi_info if self.quasi_newton else None)
        # The linear solver returns dz, dPhi_info, n_lin_solver_iter, where n_lin_solver_iter is the number of linear solver iterations.
        # dPhi_info is a tuple containing:
        # - first, the Jacobian matrix Phi'(z) or a matrix free approach to compute Phi'(z)dz
        # - second, the preconditioner M (if any).
        # dPhi_info is computed only once if self.quasi_newton is True.

        n_Newtion_iter = 0
        tot_lin_solver_iter = 0
        z = np.zeros(self.n_vars * self.s)

        if self.quasi_newton:
            dPhi_info = self.get_dPhi_info(self.Phi(z, t0, y0, f, dt), z, t0, y0, f, dt)

        for i in range(self.Newton_max_iter):
            dz, dPhi_info, n_lin_solver_iter = self.call_linear_solver(z, t0, y0, f, dt, dPhi_info if self.quasi_newton else None)
            n_Newtion_iter += 1
            tot_lin_solver_iter += n_lin_solver_iter
            if np.linalg.norm(dz) < np.linalg.norm(z) * self.Newton_tol:
                z = z + dz
                break
            z = z + dz

        return z, n_Newtion_iter, tot_lin_solver_iter

    def call_linear_solver(self, z, t0, y0, f, dt, dPhi_info=None):
        """
        Call the linear solver to solve Phi'(z)dz=-Phi(z)
        It can be either a direct solver or an iterative solver (GMRES or others)
        If it is an iterative solver, it can be matrix free or not
        If it is an iterative solver, it can have a preconditioner or not
        Before calling the linear solver, we compute Phi(z) and Phi'(z) (only once if self.quasi_newton is True)

        Args:
            z (np.array): Array of shape (s*n_vars)
            t0 (float): initial time
            y0 (np.array): initial value, an array of shape (n_vars,)
            f (function): right-hand side of the ODE
            dt (float): time step size
            dPhi_info (tuple, optional): Can be None, or a tuple containing:
                                         - first, the Jacobian matrix Phi'(z) (np.array) or a matrix free approach to compute Phi'(z)dz (scipy.sparse.linalg.LinearOperator)
                                         - second, the preconditioner M (if any) (scipy.sparse.linalg.LinearOperator) or None
                                         Defaults to None.

        Returns:
            dz (np.array): Array of shape (s*n_vars) containing the solution of the linear system Phi'(z)dz=-Phi(z)
            dPhi_info (tuple): The updated dPhi_info
            n_lin_solver_iter (int): number of linear solver iterations
        """
        # Compute Phi(z)
        Phiz = self.Phi(z, t0, y0, f, dt)
        # If dPhi_info is None, compute Phi'(z) and the preconditioner M (if any)
        if dPhi_info is None:
            dPhi_info = self.get_dPhi_info(Phiz, z, t0, y0, f, dt)

        # Solve Phi'(z)dz=-Phi(z)
        if self.solver_opts["linear_solver"] == "direct":
            # Direct solver
            dPhiz, _ = dPhi_info
            lu, piv = scipy.linalg.lu_factor(dPhiz)
            dz = scipy.linalg.lu_solve((lu, piv), -Phiz)
            n_lin_solver_iter = 0
        else:
            # Iterative solver
            # Set the callback to count the number of iterations, use it as callback=gmres_count in the call to gmres
            gmres_count = gmres_counter()
            # Extract Jacobian and preconditioner
            dPhiz, M = dPhi_info
            # Call the iterative solver (e.g., GMRES)
            dz, info = scipy.sparse.linalg.gmres(A=dPhiz, b=-Phiz, M=M, callback=gmres_count)
            # get the number of iterations from the callback
            n_lin_solver_iter = gmres_count.niter

        return dz, dPhi_info, n_lin_solver_iter

    def get_dPhi_info(self, Phiz, z, t0, y0, f, dt):
        """

        Args:
            Phiz (np.array): Array of shape (s*n_vars) containing Phi(z)
            z (np.array): Array of shape (s*n_vars)
            t0 (float): initial time
            y0 (np.array): initial value, an array of shape (n_vars,)
            f (function): right-hand side of the ODE
            dt (float): time step size

        Returns:
            dPhi_info (tuple): A tuple containing:
                                 - first, the Jacobian matrix Phi'(z) (np.array) or a matrix free approach to compute Phi'(z)dz (scipy.sparse.linalg.LinearOperator)
                                 - second, the preconditioner M (if any) (scipy.sparse.linalg.LinearOperator) or None
        """
        # Important: we need to copy z and Phiz, otherwise in quasi-Newton with matrix free approach the Jacobian will be wrong
        z_copy = z.copy()
        Phiz_copy = Phiz.copy()
        if self.solver_opts["matrix_free"]:
            # define dPhi as a LinearOperator, which is a matrix free approach to compute Phi'(z)dz
            # Use the scipy.sparse.linalg.LinearOperator class and define the matvec function directly from the constructor
            # Remember that the matvec function takes as input dz and returns Phi'(z)dz.
            # Remember that you can use the function self.dPhi_FD to approximate Phi'(z)dz using finite differences
            # Give Phiz_copy as an argument to self.dPhi_FD to save computations
            # Remember to use z_copy and Phiz_copy instead of z and Phiz, otherwise the Jacobian will be wrong in quasi-Newton
            # TODO Matrix Free Approach
            #raise NotImplementedError("Matrix Free Approach not implemented")
            dPhi = scipy.sparse.linalg.LinearOperator((self.n_vars * self.s, self.n_vars * self.s),
                                       matvec=lambda dz: self.dPhi_FD(dz, z_copy, t0, y0, f, dt, Phiz_copy))
        else:
            # Compute the Jacobian matrix Phi'(z) exactly
            dPhi = self.dPhi(z_copy, t0, y0, f, dt)

        # Compute the preconditioner M (if any)
        dPhi_prec = self.get_dPhi_preconditioner(dPhi)

        return (dPhi, dPhi_prec)

    def get_dPhi_preconditioner(self, dPhi):
        """
        Compute the preconditioner M (if any) for the linear system Phi'(z)dz=-Phi(z)

        Args:
            dPhi (np.array or scipy.sparse.linalg.LinearOperator): Jacobian matrix Phi'(z) (np.array) or a matrix free approach to compute Phi'(z)dz (scipy.sparse.linalg.LinearOperator)

        Returns:
            M (scipy.sparse.linalg.LinearOperator or None): A LinearOperator that defines the preconditioner M or None if no preconditioner is used
        """
        if self.solver_opts["linear_solver"] == "direct" or self.solver_opts["preconditioner"] == "none":
            return None
        elif self.solver_opts["preconditioner"] == "ILU":
            if self.solver_opts["matrix_free"]:
                raise IncompatibleOptions("ILU preconditioner", "matrix-free.")
            # Compute the ILU factorization for dPhi and use it as a preconditioner
            ilu = scipy.sparse.linalg.spilu(dPhi)
            # Define the preconditioner M as a LinearOperator
            M = scipy.sparse.linalg.LinearOperator(shape=dPhi.shape, matvec=ilu.solve)
            return M
        else:
            raise UnknownOption(f"preconditioner={self.solver_opts['preconditioner']} is not implemented.")

    def G(self, z, t0, y0, f, dt):
        """
        Compute the function G(z)= dt*(A kron I) F(z) where F(z) is the vector of stages F(z)=(f(y0+z1),...,f(y0+zs))

        Args:
            z (np.array): Array of shape (s*n_vars)
            t0 (float): initial time
            y0 (np.array): initial value, an array of shape (n_vars,)
            f (function): right-hand side of the ODE
            dt (float): time step size

        Returns:
            Gz (np.array): Array of shape (s*n_vars) containing G(z)
        """
        # Use the function self.fz to compute F(z)
        fz_eval = self.fz(z, t0, y0, f, dt)
        Gz = dt * np.kron(self.A, np.eye(self.n_vars)) @ fz_eval

        return Gz

    def Phi(self, z, t0, y0, f, dt):
        """
        Compute the function Phi(z)=z-G(z)

        Args:
            z (np.array): Array of shape (s*n_vars)
            t0 (float): initial time
            y0 (np.array): initial value, an array of shape (n_vars,)
            f (function): right-hand side of the ODE
            dt (float): time step size

        Returns:
            Phiz (np.array): Array of shape (s*n_vars) containing Phi(z)
        """

        return z - self.G(z, t0, y0, f, dt)

    def dPhi(self, z, t0, y0, f, dt):
        """
        Compute the Jacobian matrix dPhi(z)
        Args:
            z (np.array): Array of shape (s*n_vars)
            t0 (float): initial time
            y0 (np.array): initial value, an array of shape (n_vars,)
            f (function): right-hand side of the ODE
            dt (float): time step size
            method (int, optional): There are two ways of implementing the Jacobian approximation. Defaults to 2.

        Returns:
            dPhi (np.array): Jacobian matrix dPhi(z) of shape (s*n_vars, s*n_vars)
        """
        # Compute the Jacobian matrix dPhi(z) using scipy.optimize.approx_fprime
        # There are two ways of doing it:
        # - method=1: naive implementation, where Phi is seen as a black box passed to scipy.optimize.approx_fprime
        # - method=2: exploit the special structure of Phi(z)=I-dt*kron(A,I)@(f(y0+z0),...,f(y0+zs)), hence compute the Jacobians of f at y0+z1,...,y0+zs
        #             and then evaluate the jacobian of Phi
        # Choose what you prefer, but one of the two is more efficient than the other. Which one?
        dPhi = scipy.optimize.approx_fprime(z, self.Phi,self.FD_eps, t0, y0, f, dt)
        return dPhi

    def dPhi_FD(self, dz, z, t0, y0, f, dt, Phiz=None):
        """
        Approximate the matrix-vector product dPhi(z)dz using finite differences
        Args:
            dz (np.array): Array of shape (s*n_vars)
            z (np.array): Array of shape (s*n_vars)
            t0 (float): initial time
            y0 (np.array): initial value, an array of shape (n_vars,)
            f (function): right-hand side of the ODE
            dt (float): time step size
            Phiz (np.array, optional): Array of shape (s*n_vars) containing Phi(z), it is useful to save computations. Defaults to None.

        Returns:
            dPhi_dz (np.array): Array of shape (s*n_vars) containing the approximation of dPhi(z)dz
        """

        norm_dz = np.linalg.norm(dz)
        if norm_dz > 0:
            # Compute Phi(z) if not given
            if Phiz is None:
                Phiz = self.Phi(z, t0, y0, f, dt)
            # We chose eps such that eps*dz is about FD_eps times smaller than y0+z
            norm_yz = np.linalg.norm(y0 + z[: self.n_vars])
            eps = self.FD_eps * (1.0 + norm_yz) / norm_dz
            # Calculate phiz * v
            Phiz_next = self.Phi(z + eps * dz, t0, y0, f, dt)
            # Approximate Jacobian of PhiZ
            return (Phiz_next - Phiz) / eps
        else:  # if dz==0 then return 0 (dPhi*0=0)
            return np.zeros_like(dz)

    def fz(self, z, t0, y0, f, dt):
        """
        Compute the vector of stages F(z)=(f(t0+c_1*dt,y0+z1),...,f(t0+c_s*dt,y0+zs))
        Args:
            z (np.array): Array of shape (s*n_vars)
            t0 (float): initial time
            y0 (np.array): initial value, an array of shape (n_vars,)
            f (function): right-hand side of the ODE
            dt (float): time step size

        Returns:
            fz (np.array): Array of shape (s*n_vars) containing F(z)=(f(t0+c_1*dt,y0+z1),...,f(t0+c_s*dt,y0+zs))
        """
        n_vars = len(y0)
        s = self.s
        c = self.c
        fz = np.zeros_like(z)
        for i in range(0, s):
            index_start = i * n_vars
            index_end = (i + 1) * n_vars
            y = y0 + z[index_start:index_end]
            t = t0 + c[i] * dt
            fz[index_start:index_end] = f(t, y)
        return fz


class LobattoIIIC(ImplicitRK):
    """
    Lobatto IIIC method of order 4
    """

    def __init__(self, solver_opts, adaptivity_opts):
        """
        Initialize the Runge-Kutta method
        Args:
            solver_opts (dict): Dictionary containing the options for the nonlinear solver
            adaptivity_opts (dict): Dictionary containing the options for time step adaptivity, not used here.
        """
        A = [[1 / 6, -1 / 3, 1 / 6], [1 / 6, 5 / 12, -1 / 12], [1 / 6, 2 / 3, 1 / 6]]
        c = [0, 1 / 2, 1]
        b = [1 / 6, 2 / 3, 1 / 6]
        self.order = 4

        super().__init__(A, b, c, solver_opts, adaptivity_opts)


class Collocation(ImplicitRK):
    """
    Collocation method
    """

    def __init__(self, c, solver_opts, adaptivity_opts):
        """
        Initialize the Collocation method. Given c we compute A and b, then we call the ImplicitRK constructor for the rest.
        Args:
            c (np.array or list): Runge-Kutta nodes c in shape (s,) array or list
            solver_opts (dict): Dictionary containing the options for the nonlinear solver
            adaptivity_opts (dict): Dictionary containing the options for time step adaptivity.
        """
        # Compute A and b from c
        A, b = self.compute_Ab(c)

        # Set the order of the method, if not already set in child classes
        if not hasattr(self, "order"):
            self.order = len(c)  # s

        # Call the ImplicitRK constructor
        super().__init__(A, b, c, solver_opts, adaptivity_opts)

    def Lagrange(self, x, c, j):
        """
        Evaluates the Lagrange polynomial L_j(x) at x

        Args:
            x (np.array): Array of shape (n,) containing the points where to evaluate the Lagrange polynomial
            c (np.array): Array of shape (s,) containing the nodes of the collocation method
            j (int): Index of the Lagrange polynomial

        Returns:
            Lj (np.array): Array of shape (n,) containing the values of L_j(x)
        """
        s = len(c)
        Lj = np.ones_like(x)
        for i in range(s):
            if i != j:
                Lj *= (x - c[i]) / (c[j] - c[i])
        return Lj

    def compute_Ab(self, c):
        """
        Compute the Runge-Kutta matrix A and the weights b from the nodes c of the collocation method
        Args:
            c (np.array): Array of shape (s,) containing the nodes of the collocation method

        Returns:
            A (np.array): Array of shape (s,s) containing the Runge-Kutta matrix
            b (np.array): Array of shape (s,) containing the weights
        """
        s = len(c)
        A = np.zeros((s, s))
        b = np.zeros(s)
        for i in range(0, s):
            b[i], _ = scipy.integrate.quad(lambda x: self.Lagrange(x, c, i), 0, 1)
            for j in range(0, s):
                A[i, j], _ = scipy.integrate.quad(lambda x: self.Lagrange(x, c, j), 0, c[i])
        return A, b


class RandomCollocation(Collocation):
    """
    Random Collocation method, here the nodes c are chosen randomly.
    The goal if to prove that the order of the methodis always at least s
    """

    def __init__(self, s, solver_opts, adaptivity_opts):
        """
        Initialize the Random Collocation method. Given s we compute c, then we call the Collocation constructor for the rest.
        Args:
            s (int): Number of nodes
            solver_opts (dict): Dictionary containing the options for the nonlinear solver
            adaptivity_opts (dict): Dictionary containing the options for time step adaptivity.
        """
        # Sample s nodes in [0,1] and sort them in increasing order
        c = np.sort(np.random.rand(s))

        # We know from theory that the order of the method is at least s
        self.order = s

        # Call the Collocation constructor
        super().__init__(c, solver_opts, adaptivity_opts)


class Gauss(Collocation):
    """
    Gauss collocation method of order 2s
    """

    def __init__(self, s, solver_opts, adaptivity_opts):
        """
        Initialize the Gauss Collocation method. Given s we compute c, then we call the Collocation constructor for the rest.

        Args:
            s (int): Number of nodes
            solver_opts (dict): Dictionary containing the options for the nonlinear solver
            adaptivity_opts (dict): Dictionary containing the options for time step adaptivity.
        """
        # Compute the nodes c
        c = self.compute_c(s)
        # Set the order of the method
        self.order = 2 * s
        # Call the Collocation constructor
        super().__init__(c, solver_opts, adaptivity_opts)

    def compute_c(self, s):
        """
        Compute the nodes c of the Gauss collocation method of order 2s
        Args:
            s (int): Number of nodes

        Returns:
            c (np.array): Array of shape (s,) containing the nodes of the Gauss collocation method
        """
        # Hint: use np.polynomial.polynomial.Polynomial to define the polynomial x.
        # Then use the power **, mult *, etc to define the polynomial x**s*(1-x)**s.
        # Then use the np.polynomial.polynomial.Polynomial class methods to differentiate it and compute the roots.

        # Use np.polynomial.polynomial.Polynomial to define the polynomial x
        x = np.polynomial.polynomial.Polynomial([0, 1])
        p_s = x ** s * (1 - x) ** s
        p_s_diff = p_s.deriv(m=s)
        roots = np.polynomial.polynomial.Polynomial.roots(p_s_diff)
        c = np.real(roots)
        c.sort()
        return c


class Radau(Collocation):
    """
    Radau collocation method of order 2s-1
    """

    def __init__(self, s, solver_opts, adaptivity_opts):
        """
        Initialize the Radau Collocation method. Given s we compute c, then we call the Collocation constructor for the rest.
        Args:
            s (int): Number of nodes
            solver_opts (dict): Dictionary containing the options for the nonlinear solver
            adaptivity_opts (dict): Dictionary containing the options for time step adaptivity.
        """
        # Compute the nodes c
        c = self.compute_c(s)

        # Set the order of the method
        self.order = 2 * s - 1

        # Call the Collocation constructor
        super().__init__(c, solver_opts, adaptivity_opts)

    def compute_c(self, s):
        """
        Compute the nodes c of the Radau collocation method
        Args:
            s (int): Number of nodes

        Returns:
            c (np.array): Array of shape (s,) containing the nodes of the Radau collocation method
        """
        x = np.polynomial.polynomial.Polynomial([0, 1])
        p_s = x ** (s - 1) * (1 - x) ** s
        p_s_diff = p_s.deriv(m=s - 1)
        roots = np.polynomial.polynomial.Polynomial.roots(p_s_diff)
        c = np.real(roots)
        c.sort()
        return c

    def get_y1(self, z, t0, y0, f, dt):
        n_vars = len(y0)
        s = z.shape[0] / n_vars
        index_start = int((s - 1) * n_vars)
        return y0 + z[index_start:]

    def estimate_error(self, z, t0, y0, f, dt):
        """
        Estimate the error of the Radau method at time t0+dt.
        We do so only for s=3, otherwise we call the parent class method.

        Args:
            z (np.array): Array of shape (s*n_vars) containing the solution of the RK system Phi(z)=0
            t0 (float): initial time
            y0 (np.array): initial value, an array of shape (n_vars,)
            f (function): right-hand side of the ODE
            dt (float): time step size

        Returns:
            err (float): the estimated error
        """
        if self.s == 3:
            self.phat = 3
            # Remember to divide the error by sqrt(n_vars) to make it independent of the number of variables
            # (have a look at the DOPRI54 error estimator)
            # This is particularly important when solving PDEs, where n_vars is large
            # and can change if we change the mesh size.
            # Compute diff = ^y_n+1 - y_n+1
            b_hat_0 = 0.274888829595677
            e = np.zeros(3)
            e[0] = -b_hat_0 * (13 + 7 * np.sqrt(6)) / 3
            e[1] = -b_hat_0 * (13 - 7 * np.sqrt(6)) / 3
            e[2] = -b_hat_0 / 3
            diff = np.abs( b_hat_0 * dt * f(t0, y0) + sum((e[i] * z[i] for i in range(self.s))) )
            # Estimate the error. Divide by sqrt(diff.size) to make it independent of the number of variables
            err = np.linalg.norm(diff) / np.sqrt(diff.size)
            return err

        else:
            super().estimate_error(z, t0, y0, f, dt)
