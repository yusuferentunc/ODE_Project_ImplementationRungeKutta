import numpy as np
import matplotlib.pyplot as plt
from Problems import YeastSugar, Kepler, HeatEquation, HIRES, ZombiesOutbreak, HarmonicOscillator, SIRV, Brusselator_1D, Brusselator_2D
from Steppers.PopularRK import ExplicitEuler, ImplicitEuler, ExplicitMidpoint, ImplicitMidpoint, Heun3, ThetaMethod1, ThetaMethod2
from Steppers.ExplicitRK import DOPRI54
from Steppers.ImplicitRK import LobattoIIIC, RandomCollocation, Gauss, Radau
from TimeIntegrator import TimeIntegrator
from ReferenceSolution import reference_solution
from Exceptions import IncompatibleOptions, UnknownOption
import colorama


# Load a problem from Problems.py
P = ZombiesOutbreak()

# Time step adaptivity options
adaptivity_opts = {
    "enable_adaptivity": False,  # Set to True for adaptive time stepping
    "error_tolerance": 1e-4,  # Error tolerance for adaptive time stepping
    "dt_safe_fac": 0.9,  # Safety factor for time step adjustment
    "dt_facmax": 5.0,  # Maximum factor by which time step can increase
    "dt_facmin": 0.1,  # Minimum factor by which time step can decrease
}
# Nonlinear solver options
solver_opts = {
    "nonlinear_solver": "scipy",  # Choose the nonlinear solver: scipy or newton
    "quasi_newton": False,  # Use quasi-Newton method if available
    "linear_solver": "direct",  # Choose the linear solver: direct or iterative
    "matrix_free": False,  # Use matrix-free methods (only for iterative linear solver)
    "preconditioner": "none",  # Choose the preconditioner: none or ILU. ILU is possible only in non matrix-free mode and with iterative linear solver
}

# We try to define the stepper. Since many features are missing then it is possible that an exception will be raised
try:
    RK = LobattoIIIC(solver_opts=solver_opts, adaptivity_opts=adaptivity_opts)
    RK = RandomCollocation(s=3, solver_opts=solver_opts, adaptivity_opts=adaptivity_opts)
    RK = Gauss(s=2, solver_opts=solver_opts, adaptivity_opts=adaptivity_opts)
    # RK = Radau(s=2, solver_opts=solver_opts, adaptivity_opts=adaptivity_opts)
    # RK = DOPRI54(adaptivity_opts=adaptivity_opts)
except (NotImplementedError, IncompatibleOptions, UnknownOption) as err:
    print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)
    exit()

# Set how many time steps we want to consider in the efficiency experiment
# and the maximal one, which will be halved in each iteration
N_dt = 5
dt_max = 0.2

# No need to modify below this line ------------------------------------

# Define the list of time steps
dts = np.array([dt_max / 2**i for i in range(0, N_dt)])

# load the reference solution
yex = reference_solution(P, dts.min(), RK.order)

try:
    print("Performing convergence experiment...")
    err = np.zeros(N_dt)
    for i in range(N_dt):
        # Define the time integrator. It will solve P using RK.
        ti = TimeIntegrator(RK, P, adaptivity_opts)
        # Solve the problem using the defined time step (dt)
        t, y, et, n_steps, n_rejected_steps, avg_Newton_iter, avg_lin_solver_iter = ti.integrate(dts[i], verbose=False)
        err[i] = np.linalg.norm(y[:, -1] - yex)
except (NotImplementedError, IncompatibleOptions, UnknownOption) as err:
    print(colorama.Fore.RED + f"ERROR: {err}" + colorama.Style.RESET_ALL)
    exit()

# For latex text in matplotlib
plt.rcParams["text.usetex"] = False
fig, ax = plt.subplots()
# plot the error
ax.loglog(dts, err, label="Err", linestyle="dashed", color="black", marker="o")
# plot some convergence rates
for p in range(1, 6):
    ax.loglog(dts, err[0] / dts[0] ** p * dts**p, label=f"$O(\Delta t^{p})$")
ax.legend()
plt.show()
