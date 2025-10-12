import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid

# ----------------------------------------------------------------------
# Example ODE:  y'' + 2y' + y = 2x
# Initial conditions: y(0) = 0 , y'(0) = 0
# ----------------------------------------------------------------------

# === Homogeneous ODE: y'' + 2y' + y = 0 ===============================
def ode_system_homogeneous(x, y):
    y0, y1 = y
    dydx0 = y1
    dydx1 = -2*y1 - y0
    return [dydx0, dydx1]

# === Non-Homogeneous ODE: y'' + 2y' + y = 2x ==========================
def ode_system_nonhomogeneous(x, y):
    y0, y1 = y
    dydx0 = y1
    dydx1 = 2*x - 2*y1 - y0
    return [dydx0, dydx1]

# ===============================================================
x_span = (0, 10)
y0 = [0, 0]
x_points = np.linspace(*x_span, 400)

# =========================================================================

sol_hom = solve_ivp(ode_system_homogeneous, x_span, y0, t_eval=x_points, method='RK45')
sol_nonhom = solve_ivp(ode_system_nonhomogeneous, x_span, y0, t_eval=x_points, method='RK45')

#Non-Homogeneous Equation

def greens_function(x, s):
    return np.where(x >= s, (x - s) * np.exp(-(x - s)), 0.0)

def forcing(s):
    return 2 * s  # f(s) = 2x (same as RHS)

greens_solution = []
for xi in x_points:
    integrand = greens_function(xi, x_points) * forcing(x_points)
    greens_solution.append(trapezoid(integrand, x_points))
greens_solution = np.array(greens_solution)

# graph

# --- Plot 1: Homogeneous Solution ---
plt.figure(figsize=(8,4))
plt.plot(sol_hom.t, sol_hom.y[0], label="Homogeneous Solution (y'' + 2y' + y = 0)")
plt.title("Homogeneous ODE Solution")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: Non-Homogeneous vs Green’s Function ---
plt.figure(figsize=(9,5))
plt.plot(sol_nonhom.t, sol_nonhom.y[0], label="Numerical Solution (RKF45)", color='tab:blue')
plt.plot(x_points, greens_solution, '--', label="Green’s Function Solution", color='tab:orange')
plt.title("Comparison of Green’s Function and Numerical Solutions")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

