import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import traceback
import time

# ----------------------------------------------------------------------
# Example ODE: y'' + 2y' + y = 2x
# Initial conditions: y(0) = 0, y'(0) = 0
# ----------------------------------------------------------------------

# Homogeneous ODE System: y'' + 2y' + y = 0
def ode_system_homogeneous(x, y):
    y0, y1 = y
    dydx0 = y1
    dydx1 = -2 * y1 - y0
    return [dydx0, dydx1]

# Non-Homogeneous ODE System: y'' + 2y' + y = 2x
def ode_system_nonhomogeneous(x, y):
    y0, y1 = y
    dydx0 = y1
    dydx1 = 2 * x - 2 * y1 - y0
    return [dydx0, dydx1]

# Green's Function and forcing term
def greens_function(x, s):
    # Green's function is only active for x >= s
    return np.where(x >= s, (x - s) * np.exp(-(x - s)), 0.0)

def forcing(s):
    return 2 * s  # RHS f(s) = 2x

# Main execution function
def main():
    print("Solving ODEs...")
    x_span = (0, 10)
    y0 = [0, 0]  # Initial conditions: y(0) = 0, y'(0) = 0

    # Create x points and ensure there is an odd number of points for Simpson's Rule
    num_points = 401  # Must be odd
    x_points = np.linspace(x_span[0], x_span[1], num_points)

    # Solve ODEs using solve_ivp (RK45 method)
    sol_hom = solve_ivp(ode_system_homogeneous, x_span, y0, t_eval=x_points, method='RK45')
    sol_nonhom = solve_ivp(ode_system_nonhomogeneous, x_span, y0, t_eval=x_points, method='RK45')

    # Compute Green's function solution using Simpson's Rule
    print("Computing Green's function integral using Simpson's Rule...")
    greens_solution = []
    dx = x_points[1] - x_points[0]  # Step size

    for xi in x_points:
        integrand = greens_function(xi, x_points) * forcing(x_points)

        # Simpson's Rule:
        simpson_sum = integrand[0] + integrand[-1]              # f(x0) + f(xn)
        simpson_sum += 4 * np.sum(integrand[1:-1:2])            # 4 * odd indexed terms
        simpson_sum += 2 * np.sum(integrand[2:-1:2])            # 2 * even indexed terms
        integral_value = (dx / 3) * simpson_sum

        greens_solution.append(integral_value)

    greens_solution = np.array(greens_solution)

    # --- Plot 1: Homogeneous Solution ---
    plt.figure(figsize=(8, 4))
    plt.plot(sol_hom.t, sol_hom.y[0], label="Homogeneous Solution (y'' + 2y' + y = 0)")
    plt.title("Homogeneous ODE Solution")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Non-Homogeneous vs Green's Function ---
    plt.figure(figsize=(9, 5))
    plt.plot(sol_nonhom.t, sol_nonhom.y[0], label="Numerical Solution (RK45)")
    plt.plot(x_points, greens_solution, '--', label="Green's Function Solution (Simpson's Rule)")
    plt.title("Comparison of Green's Function and Numerical Solutions")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Standard Python entry point
if __name__ == "__main__":
    print("Starting program...")
    try:
        main()
        print("\nProgram finished successfully.")
    except Exception as e:
        print("\n===== ERROR OCCURRED =====")
        print(e)
        print("\nFull traceback:")
        traceback.print_exc()
    finally:
        print("\nPress Enter to exit...")
        try:
            input()
        except:
            time.sleep(10)
