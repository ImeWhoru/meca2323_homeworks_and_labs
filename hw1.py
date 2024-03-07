import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange


def parameters():
  hung_mass = np.array([0, .0682, .182, .295, .408, .522, .748, .975])
  tension_UD = np.array([.38, .7, 1.23, 1.76, 2.3, 2.82, 3.89, 4.9])
  C_De = .09
  g = 9.81
  U = 2.6
  dynamic_pressure = []
  rho = 1.2112
  theta = np.arange(0,190,10)
  theta_stagnation_point = np.arange(-10,15,5)
  dynamic_pressure_around_stagnation_point = np.array([116.0, 128.5, 137.2, 140.4, 137.3])

  return hung_mass, tension_UD, C_De, g, U, dynamic_pressure, rho, theta, theta_stagnation_point, dynamic_pressure_around_stagnation_point


def regression_calibration_data(hung_mass, tension_UD, g, display):
  # Given data
  hung_mass *= g
  adjusted_tension = tension_UD - tension_UD[0]

  # Construct the matrix A for the linear regression
  A = np.vstack([hung_mass, np.ones(len(hung_mass))]).T

  # Solve for the least squares regression coefficients (k_D and intercept)
  coefficients = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), adjusted_tension)
  # Extract the value of k_D
  global k_D
  k_D = coefficients[0]

  print(f"k_D = {k_D:.4f} [N/V]")
  if (display == True):
    # Plot the data points along with the regression line
    plt.scatter(hung_mass/g, adjusted_tension, label="Data Points")
    plt.plot(hung_mass/g, k_D * hung_mass, color='red', label="Least Squares Regression")
    plt.xlabel("Hung mass [kg]")
    plt.ylabel("Adjusted Tension (tension_ud - 0.38)")
    plt.title("Least Squares Regression")
    plt.legend()
    plt.show()

  return k_D
def drag_computation(k_D, U, tension_UD):
  F_D = k_D*(U - tension_UD[0])
  print(f'F_D = {F_D:.4f} [N]')

def U_infinity_computation(stagnation_dynamic_pressure, rho):
  U_infinity = (2 * stagnation_dynamic_pressure/ rho)**.5
  print(f'U_infty = {U_infinity:.4f} [m/s]')

def interpolation(theta, dyn_pressure, display):

  # Compute the Lagrange interpolating polynomial
  poly = lagrange(theta_stagnation_point, dynamic_pressure_around_stagnation_point)

  # Find the maximum value and corresponding x-value
  max_value = np.max(dynamic_pressure_around_stagnation_point)
  max_x = theta_stagnation_point[np.argmax(dynamic_pressure_around_stagnation_point)]

  print(f"Maximum value occurs at x = {max_x:.3f}, y = {max_value:.3f}.")

  # Plot the Lagrange polynomial
  if (display == True):
    x_new = np.linspace(min(theta_stagnation_point) - 4, max(theta_stagnation_point) + 4, 101)
    plt.scatter(theta_stagnation_point, dynamic_pressure_around_stagnation_point, label='Data')
    plt.plot(x_new, poly(x_new), label='Lagrange Polynomial', color='red')
    plt.plot(max_x, max_value, 'rx', markersize=10, label=f'Maximum Point (theta = {max_x}[°], Dynamic Pressure = {max_value} [Pa])')  # Mark maximum point
    plt.xlabel('Theta [°]')
    plt.ylabel('Dynamic Pressure [Pa]')
    plt.title('Lagrange Interpolation of the stagnation point')
    plt.legend()
    plt.grid()
    plt.savefig('stagnation_point_lagrange_interpolation', transparent = True)
    plt.show()

if __name__ == '__main__':
  hung_mass, tension_UD, C_De, g, U, dynamic_pressure, rho, theta, theta_stagnation_point, dynamic_pressure_around_stagnation_point = parameters()
  k_D = regression_calibration_data(hung_mass, tension_UD, g, display = False)
  F_D = drag_computation(k_D, U, tension_UD)
  interpolation(theta_stagnation_point, dynamic_pressure_around_stagnation_point, display = False)
  U_infinity = U_infinity_computation(dynamic_pressure_around_stagnation_point[-2], rho)





