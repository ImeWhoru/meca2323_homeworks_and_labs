import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange


def parameters():
  hung_mass = np.array([0, .0682, .182, .295, .408, .522, .748, .975])
  tension_UD = np.array([.38, .7, 1.23, 1.76, 2.3, 2.82, 3.89, 4.9])
  C_De = .09
  g = 9.81
  U = 2.6
  D = .05
  L = .5
  rho = 1.2112
  theta_stagnation_point = np.arange(-10,15,5)
  dynamic_pressure_around_stagnation_point = np.array([116.0, 128.5, 137.2, 140.4, 137.3])
  theta = np.arange(0, 180, 10)
  dynamic_pressure = np.array([137.2, 137.3, 110.65, 55, -19.5, -92.3, -151.5 -180.5, -180.5, -146.5, -143.7, -146.5, -148.5, -147.5, -153.5, -159.5, -160.5, -162.5, -163.5])

  return hung_mass, tension_UD, C_De, g, U, dynamic_pressure, rho, theta, theta_stagnation_point, dynamic_pressure_around_stagnation_point, D, L


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

  return U_infinity

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

def computation_pressure_drag_coefficient(dynamic_pressure, rho, U_infty, display):
  print(f'rho = {rho}')
  print(f'U_infinity = {U_infty}')
  Cp_theta = dynamic_pressure*2/(rho*U_infty**2)
  if (display == True):
    plt.plot(theta, Cp_theta)
    plt.xlabel('theta [°]')
    plt.ylabel('local pressure drag coefficient [-]')
    plt.title('Local pressure drag coefficients around half of the cylinder')
    plt.grid()
    plt.show()

  return Cp_theta

def computation_local_pressure_drag_force(Cp_theta, rho, D, L, U_infty):
  Fp_theta = .5*rho*D*L*np.pi/17*U_infty**2 * Cp_theta

  return Fp_theta

def computation_total_pressure_drag_force(Fp_theta):
  F_p = np.sum(Fp_theta)
  print(f'F_p = {F_p:.4f}')
  return F_p

def computation_total_pressure_drag_coefficient(F_p, rho, U_infty, D, L):
  Cp = F_p/(.5*rho*U_infty**2*np.pi*D*L)
  print(f'Cp = {Cp}')
  return Cp

def computation_CD(k_D, rho, u, D, L, C_De):
  C_D = k_D*(2.6-.38)/(.5*rho*u*u*D*L) - C_De
  print(f'C_D = {C_D}')

if __name__ == '__main__':
  hung_mass, tension_UD, C_De, g, U, dynamic_pressure, rho, theta, theta_stagnation_point, dynamic_pressure_around_stagnation_point, D, L = parameters()
  k_D = regression_calibration_data(hung_mass, tension_UD, g, display = False)
  F_D = drag_computation(k_D, U, tension_UD)
  interpolation(theta_stagnation_point, dynamic_pressure_around_stagnation_point, display = False)
  U_infinity = U_infinity_computation(dynamic_pressure_around_stagnation_point[-2], rho)
  Cp_theta = computation_pressure_drag_coefficient(dynamic_pressure, rho, U_infinity, display = False)
  Fp_theta = computation_local_pressure_drag_force(Cp_theta, rho, D, L, U_infinity)
  Fp = computation_total_pressure_drag_force(Fp_theta)
  Cp = computation_total_pressure_drag_coefficient(Fp, rho, U_infinity, D, L)
  Cd = computation_CD(k_D, rho, U_infinity, D, L, C_De)





