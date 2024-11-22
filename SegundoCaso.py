import numpy as np
import matplotlib.pyplot as plt

# Sistema de EDs
def f_system(x, y, z):
    dy_dx = z
    dz_dx = -5 * z - 6 * y
    return dy_dx, dz_dx

# Solución analítica
def y_analytic(x):
    return 3 * np.exp(-2 * x) - 2 * np.exp(-3 * x)

# Parámetros del problema
x0, y0, z0 = 0, 1, 0  # Condiciones iniciales: y(0) = 1, y'(0) = 0
h = 0.1  # Tamaño del paso
x_max = 5  # Intervalo de simulación

# Método de Euler
n_steps = int((x_max - x0) / h)
x_vals = np.linspace(x0, x_max, n_steps + 1)
y_euler = np.zeros(n_steps + 1)
z_euler = np.zeros(n_steps + 1)
y_euler[0], z_euler[0] = y0, z0

for i in range(n_steps):
    dy_dx, dz_dx = f_system(x_vals[i], y_euler[i], z_euler[i])
    y_euler[i + 1] = y_euler[i] + h * dy_dx
    z_euler[i + 1] = z_euler[i] + h * dz_dx

# Método de Newton-Raphson
def newton_raphson_2nd_order(y_prev, z_prev, x_curr, h, tol=1e-6, max_iter=100):
    y_guess, z_guess = y_prev, z_prev
    for _ in range(max_iter):
        # Funciones de iteración
        g1 = y_guess - y_prev - h * z_guess
        g2 = z_guess - z_prev - h * (-5 * z_guess - 6 * y_guess)
        
        # Jacobiano
        J = np.array([[1, -h], [-6 * h, 1 + 5 * h]])
        F = np.array([g1, g2])
        
        # Corrección
        delta = np.linalg.solve(J, -F)
        y_guess += delta[0]
        z_guess += delta[1]
        
        if np.linalg.norm(delta, ord=2) < tol:
            return y_guess, z_guess
    raise ValueError("Newton-Raphson no convergió")

y_newton = np.zeros(n_steps + 1)
z_newton = np.zeros(n_steps + 1)
y_newton[0], z_newton[0] = y0, z0

for i in range(n_steps):
    y_newton[i + 1], z_newton[i + 1] = newton_raphson_2nd_order(
        y_newton[i], z_newton[i], x_vals[i], h
    )

# Solución analítica
y_exact = y_analytic(x_vals)

# Mostrar soluciones en la terminal
print("x\tEuler\t\tNewton-Raphson\tSolución Analítica")
for i in range(len(x_vals)):
    print(f"{x_vals[i]:.2f}\t{y_euler[i]:.6f}\t{y_newton[i]:.6f}\t{y_exact[i]:.6f}")

# Graficar los resultados
plt.plot(x_vals, y_euler, label="Método de Euler", marker='o')
plt.plot(x_vals, y_newton, label="Método Newton-Raphson", marker='s')
plt.plot(x_vals, y_exact, label="Solución Analítica", linestyle='--')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparación de Métodos Numéricos y Solución Analítica (2do orden)")
plt.legend()
plt.grid()
plt.show()
