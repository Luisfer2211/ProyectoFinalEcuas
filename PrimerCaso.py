import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
def f(x, y):
    return -2 * y + 1  # Derivada: dy/dx = -2y + 1

def f_analytic(x):
    return 0.5 * (1 - np.exp(-2 * x))  # Solución analítica

x0, y0 = 0, 0  # Condición inicial: y(0) = 0
h = 0.1  # Tamaño del paso
x_max = 5  # Valor máximo de x para la simulación

# Método de Euler
n_steps = int((x_max - x0) / h)  # Número de pasos
x_vals = np.linspace(x0, x_max, n_steps + 1)  # Valores de x
y_euler = np.zeros(n_steps + 1)  # Valores de y con Euler
y_euler[0] = y0  # Condición inicial

for i in range(n_steps):
    y_euler[i + 1] = y_euler[i] + h * f(x_vals[i], y_euler[i])

# Método de Newton-Raphson
def newton_raphson(y_prev, x_curr, h, tol=1e-6, max_iter=100):
    y_guess = y_prev  # Suposición inicial
    for _ in range(max_iter):
        # Función de iteración de Newton-Raphson
        g = y_guess - y_prev - h * f(x_curr, y_guess)
        g_prime = 1 - h * (-2)  # Derivada parcial respecto a y
        y_new = y_guess - g / g_prime
        if abs(y_new - y_guess) < tol:  # Verificar convergencia
            return y_new
        y_guess = y_new
    raise ValueError("El método de Newton-Raphson no convergió")

y_newton = np.zeros(n_steps + 1)
y_newton[0] = y0  # Condición inicial

for i in range(n_steps):
    y_newton[i + 1] = newton_raphson(y_newton[i], x_vals[i], h)

# Solución analítica
y_exact = f_analytic(x_vals)

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
plt.title("Comparación de Métodos Numéricos y Solución Analítica Primer Grado")
plt.legend()
plt.grid()
plt.show()
