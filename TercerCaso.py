import numpy as np
import matplotlib.pyplot as plt

# Definición del sistema de ecuaciones
def f(t, z):
    x, y = z
    dxdt = x + 2 * y
    dydt = 3 * x - y
    return np.array([dxdt, dydt])

# Solución analítica
def sol_analitica(t, z0):
    A = np.array([[1, 2], [3, -1]])
    eigvals, eigvecs = np.linalg.eig(A)  # Autovalores y autovectores de A
    C = np.linalg.solve(eigvecs, z0)  # Coeficientes de la solución general
    sol = np.zeros((len(t), len(z0)))
    for i, ti in enumerate(t):  # Calcular para cada t
        sol[i] = np.dot(eigvecs, C * np.exp(eigvals * ti))
    return sol

# Método de Euler
def metodo_euler(f, t, z0, h):
    z = np.zeros((len(t), len(z0)))
    z[0] = z0
    for i in range(1, len(t)):
        z[i] = z[i - 1] + h * f(t[i - 1], z[i - 1])
    return z

# Método de Newton-Raphson (iterativo en cada paso)
def metodo_newton_raphson(f, t, z0, h, tol=1e-6, max_iter=50):
    z = np.zeros((len(t), len(z0)))
    z[0] = z0
    for i in range(1, len(t)):
        z_guess = z[i - 1] + h * f(t[i - 1], z[i - 1])  # Predicción inicial (Euler)
        for _ in range(max_iter):  # Iteraciones para convergencia
            F = z_guess - z[i - 1] - h * f(t[i], z_guess)
            J = np.array([[1 - h, -2 * h], [-3 * h, 1 + h]])  # Jacobiano aproximado
            delta = np.linalg.solve(J, -F)
            z_guess += delta
            if np.linalg.norm(delta) < tol:
                break
        z[i] = z_guess
    return z

# Configuración del problema
t0, tf, h = 0, 2, 0.1
t = np.arange(t0, tf + h, h)
z0 = np.array([1, 0])  # Condiciones iniciales

# Soluciones
sol_analitica = sol_analitica(t, z0)
sol_euler = metodo_euler(f, t, z0, h)
sol_newton_raphson = metodo_newton_raphson(f, t, z0, h)

# Mostrar los resultados en la terminal
print(f"{'t':>5} {'Euler (x)':>12} {'Euler (y)':>12} {'Newton-Raphson (x)':>20} {'Newton-Raphson (y)':>20} {'Analítica (x)':>15} {'Analítica (y)':>15}")
for i in range(len(t)):
    print(f"{t[i]:5.2f} {sol_euler[i, 0]:12.6f} {sol_euler[i, 1]:12.6f} {sol_newton_raphson[i, 0]:20.6f} {sol_newton_raphson[i, 1]:20.6f} {sol_analitica[i, 0]:15.6f} {sol_analitica[i, 1]:15.6f}")

# Graficar los resultados
plt.figure(figsize=(14, 6))
plt.plot(t, sol_analitica[:, 0], 'g--', label='Solución Analítica (x)', linewidth=2)
plt.plot(t, sol_analitica[:, 1], 'g-.', label='Solución Analítica (y)', linewidth=2)
plt.plot(t, sol_euler[:, 0], 'b-o', label='Método de Euler (x)', markersize=4)
plt.plot(t, sol_euler[:, 1], 'b-s', label='Método de Euler (y)', markersize=4)
plt.plot(t, sol_newton_raphson[:, 0], 'r-o', label='Método Newton-Raphson (x)', markersize=4)
plt.plot(t, sol_newton_raphson[:, 1], 'r-s', label='Método Newton-Raphson (y)', markersize=4)
plt.title('Comparación de Métodos para un Sistema de Ecuaciones Diferenciales')
plt.xlabel('t')
plt.ylabel('Valores de x e y')
plt.grid()
plt.legend()
plt.show()
