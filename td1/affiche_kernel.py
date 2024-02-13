import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Définir le noyau de la chaleur
def gaussian_kernel(x, t, sigma=1.0):
    return 1 / (sigma * np.sqrt(2 * np.pi * t)) * np.exp(-x**2 / (2 * sigma**2 * t))

# Définir la fonction 1/x^2, en évitant la division par zéro
def inverse_square(x, t):
    # Utiliser np.where pour éviter la division par zéro en remplaçant 0 par np.nan
    return np.where(x != 0, 1 / x**2, np.nan)

# Paramètres initiaux
x_values = np.linspace(-5, 5, 400)
t_values = np.linspace(1, 0.01, 100)  # t décroît jusqu'à presque 0
sigma = 1.0  # Coefficient de diffusion

fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(0, 1.5)
line, = ax.plot([], [], lw=2, label='Gaussian Kernel')
line2, = ax.plot(x_values, inverse_square(x_values), lw=2, label='1/x^2', color='orange')

# Initialiser l'animation en créant un tracé vide
def init():
    line.set_data([], [])
    return line, line2

# Mise à jour de l'animation à chaque frame
def update(t):
    y_values = gaussian_kernel(x_values, t, sigma)
    line.set_data(x_values, y_values)
    # La fonction 1/x^2 ne change pas, donc pas besoin de la mettre à jour ici
    return line, line2

# Ajouter une légende
ax.legend()

ani = FuncAnimation(fig, update, frames=t_values, init_func=init, blit=True, interval=10)

plt.show()
