""" 
CREATED: 20.10.2024
AUTHOR: Jonathan Will
UPDATED: 30.10.2024

This is a first test for brownian mitions/dynamics of single/few particles in python


"""
import matplotlib.pyplot as plt
import numpy as np


# GLOBAL VARIABLES:
friction:float = 1
dt:float = 2


# PARTICLE:
class Particle2D:
    """ Particle2D, class used to simulate the particles."""
    def __init__(self, x0:float, y0:float):
        self.x:float = x0
        self.y:float = y0

        # temporal history:
        self.hist_x = [self.x]
        self.hist_y = [self.y]


    def move(self, dx:float, dy:float):
        self.x +=dx
        self.y +=dy

        self.hist_x.append(self.x)
        self.hist_y.append(self.y)


# METHODS:
def rng_normal():
    """ rng used for the random kicks by the thermal force."""
    return np.random.normal()


def external_force(x:float, y:float, t:float):
    """ external force, defined by position and time"""
    return np.array([0, 0])


def thermal_force(x:float, y:float, t:float):
    """ thermal force, simulates the random kicks, done for brownian motion."""
    return np.array([rng_normal(), rng_normal()])


def borwnian_move(x:float, y:float, t:float):
    """ method to calculate the change of a particles position done by a brownian movement."""
    Fex = external_force(x, y, t)
    Ft = thermal_force(x, y, t)
    return dt/friction*(Fex + Ft)



# MAIN CODE:
particles = []


particles.append(Particle2D(0, 0))


# simulation loop
t = 0
for i in range(10000):
    # itterate over every particle:
    for p in particles:
        p.move(*borwnian_move(p.x, p.y, t))

    t += dt


# visualice results

fig, axes = plt.subplots(1,1, figsize=(10, 10))


for p in particles:
    axes.plot(p.hist_x, p.hist_y)

plt.show()