""" 
CREATED: 20.10.2024
AUTHOR: Jonathan Will
UPDATED: 01.11.2024

This is a first test for brownian mitions/dynamics of single/few particles in python.
The aim of this code is to validify the simulation code and calculate the diffusion constant


"""
import matplotlib.pyplot as plt
import numpy as np


# GLOBAL VARIABLES:
friction:float = 1
dt:float = 1e-10 # s, timescale
kb:float=1.380649e-23 # J/K bolzmann constant
T = 290 # K, temperature


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
    return (friction * kb*T)*np.array([rng_normal(), rng_normal()])


def borwnian_move(x:float, y:float, t:float):
    """ method to calculate the change of a particles position done by a brownian movement."""
    Fex = external_force(x, y, t)
    Ft = thermal_force(x, y, t)
    return dt/friction*(Fex + Ft)



# MAIN CODE:
particles = []


particles.append(Particle2D(0, 0))


# simulation loop
t = 0.1
tarr = [t]
for i in range(1000):
    # itterate over every particle:
    for p in particles:
        p.move(*borwnian_move(p.x, p.y, t))

    t += dt
    tarr.append(t)



# calculate results:

for p in particles:
    
    # mean position
    print(f'<r> = ({np.mean(p.hist_x)}+/- {np.std(p.hist_x)}, {np.mean(p.hist_y)}+/- {np.std(p.hist_y)})')


    #<(r(t) - r(t=0))^2> = 4Dt
    
    # calculate delta to start position
    xarr = np.subtract(p.hist_x, p.hist_x[0])
    yarr = np.subtract(p.hist_y, p.hist_y[0])



    # square elements
    xarr = np.power(p.hist_x, 2)
    yarr = np.power(p.hist_y, 2)

    arr = np.add(xarr, yarr) # (r(t) - r(t=0))^2

    darr = np.divide(arr, tarr) # list of diffusion values


    print(f' <D> = {np.mean(darr)} +/- {np.std(darr)}')
    print(f' <gamma> = {kb*T / np.mean(darr)}')
    


# visualice results
fig, axes = plt.subplots(1,1, figsize=(10, 10))


for p in particles:
    axes.plot(p.hist_x, p.hist_y, ".--")
    axes.plot(p.hist_x[-1], p.hist_y[-1], "o", color="red")

plt.show()