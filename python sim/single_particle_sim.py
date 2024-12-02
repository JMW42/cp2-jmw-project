""" 
CREATED: 20.10.2024
AUTHOR: Jonathan Will
UPDATED: 29.11.2024

This is a first test for brownian mitions/dynamics of single/few particles in python.
The aim of this code is to validify the simulation code and calculate the diffusion constant


"""

from uncertainties import ufloat
import matplotlib.pyplot as plt
import numpy as np


# GLOBAL VARIABLES:
kb:float=1 # bolzmann constant
T:float = 1 # K, temperature
R:float = 1 # particle radius
friction:float = 1 # friction constant
dt:float = R**2/(kb*T)/100 # timescale
rng_width:float = np.sqrt(8*friction*kb*T/dt) # width of the normal distributed rng

print(f' dt = {dt}')


# PARTICLE:
class Particle2D:
    """ Particle2D, class used to simulate the particles."""
    def __init__(self, x0:float, y0:float):
        self.x:float = x0
        self.y:float = y0

        # movement history
        self.trace = [[self.x, self.y]]
    

    def move(self, dx:float, dy:float):
        self.x +=dx
        self.y +=dy

        self.trace.append([self.x, self.y])


# METHODS:
def rng_normal():
    """ rng used for the random kicks by the thermal force."""
    return np.random.normal(scale=rng_width)


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


for i in range(10):
    particles.append(Particle2D(0, 0))


# simulation loop
t = 1e-20
tarr = [t]
for i in range(1000):
    # itterate over every particle:
    for p in particles:
        p.move(*borwnian_move(p.x, p.y, t))

    t += dt
    tarr.append(t)



# calculate results:

i = -1
for p in particles:
    i += 1
    print(f'-'*40)
    print(f"Evaluating particle: {i}")
    
    
    
    # mean position
    ufrx = ufloat(np.mean(list(zip(*p.trace))[0]), np.std(list(zip(*p.trace))[0]))
    ufry = ufloat(np.mean(list(zip(*p.trace))[1]), np.std(list(zip(*p.trace))[1]))
    
    # check that <r> = (0, 0)
    print(f' <r> = ({ufrx} , {ufry})')


    # calculate diffusion coefficient for each timestep
    #with: <(r(t) - r(t=0))^2> = 4Dt

    # caluclate list of distance vectors to starting position    
    dr = np.subtract(p.trace, p.trace[0]) #r(t) - r(t=0)

    # calculate power 2 term in formular
    dr2 = np.add(np.power(list(zip(*p.trace))[0],2), np.power(list(zip(*p.trace))[1], 2)) # (r(t) - r(t=0))^2 = dr^2

    # calculate D(t) as list corresponding to t in tarr
    darr = np.divide(dr2, np.multiply(tarr, 4)) # (r(t) - r(t=0))^2/(4t) = dr^2/(4t) = D(t)

    ufd = ufloat(np.mean(darr), np.std(darr)) # ufloat object of D(t)
    print(f' <D> = {ufd}')
    print(f' compare: D = {kb*T/friction}')





# visualice results
fig, axes = plt.subplots(1,1, figsize=(10, 10))


for p in particles:
    axes.plot(list(zip(*p.trace))[0], list(zip(*p.trace))[1], ".--")
    axes.plot(list(zip(*p.trace))[0][-1], list(zip(*p.trace))[1][-1], "o", color="red")
    axes.plot(list(zip(*p.trace))[0][0], list(zip(*p.trace))[1][0], "o", color="navy")
    axes.plot([ufrx.nominal_value], [ufry.nominal_value], "o", color="lime")

plt.show()