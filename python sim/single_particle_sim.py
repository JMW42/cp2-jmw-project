""" 
CREATED: 20.10.2024
AUTHOR: Jonathan Will
UPDATED: 08.12.2024

This is a first test for brownian mitions/dynamics of single/few particles in python.
The aim of this code is to validify the simulation code and calculate the diffusion constant


"""

from uncertainties import ufloat
import matplotlib.pyplot as plt
import numpy as np


# GLOBAL VARIABLES:
n_steps:int = 1000 # number of simulation steps
n_particles:int = 100 # number of prticles
kb:float=1 # bolzmann constant
T:float = 1 # K, temperature
R:float = 1 # particle radius
friction:float = 1 # friction constant
dt:float = R**2/(kb*T)/100 # timescale
rng_width:float = np.sqrt(2*friction*kb*T/dt) # width of the normal distributed rng

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


for i in range(n_particles):
    particles.append(Particle2D(0, 0))


# simulation loop
t = 1e-20
tarr = [t]
for i in range(n_steps):
    print(f"calculating step: {i}")
    # itterate over every particle:
    for p in particles:
        p.move(*borwnian_move(p.x, p.y, t))

    t += dt
    tarr.append(t)



# calculate results:

# reorganize the traces as 1 array for each axis with the subarrays being the path of the given particle
xarr = [list(list(zip(*p.trace))[0]) for p in particles] # array of xpositions of particles
yarr = [list(list(zip(*p.trace))[1]) for p in particles] # array of ypositions of particles


# calculate <r^2(t)>: enssemble mean position for given t
# calculate ensemble mean for x and y position

xarr2 = np.power(xarr, 2)
yarr2 = np.power(yarr, 2)

r2 =  np.add(xarr2, yarr2)
r2_mean = np.mean(r2, axis=0)


# <(r(t)-r_0)^2>=4Dt mit D=k_BT/gamma.

# 1. calculate r(t)-r_0

dxarr = []
for xtrace in xarr:
    dxarr.append(np.subtract(xtrace, xtrace[0]))

dxarr = np.asarray(dxarr)


dyarr = []
for ytrace in yarr:
    dyarr.append(np.subtract(ytrace, ytrace[0]))

dyarr = np.asarray(dyarr)

# 2. calculate (r(t)-r_0)^2
dr2 = np.add(np.power(dxarr, 2), np.power(dyarr, 2))

# 3. calculate <(r(t)-r_0)^2>
dr2_mean = np.mean(dr2, axis=0) 


# 4. calculate <(r(t)-r_0)^2>/4t=<D(t)>
Darr = np.divide(dr2_mean, np.multiply(tarr, 4))




# visualice results
fig, axes = plt.subplots(1,1, figsize=(10, 10))


for p in particles:
    axes.plot(list(zip(*p.trace))[0], list(zip(*p.trace))[1], ".--")
    axes.plot(list(zip(*p.trace))[0][-1], list(zip(*p.trace))[1][-1], "o", color="red")
    axes.plot(list(zip(*p.trace))[0][0], list(zip(*p.trace))[1][0], "o", color="navy")

plt.show()




fig, axes = plt.subplots(1,1, figsize=(10, 10))

axes.plot(tarr, r2_mean, label="<r(t)^2>", color="navy")

axes.set_ylabel("Ensemble average <r(t)^2>")
axes.set_xlabel("Time t [a.u.]")
axes.set_yscale('log')
axes.set_xscale('log')
axes.set_xlim([0.1, 10])
axes.set_ylim([0.3, 50])
axes.grid()

plt.show()



fig, axes = plt.subplots(1,1, figsize=(10, 10))

axes.plot(tarr, Darr, label="<D(t)>", color="navy")

axes.set_ylabel("<D(t)>")
axes.set_xlabel("Time t [a.u.]")
axes.grid()

plt.show()