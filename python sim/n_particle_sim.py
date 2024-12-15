""" 
CREATED: 08.12.2024
AUTHOR: Jonathan Will
UPDATED: 14.12.2024

Implementation of an n-particle brownian motion simulation.

"""

# ########################################################################################################################
# ########################################################################################################################
# IMPORTS:

import matplotlib.pyplot as plt
import numpy as np


# ########################################################################################################################
# ########################################################################################################################
# GLOBAL VARIABLES:

# simulation related
NUMBER_STEPS:int = 1000 # number of simulation steps
NUMBER_PARTICLES:int = 100 # number of prticles
EVALUATE_ENSEMBLE = False

# nature constants
kb:float=1 # bolzmann constant

# environment / system constant
T:float = 1 # K, temperature
R:float = 1 # particle radius
friction:float = 1 # friction constant

# derived variables/constants
dt:float = R**2/(kb*T)/100 # timescale
rng_width:float = np.sqrt(2*friction*kb*T/dt) # width of the normal distributed rng

# further variables:
particles:list = [] # list of all particles
particles_traces_x:list = [] # x component for the traces of all particles
particles_traces_y:list = [] # y component for the traces of all particles

t:np.ndarray = np.arange(dt/100, NUMBER_STEPS*dt, dt) # list of all simulated time values:


# lattice parameters:
a1 = np.asarray([1,0]) # lattice basis vector
a2 = np.asarray([0,1]) # lattice basis vector

# ########################################################################################################################
# ########################################################################################################################
# CLASSES:

class Particle2D:
    """ Particle2D, class used to simulate the particles."""
    def __init__(self, x0:float=0, y0:float=0, id:int=0):
        self.x:float = x0
        self.y:float = y0
        
        # get id for later usage of the trace lists
        self.id:int = id

        # append new list to trace list
        particles_traces_x.append([x0])
        particles_traces_y.append([y0])
    

    def move(self, dx:float, dy:float) -> None:
        self.x +=dx
        self.y +=dy

        # add new position to trace lists
        particles_traces_x[self.id].append(self.x)
        particles_traces_y[self.id].append(self.y)


# ########################################################################################################################
# ########################################################################################################################
# METHODS:

def rng_normal()->np.ndarray:
    """ rng used for the random kicks by the thermal force."""
    return np.random.normal(scale=rng_width)


def external_force(x:float, y:float, t:float)->np.ndarray:
    """ external force, defined by position and time"""
    return np.array([0, 0])


def thermal_force(x:float, y:float, t:float)->np.ndarray:
    """ thermal force, simulates the random kicks, done for brownian motion."""
    return (friction * kb*T)*np.array([rng_normal(), rng_normal()])


def borwnian_move(x:float, y:float, t:float)->np.ndarray:
    """ method to calculate the change of a particles position done by a brownian movement."""
    Fex = external_force(x, y, t)
    Ft = thermal_force(x, y, t)
    return dt/friction*(Fex + Ft)


def capture_particle_snapshot(name):
    """ Draws the current position of the particles as matplotlib plot"""

    fig, axes = plt.subplots(1,1, figsize=(10, 10))

    for p in particles:
        axes.plot(p.x, p.y, "o", alpha=0.5, color="black")

    plt.savefig(f'data/n_particle/snapshot/{name}')
    plt.close(fig)


# ########################################################################################################################
# ########################################################################################################################
# MAIN CODE:

plt.ioff()


# create particles:
for i in range(NUMBER_PARTICLES):
    
    n1:int = int(i/np.sqrt(NUMBER_PARTICLES)) # n1 gitter coordinate
    n2:int = int(i%np.sqrt(NUMBER_PARTICLES)) # n2 gitter coordinate

    pos = np.add(np.multiply(a1, n1), np.multiply(a2, n2)) # position in global coordinate grid

    p:Particle2D = Particle2D(*pos, i) # initialise particle
    particles.append(p) # add particle to particle list




# simulation:
for i in range(1, NUMBER_STEPS):
    print(f'{i}') # print current simulation step

    #capture_particle_snapshot(f"{i}.png")

    for p in particles:
        p.move(*borwnian_move(p.x, p.y, t))
    




# evaluate results:
## show all particle tarces:

fig, axes = plt.subplots(1,1, figsize=(10, 10))

for xarr, yarr in zip(particles_traces_x, particles_traces_y):
    axes.plot(xarr, yarr, ".--", alpha=0.5)
    axes.plot(xarr[0], yarr[0], "o", alpha=1, color="navy")
    axes.plot(xarr[-1], yarr[-1], "o", alpha=1, color="red")

fig.savefig("data/n_particle/particle_traces.png")
plt.show()



# SAVE PARTICLE TRACES:
np.savetxt("data/n_particle/x_traces.txt", particles_traces_x)
np.savetxt("data/n_particle/y_traces.txt", particles_traces_y)



# EVALUATE ENSEBLE VARIABLES <r^2(t)> and <D(t))>
if EVALUATE_ENSEMBLE:
    # calculate ensemble average position:
    ## first calculate r^2 = x^2 + y^2 for every particle, for each position
    r2 =  np.add(np.power(particles_traces_x, 2), np.power(particles_traces_y, 2))
    r2_mean = np.mean(r2, axis=0) # ensemble average for given value of time 


    # <(r(t)-r_0)^2>=4Dt mit D=k_BT/gamma.

    # 1. calculate r(t)-r_0

    dxarr = []
    for xtrace in particles_traces_x:
        dxarr.append(np.subtract(xtrace, xtrace[0]))

    dxarr = np.asarray(dxarr)


    dyarr = []
    for ytrace in particles_traces_y:
        dyarr.append(np.subtract(ytrace, ytrace[0]))

    dyarr = np.asarray(dyarr)

    # 2. calculate (r(t)-r_0)^2
    dr2 = np.add(np.power(dxarr, 2), np.power(dyarr, 2))

    # 3. calculate <(r(t)-r_0)^2>
    dr2_mean = np.mean(dr2, axis=0) 


    # 4. calculate <(r(t)-r_0)^2>/4t=<D(t)>
    Darr = np.divide(dr2_mean, np.multiply(t, 4))


    fig, axes = plt.subplots(1,1, figsize=(10, 10))

    axes.plot(t[1:], r2_mean[1:], label="<r(t)^2>", color="navy")

    axes.set_ylabel("Ensemble average <r(t)^2>")
    axes.set_xlabel("Time t [a.u.]")
    axes.set_yscale('log')
    axes.set_xscale('log')
    axes.grid()

    plt.show()



    fig, axes = plt.subplots(1,1, figsize=(10, 10))

    axes.plot(t[1:], Darr[1:], label="<D(t)>", color="navy")

    axes.set_ylabel("<D(t)>")
    axes.set_xlabel("Time t [a.u.]")
    axes.grid()

    plt.show()

