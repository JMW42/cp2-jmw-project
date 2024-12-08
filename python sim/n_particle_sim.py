""" 
CREATED: 08.12.2024
AUTHOR: Jonathan Will
UPDATED: 08.12.2024

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

# nature constants
kb:float=1 # bolzmann constant

# environment / system constant
T:float = 1 # K, temperature
R:float = 1 # particle radius
friction:float = 1 # friction constant

# derived variables/constants
dt:float = R**2/(kb*T)/100 # timescale
RNG_WIDTH:float = np.sqrt(2*friction*kb*T/dt) # width of the normal distributed rng

# further variables:
particles:list = [] # list of all particles
particles_traces_x:list = [] # x component for the traces of all particles
particles_traces_y:list = [] # y component for the traces of all particles

t:np.ndarray = np.arange(dt/100, NUMBER_STEPS*dt, dt) # list of all simulated time values:


# ########################################################################################################################
# ########################################################################################################################
# CLASSES:

class Particle2D:
    """ Particle2D, class used to simulate the particles."""
    def __init__(self, x0:float=0, y0:float=0):
        self.x:float = x0
        self.y:float = y0
        
        # get id for later usage of the trace lists
        self.id = len(particles)

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




# ########################################################################################################################
# ########################################################################################################################
# MAIN CODE:






