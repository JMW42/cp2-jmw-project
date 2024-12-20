""" 
AUTHOR: Jonathan Will
CREATED: unknown
UPDATED: 01.11.2024

The rnpg.py file is a script to test various rngs, validate code for the simulation and calculate further parameters.

"""

import numpy as np
import matplotlib.pyplot as plt



# GLOBAL VARIABLES:
kb:float=1.380649e-23 # J/K bolzmann constant
T = 290 # K, temperature
R = 1 # particle radius
friction:float = kb*T/R # friction constant
dt:float = R**2/(kb*T) # s, timescale
rng_width:float = np.sqrt(8*friction*kb*T/dt) # width of the normal distributed rng


hist = []



def rng_normal2():
    res = 0

    for i in range(20):
        res += np.random.randint(-1, 1)

    return res

    
    #


def rng_normal():
    """ rng used for the random kicks by the thermal force."""
    return np.random.normal(scale=rng_width)


for i in range(10000):
    hist.append(rng_normal())


def thermal_force(x:float, y:float, t:float):
    """ thermal force, simulates the random kicks, done for brownian motion."""
    return np.array([rng_normal(), rng_normal()])



# visualice:
fig, axes = plt.subplots(1,1, figsize=(10, 5))

axes.hist(hist, bins=100)

plt.show()



# evaluate rng force
farr = []
for i in range(1000000):
    f = thermal_force(0, 0, 0)
    farr.append(np.dot(f, f))


# second moment of force squared (dot product):
fsm=np.mean(farr)

print(f'average thermalforce: {fsm}')



friction = dt*fsm / (4*kb*T)

print(f'friction={friction}')




