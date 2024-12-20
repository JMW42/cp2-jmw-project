""" 
CREATED: 20.10.2024
AUTHOR: Jonathan Will
UPDATED: 20.10.2024

This is a first test for brownian mitions/dynamics of single/few particles


"""
import matplotlib.pyplot as plt
import numpy as np


kb:float=1.380649e-23 # J/K bolzmann constant
T = 290 # K, temperature
R = 1 # particle radius
friction:float = kb*T/R # friction constant
dt:float = R**2/(kb*T) # s, timescale
rng_width:float = np.sqrt(8*friction*kb*T/dt) # width of the normal distributed rng


def brownian_walk():
    """ brownian walk function"""
    return dt/friction * np.random.normal()


def brownian_walk_2d():
    """ 2d implementation of the brownian walk"""
    return [brownian_walk(), brownian_walk()]



r = np.zeros(2)

r_x_hist = [r[0]]
r_y_hist = [r[1]]

for n in range(100):

    r = np.add(r, brownian_walk_2d())
    
    r_x_hist.append(r[0])
    r_y_hist.append(r[1])



# visualice:
fig, axes = plt.subplots(1,1, figsize=(10, 10))

axes.plot(r_x_hist, r_y_hist, "o--")

plt.show()





