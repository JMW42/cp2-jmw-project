""" 
AUTHOR: Jonathan Will
CREATED: unknown
UPDATED: 01.11.2024

The rnpg.py file is a script to test various rngs, validate code for the simulation and calculate further parameters.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


# GLOBAL VARIABLES:
kb:float=1 # J/K bolzmann constant
T = 1 # K, temperature
R = 1 # particle radius
friction:float = kb*T/R # friction constant
dt:float = R**2/(kb*T)/100 # s, timescale
rng_width:float = np.sqrt(2*friction*kb*T/dt) # width of the normal distributed rng


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


for i in range(300*100*2):
    hist.append(rng_normal())


def thermal_force(x:float, y:float, t:float):
    """ thermal force, simulates the random kicks, done for brownian motion."""
    return np.array([rng_normal(), rng_normal()])


def func(x, ampl, x0, sigma, offset):
    return ampl* np.exp( - (x - x0)**2 / (2 * sigma**2)) + offset


# visualice:
fig, axes = plt.subplots(1,1, figsize=(10, 5))

count, bins, patches, = axes.hist(hist, bins=100, color="grey", ec='black', label="np.random.normal")
xarr = bins[:-1] + (bins[1]-bins[0])/2


popt, pcov = opt.curve_fit(func, xarr, count)
perr = np.sqrt(np.diag(pcov))

for name, value, error in zip(["ampl", "x0", "sigma", "offset"], popt, perr):
    print(f"{name} = {value} +/- {error}")


axes.plot(xarr, count, ".", linewidth=2, color='navy')
axes.plot(xarr, func(xarr, *popt), "-", linewidth=2, color='r', label="Fit")


axes.set_xlabel("RNG", fontsize=20)
axes.set_ylabel("Counts", fontsize=20)
axes.legend()
axes.grid()


fig.savefig("data/rng_hist.png", bbox_inches='tight')

plt.show()



# evaluate rng force
farr = []
for i in range(200):
    f = thermal_force(0, 0, 0)
    farr.append(np.dot(f, f))


# second moment of force squared (dot product):
fsm=np.mean(farr)

print(f'average thermalforce: {fsm}')



friction = dt*fsm / (4*kb*T)

print(f'friction={friction}')




