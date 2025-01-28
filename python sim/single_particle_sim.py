""" 
CREATED: 20.10.2024
AUTHOR: Jonathan Will
UPDATED: 14.12.2024

This is a first test for brownian mitions/dynamics of single/few particles in python.
The aim of this code is to validify the simulation code and calculate the diffusion constant


"""

from uncertainties import ufloat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sci


# GLOBAL VARIABLES:
n_steps:int = 10000 # number of simulation steps
n_particles:int = 100 # number of prticles
kb:float=1 # bolzmann constant
T:float = 1 # K, temperature
R:float = 1 # particle radius
friction:float = 1 # friction constant
dt:float = R**2/(kb*T)/10000 # timescale
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


def fitfunc_r2(x, a, b):
    return np.add(np.multiply(x, a), b)



def fitfunc_linear(x, slope):
    return np.multiply(x, slope)


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


tarr = np.asarray(tarr)

# calculate results:

# reorganize the traces as 1 array for each axis with the subarrays being the path of the given particle
xarr = [list(list(zip(*p.trace))[0]) for p in particles] # array of xpositions of particles
yarr = [list(list(zip(*p.trace))[1]) for p in particles] # array of ypositions of particles

np.savetxt("data/single_particle/xarr.txt", xarr)
np.savetxt("data/single_particle/yarr.txt", yarr)


# calculate <r^2(t)>: enssemble mean position for given t
# calculate ensemble mean for x and y position

xarr2 = np.power(xarr, 2) # squared x positions of particles
yarr2 = np.power(yarr, 2) # squared y positions of particles

r2 =  np.add(xarr2, yarr2)
r2_mean = np.mean(r2, axis=0)


popt, pcov = sci.optimize.curve_fit(fitfunc_linear, tarr, r2_mean)
perr = np.sqrt(np.diag(pcov))


print("R2 FIT:")
for name, value, err in zip(["slope", "y0"], popt, perr):
    print(f" {name}={value}+/-{err}")



#print("a")
r2_log = np.log10(r2_mean)
tarr2 = tarr[~np.isinf(r2_mean)]
r2_log = r2_mean[~np.isinf(r2_mean)]


popt2, pcov2 = sci.optimize.curve_fit(fitfunc_r2, tarr2, r2_log)
perr2 = np.sqrt(np.diag(pcov2))



print("R2 FIT 2:")
for name, value, err in zip(["slope", "y0"], popt2, perr2):
    print(f" {name}={value}+/-{err}")


# save r2_mean dataset
df = pd.DataFrame({"time [a.u.]":tarr, "<r^2> [a.u.]":r2_mean})
df.to_csv("data/single_particle/r2_ensemble.csv")


# <(r(t)-r_0)^2>=4Dt mit D=k_BT/gamma.

# 1. calculate r(t)-r_0
dxarr = []
for xtrace in xarr: # iterate over individual x comp trace for one particle
    dxarr.append(np.subtract(xtrace, xtrace[0]))

dxarr = np.asarray(dxarr)


dyarr = []
for ytrace in yarr: # iterate over individual y comp trace for one particle
    dyarr.append(np.subtract(ytrace, ytrace[0]))

dyarr = np.asarray(dyarr)

# 2. calculate (r(t)-r_0)^2
dr2 = np.add(np.power(dxarr, 2), np.power(dyarr, 2))

# 3. calculate <(r(t)-r_0)^2>
dr2_mean = np.mean(dr2, axis=0) 


# 4. calculate <(r(t)-r_0)^2>/4t=<D(t)>
Darr = np.divide(dr2_mean, np.multiply(tarr, 4))
deltaD = np.abs(np.ones(len(Darr)) - Darr)


# save diffusion constant dataset
df = pd.DataFrame({"time [a.u.]":tarr, "D [a.u.]":Darr})
df.to_csv("data/single_particle/diffusion_constant_ensemble.csv")




# visualice results
fig, axes = plt.subplots(1,1, figsize=(10, 10))


for p in particles:
    axes.plot(list(zip(*p.trace))[0], list(zip(*p.trace))[1], ".--", label="Trajectory")
    axes.plot(list(zip(*p.trace))[0][0], list(zip(*p.trace))[1][0], "o", color="navy", label="Start")
    axes.plot(list(zip(*p.trace))[0][-1], list(zip(*p.trace))[1][-1], "o", color="red", label="End")
    


axes.set_xlabel("X Position in [a.u.]", fontsize=20)
axes.set_ylabel("Y Position in [a.u.]", fontsize=20)
#axes.legend(fontsize=20)
axes.grid()

fig.savefig("data/single_particle/traces.png", bbox_inches='tight')
plt.show()



fig, axes = plt.subplots(1,1, figsize=(10, 10))

axes.plot(tarr[1:], r2_mean[1:], label=r"$<r(t)^2>$", color="navy")
#axes.plot(tarr, fitfunc_r2(tarr, *popt), label="Fit.", color="red")

axes.plot(tarr[1:], fitfunc_linear(tarr[1:], *popt), label="Fitted Linear curve", color="red")
axes.plot(tarr[1:], fitfunc_linear(tarr[1:], 1), label="y=x", color="orange")
#axes.plot(tarr2, fitfunc_r2(tarr2, *popt2), label="Linear Fit", color="red")

axes.set_ylabel(r"Ensemble average $<r(t)^2>$", fontsize=20)
axes.set_xlabel("Time t [a.u.]", fontsize=20)
axes.set_yscale('log')
axes.set_xscale('log')
#axes.set_xlim([0.001, np.max(tarr)])
#axes.set_ylim([0.01, 20])
axes.legend(fontsize=20)
axes.grid()

fig.savefig("data/single_particle/r2_ensemble.png", bbox_inches='tight')
plt.show()



fig, axes = plt.subplots(1,1, figsize=(10, 10))

axes.plot(tarr[1:], Darr[1:], label="<D(t)>", color="navy")
axes.plot(tarr[1:], np.ones(len(tarr[1:])), "--", label="Expected value", color="red")

axes.plot(tarr[1:], deltaD[1:], label="|1 - <D(t)>|", color="green")
axes.plot(tarr[1:], np.zeros(len(deltaD[1:])), "--", color="lime")

axes.set_ylabel("<D(t)>", fontsize=20)
axes.set_xlabel("Time t [a.u.]", fontsize=20)
axes.set_ylim([-0.1, 1.5])
axes.legend(fontsize=20)
axes.grid()

fig.savefig("data/single_particle/diffusion_constant_ensemble.png", bbox_inches='tight')
plt.show()


