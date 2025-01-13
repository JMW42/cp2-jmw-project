import matplotlib.pyplot as plt
import numpy as np
import time



# nature constants
kb:float=1 # bolzmann constant

# environment / system constant
T:float = 1 # K, temperature
R:float = 1 # particle radius
friction:float = 1 # friction constant

# interaction parameters:
k_int:float = 60 # interaction constant, spring constant
int_exp:int = 1 # interaction exponential



def repulsive_force(r:np.ndarray) -> np.ndarray:
    return  (2*R - r)*k_int*np.heaviside(2*R-r, 0)


r = np.linspace(0, 4*R, 200)



Farr = repulsive_force(r)
print(Farr)

fig, axes = plt.subplots(1,1, figsize=(10, 5))

axes.plot(r/(2*R), Farr/k_int, "-", color="navy")

axes.set_ylabel(r"Repulsive force $\frac{F_{rep.}(r)}{k}$ in [a.u.]", fontsize=20)
axes.set_xlabel(r"$\frac{r}{2R}$ in [a.u.]", fontsize=20)
axes.grid()

fig.savefig("data/repulsive_force.png")

plt.show()