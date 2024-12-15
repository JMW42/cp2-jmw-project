""" 
CREATED: 08.12.2024
AUTHOR: Jonathan Will
UPDATED: 14.12.2024

Python script to render videos of particle traces with given matrix files creates by single_particle and n_particle simulations.

"""

# ########################################################################################################################
# ########################################################################################################################
# IMPORTS:

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import itertools

# ########################################################################################################################
# ########################################################################################################################
# GLOBAL VARIABLES:

x_trace_file = "data/n_particle/x_traces.txt"
y_trace_file = "data/n_particle/y_traces.txt"


# ########################################################################################################################
# ########################################################################################################################
# MAIN CODE:


particles_traces_x = np.loadtxt(x_trace_file) # x component for the traces of all particles
particles_traces_y = np.loadtxt(y_trace_file) # y component for the traces of all particles



# visualice all traces:
fig, axes = plt.subplots(1,1, figsize=(10, 10))

for xarr, yarr in zip(particles_traces_x, particles_traces_y):
    axes.plot(xarr, yarr, ".--", alpha=0.5)
    axes.plot(xarr[0], yarr[0], "o", alpha=1, color="navy")
    axes.plot(xarr[-1], yarr[-1], "o", alpha=1, color="red")

fig.savefig("data/n_particle/particle_traces_render.png")
plt.show()





fig, axes = plt.subplots()

axes.set_xlim([np.min(particles_traces_x)-np.min(particles_traces_x)*0.1, np.max(particles_traces_x)*1.1])
axes.set_ylim([np.min(particles_traces_y)-np.min(particles_traces_y)*0.1, np.max(particles_traces_y)*1.1])

axes.set_xlabel("x position [a.u.]")
axes.set_ylabel("y position [a.u.]")


width = np.max(particles_traces_x) - np.min(particles_traces_x)
height = np.max(particles_traces_y) - np.min(particles_traces_y)


def update(i):
    axes.clear()
    print(i)
    axes.scatter(particles_traces_x[:, i], particles_traces_y[:, i], color="black")

    axes.text(0.95, 0.95, f"step: {i} of {len(particles_traces_x[0])-1}",
        horizontalalignment='right',
        verticalalignment='top',
        transform=axes.transAxes)
    #axes.text(np.min(particles_traces_x), np.max(particles_traces_y), f"setp: {i} of {len(particles_traces_x[0])}")


    axes.set_xlim([np.min(particles_traces_x)-width*0.1, np.max(particles_traces_x)+width*0.1])
    axes.set_ylim([np.min(particles_traces_y)-height*0.1, np.max(particles_traces_y)+height*0.1])
    axes.set_xlabel("x position [a.u.]")
    axes.set_ylabel("y position [a.u.]")



ani = animation.FuncAnimation(fig, update, frames=len(particles_traces_x[0]), interval=0.01)
ani.save('data/n_particle/animation.gif', writer='pillow')
#len(particles_traces_x[0])

print("DONE")
