""" 
CREATED: 08.12.2024
AUTHOR: Jonathan Will
UPDATED: 19.12.2024

Python script to render videos of particle traces with given matrix files creates by single_particle and n_particle simulations.

"""

# ########################################################################################################################
# ########################################################################################################################
# IMPORTS:

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import itertools
import time

# ########################################################################################################################
# ########################################################################################################################
# GLOBAL VARIABLES:

x_trace_file = "data/n_particle/x_traces.txt"
y_trace_file = "data/n_particle/y_traces.txt"

R = 1


# simulation related
a:float = 2 # lattice parameter
GRID_SIZE = [20, 10]

NUMBER_STEPS:int = 100 # number of simulation steps 
NUMBER_PARTICLES:int = np.prod(GRID_SIZE) # number of prticles, calculated later
EVALUATE_ENSEMBLE:bool = False
BOUNDARY_BOX:list = [GRID_SIZE[0]*a/2, GRID_SIZE[1]*a*np.sqrt(3)] # size of the simulation box, calculated later




# ########################################################################################################################
# ########################################################################################################################
# MAIN CODE:


particles_traces_x = np.loadtxt(x_trace_file) # x component for the traces of all particles
particles_traces_y = np.loadtxt(y_trace_file) # y component for the traces of all particles



# visualice all traces:
fig, axes = plt.subplots(1,1, figsize=(int(GRID_SIZE[0]/2), int(GRID_SIZE[1]*np.sqrt(3))))

for xarr, yarr in zip(particles_traces_x, particles_traces_y):
    axes.plot(xarr, yarr, ".--", alpha=0.5)
    axes.plot(xarr[0], yarr[0], "o", alpha=1, color="navy")
    axes.plot(xarr[-1], yarr[-1], "o", alpha=1, color="red")

fig.savefig("data/n_particle/particle_traces_render.png")
plt.show()



time_start = time.time()

fig, axes = plt.subplots(1,1, figsize=(int(GRID_SIZE[0]/2), int(GRID_SIZE[1]*np.sqrt(3))))

axes.set_xlim([-1.2*BOUNDARY_BOX[0], 1.2*BOUNDARY_BOX[0]])
axes.set_ylim([-1.2*BOUNDARY_BOX[1], 1.2*BOUNDARY_BOX[1]])

axes.set_xlabel("x position [a.u.]")
axes.set_ylabel("y position [a.u.]")


width = np.max(particles_traces_x) - np.min(particles_traces_x)
height = np.max(particles_traces_y) - np.min(particles_traces_y)


def update(i):
    axes.clear()

    j = i-15
    if j < 0:
        j = 0


    print(j)

    #axes.scatter(particles_traces_x[:, i], particles_traces_y[:, i], color="navy")

    for x, y in zip(particles_traces_x[:, j], particles_traces_y[:, j]):

        # draw particles
        c = plt.Circle((x, y), R, facecolor="navy", edgecolor="black", linestyle="-", linewidth=2)
        #axes.set_aspect( 1 )
        axes.add_artist(c)

        # calculate virtual particles (copies outside of box)
        for Eab in [[1,0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]:
            rb = np.asarray([x, y]) + np.multiply(Eab, BOUNDARY_BOX)

            if (np.abs(rb[0]) < BOUNDARY_BOX[0]/2 + 4.5*R) and (np.abs(rb[1]) < BOUNDARY_BOX[1]/2 + 4.5*R):
                # only if close enought to boundary draw particle
                c = plt.Circle((rb[0], rb[1]), R, facecolor="orange", edgecolor="black", linestyle="-", linewidth=2)
                axes.add_artist(c)


    axes.text(0.95, 0.95, f"step: {i-15} of {len(particles_traces_x[0])-1}",
        horizontalalignment='right',
        verticalalignment='top',
        transform=axes.transAxes)
    #axes.text(np.min(particles_traces_x), np.max(particles_traces_y), f"setp: {i} of {len(particles_traces_x[0])}")

    # draw virtual particles:

    
    


    axes.plot([BOUNDARY_BOX[0]/2, BOUNDARY_BOX[0]/2, -1*BOUNDARY_BOX[0]/2, -1*BOUNDARY_BOX[0]/2, BOUNDARY_BOX[0]/2], [-1*BOUNDARY_BOX[1]/2, BOUNDARY_BOX[1]/2, BOUNDARY_BOX[1]/2, -1*BOUNDARY_BOX[1]/2, -1*BOUNDARY_BOX[1]/2], color="black")

    #axes.set_xlim([np.min(particles_traces_x)-width*0.1, np.max(particles_traces_x)+width*0.1])
    #axes.set_ylim([np.min(particles_traces_y)-height*0.1, np.max(particles_traces_y)+height*0.1])
    axes.set_xlim([-1.2*BOUNDARY_BOX[0], 1.2*BOUNDARY_BOX[0]])
    axes.set_ylim([-1.2*BOUNDARY_BOX[1], 1.2*BOUNDARY_BOX[1]])
    axes.set_xlabel("x position [a.u.]")
    axes.set_ylabel("y position [a.u.]")



ani = animation.FuncAnimation(fig, update, frames=len(particles_traces_x[0])+15, interval=150)
ani.save('data/n_particle/animation.gif', writer='pillow')
#len(particles_traces_x[0])


time_end = time.time()
print(f" tooK : {time_end - time_start} sec.")

print("DONE")

