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



R = 1


# simulation related
LATTICE_PARAMETER:float = 2 # lattice parameter
GRID_SIZE = [20, 10]

NUMBER_STEPS:int = 10 # number of simulation steps
NUMBER_PARTICLES:int = np.prod(GRID_SIZE) # number of prticles, calculated later
EVALUATE_ENSEMBLE:bool = False
BOUNDARY_BOX:list = [GRID_SIZE[0]*LATTICE_PARAMETER/2, GRID_SIZE[1]*LATTICE_PARAMETER*np.sqrt(3)] # size of the simulation box, calculated later

# rotation:
ROTATION_ANGLE = np.pi/2*0
ROTATION_RADIUS = LATTICE_PARAMETER*3

# interaction parameters:
k_int:float = 50 # interaction constant, spring constant


dirname = f"data/n_particle/k={k_int}_a={str(LATTICE_PARAMETER).replace('.', ',')}_STEPS={str(NUMBER_STEPS)}_R={str(ROTATION_RADIUS)}_THETA={str(ROTATION_ANGLE).replace('.', ',')}"
#os.makedirs(dirname, exist_ok=True)

x_trace_file = f"{dirname}/x_traces.txt"
y_trace_file = f"{dirname}/y_traces.txt"

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

fig.savefig(f"{dirname}/particle_traces_render.png")
plt.show()



time_start = time.time()

fig, axes = plt.subplots(1,1, figsize=(int(GRID_SIZE[0]/2), int(GRID_SIZE[1]*np.sqrt(3))))

axes.set_xlim([-1.2*BOUNDARY_BOX[0], 1.2*BOUNDARY_BOX[0]])
axes.set_ylim([-1.2*BOUNDARY_BOX[1], 1.2*BOUNDARY_BOX[1]])

axes.set_xlabel("X position [a.u.]", fontsize=20)
axes.set_ylabel("Y position [a.u.]", fontsize=20)


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
    axes.set_xlim([-1.1*BOUNDARY_BOX[0] - 5*R, 1.1*BOUNDARY_BOX[0] + 5*R])
    axes.set_ylim([-1.1*BOUNDARY_BOX[1] - 5*R, 1.1*BOUNDARY_BOX[1] + 5*R])

    axes.set_xlabel("X position [a.u.]", fontsize=20)
    axes.set_ylabel("Y position [a.u.]", fontsize=20)



ani = animation.FuncAnimation(fig, update, frames=len(particles_traces_x[0])+15, interval=1)
ani.save(f'{dirname}/animation.gif', writer='pillow', fps=2, bitrate=1, dpi=50)
#len(particles_traces_x[0])


time_end = time.time()
dt = time_end - time_start
print(f" tooK : {int(dt/3600)%60} hours. {int(dt/60)%60} min. {dt%60} sec.")

print("DONE")

