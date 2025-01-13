""" 
CREATED: 08.12.2024
AUTHOR: Jonathan Will
UPDATED: 19.12.2024

Implementation of an n-particle brownian motion simulation.

"""

# ########################################################################################################################
# ########################################################################################################################
# IMPORTS:

import matplotlib.pyplot as plt
import numpy as np
import time

# ########################################################################################################################
# ########################################################################################################################
# GLOBAL VARIABLES:

# simulation related
a:float = 2 # lattice parameter
GRID_SIZE = [20, 10]

NUMBER_STEPS:int = 100 # number of simulation steps
NUMBER_PARTICLES:int = np.prod(GRID_SIZE) # number of prticles, calculated later
EVALUATE_ENSEMBLE:bool = False
BOUNDARY_BOX:list = [GRID_SIZE[0]*a/2, GRID_SIZE[1]*a*np.sqrt(3)] # size of the simulation box, calculated later

ROTATION_ANGLE = np.pi/2
ROTATION_RADIUS = a*3


print(f"NUMBER OF PARTICLES: {NUMBER_PARTICLES}")
print(f"GRID SIZE: {GRID_SIZE[0]} x {GRID_SIZE[1]}")
print(f"ROTATION ANGLE: {ROTATION_ANGLE}")


# nature constants
kb:float=1 # bolzmann constant

# environment / system constant
T:float = 1 # K, temperature
R:float = 1 # particle radius
friction:float = 1 # friction constant

# interaction parameters:
k_int:float = 10 # interaction constant, spring constant

# derived variables/constants
dt:float = R**2/(kb*T)/10000 # timescale
rng_width:float = np.sqrt(2*friction*kb*T/dt) # width of the normal distributed rng
boundary_box_half = BOUNDARY_BOX/2 # half size of the boundary box, max coordinate values

# further variables:
particles:list = [] # list of all particles
particles_traces_x:list = [[]]*NUMBER_PARTICLES # x component for the traces of all particles
particles_traces_y:list = [[]]*NUMBER_PARTICLES # y component for the traces of all particles

t:np.ndarray = np.arange(dt/100, NUMBER_STEPS*dt, dt) # list of all simulated time values:
print(f"NUMBER OF TIMESTEPS: {len(t)}")
print(f'TIME: {t[0]}, .... , {t[-1]}')
print(f'TIMESTEP: {dt}')

avg_move = []

input("continue .....")
# ########################################################################################################################
# ########################################################################################################################
# CLASSES:

class Particle2D:
    """ Particle2D, class used to simulate the particles."""
    def __init__(self, x0:float=0, y0:float=0, id:int=None):
        
        self.r = np.asarray([x0, y0]) # particle position
        self.F = np.asarray([0, 0]) # force acting on particle for a given time step


        # get id for later usage of the trace lists
        self.id:int = id
        if not self.id:self.id = len(particles)

        # append new list to trace list
        try:
            particles_traces_x[self.id] = [x0]
            particles_traces_y[self.id] = [y0]
        except Exception as E:
            print(self.id)
            print(E)
            input("asdw")
    
    # property: x-position
    @property
    def x(self) -> float:
        return self.r[0]
    
    @x.setter
    def x(self, x:float) -> None:
        self.r[0] = x

    # property: y-position
    @property
    def y(self) -> float:
        return self.r[1]
    
    @y.setter
    def y(self, y:float) -> None:
        self.r[1] = y


    @property
    def Q(self) ->np.ndarray:
        """ returns the quadrant vector for a given particle"""
        return np.asarray([msign(self.x), msign(self.y)])


    # method: move particle
    def move(self, dx:float, dy:float) -> None:
        """ method used to move a given particle"""
        self.x +=dx
        self.y +=dy

        # boundary check left and right boundary
        if np.abs(self.x) > boundary_box_half[0]:
            self.x -= msign(self.x)*BOUNDARY_BOX[0]

        # boundary check upper and lower boundary
        if np.abs(self.y) > boundary_box_half[1]:
            self.y -= msign(self.y)*BOUNDARY_BOX[1]
        
        # add new position to trace lists
        particles_traces_x[self.id].append(self.x)
        particles_traces_y[self.id].append(self.y)

    
    def move_by_force(self):
        """ updates the particle position based on the stored force values. May the force be with you!!!!"""
        dr = dt/friction*self.F
        avg_move.append(np.linalg.norm(dr, 2))
        self.move(*dr)


# ########################################################################################################################
# ########################################################################################################################
# METHODS:

def msign(num) -> int:
    """ modified sign function, to get 1 for x >= 0 and -1 for x < 0 """
    return 1 if num >= 0 else -1


def rng_normal()->np.ndarray:
    """ rng used for the random kicks by the thermal force."""
    return np.random.normal(scale=rng_width)



def repulsive_force(rA: np.ndarray, rB: np.ndarray) -> np.ndarray:
    """ calculates the repulsive force between two particle position A and B acting on A"""
    F:np.ndarray = np.asarray([0.0, 0.0])

    # 1. calculate vectorial distance and corresponding length
    vec:np.ndarray = rA - rB # distance vector between rA and rB (rB -> rA), acting on rA
    
    length:float = np.sum(np.abs(vec)) # squared length of the difference between rA and rB
    #length:float = np.sum(np.power(vec, 2))

    # 2. check if positions are within interaction distance:
    if length < 4*R**2:
        length = np.sqrt(length) # propper length calculatiobn, applay square root
        vec /= length # normalize vector

        #F += (vec/length**int_exp)*k_int # repulsive interaction force
        F += (vec)*(2*R - length)*k_int
        # k_int: interaction coeficient
        # int_exp: interaction exponential
    
    return F



def external_force_on_particle(p1:Particle2D, t:float)->np.ndarray:
    """ external force, defined by position and time"""
    return np.array([0, 0])



def thermal_force_on_particle(p1:Particle2D, t:float)->np.ndarray:
    """ thermal force, simulates the random kicks, done for brownian motion."""
    return (friction * kb*T)*np.array([rng_normal(), rng_normal()])



def repulsive_force_on_particle(p1:Particle2D):
    """ Method for calculating the repulsve force a given particle p1 experiences. Takes normal interaction withing abd over boundaries into account"""
    F:np.ndarray = np.asarray([0.0, 0.0])

    # standart repulsive force for particle p1 interactions with p2

    for p2 in particles:
        if p2 is p1: continue # ignore, because p2 is self

        # casual interaction within boundaries
        F += repulsive_force(p1.r, p2.r)

        # boundary interaction:
        if ((boundary_box_half[0] - np.abs(p1.x)) < 2*R) or ((boundary_box_half[1] - np.abs(p1.y)) < 2*R): # check for proximity of boundary:
            
            # 1. calculate trafo vector based on quandrant vectors
            #Eab:np.ndarray = np.linalg.norm((p1.Q - p2.Q), 2)
            Eab:np.ndarray = (p1.Q - p2.Q)/2

            # check if interaction with particle within same quadrant
            if np.sum(np.abs(Eab)) == 0: continue

            # 2. calculate virtual position for interaction over boundary:
            r2:np.ndarray = np.asarray([p2.x, p2.y]) + np.multiply(Eab, BOUNDARY_BOX) # position of virtual boundary particle

            # interaction over the boundary (with virtual/translated particles)
            F += repulsive_force(p1.r, r2)

    return F



def borwnian_move(p:Particle2D, x:float, y:float, t:float)->np.ndarray:
    """ DEPRECATED: method to calculate the change of a particles position done by a brownian movement."""
    Fex:np.ndarray = external_force_on_particle(x, y, t)  # exteenal additional force
    Ft:np.ndarray = thermal_force_on_particle(x, y, t) # thermal force, borwnian motion
    Fint:np.ndarray = repulsive_force_on_particle(p) # repulsive_interaction(x, y, t) # interactin force between particles
    
    return dt/friction*(Fex + Ft + Fint)



def calculate_force_on_particle(p:Particle2D, t:float)->np.ndarray:
    """ calculates the sum of forces acting on a given particle p at a given time t."""
    F = np.asanyarray([0.0, 0.0])

    F += external_force_on_particle(p, t)  # exteenal additional force
    F += thermal_force_on_particle(p, t) # thermal force, borwnian motion
    F += repulsive_force_on_particle(p)

    return F


def rotate_point(x, y, theta) -> tuple:
    """ Function to rotate a given vector with x and y counter clockwise by theta in rad"""
    xnew = x*np.cos(theta) - y*np.sin(theta)
    ynew = x*np.sin(theta) + y*np.cos(theta)

    return (xnew, ynew)



def visualize_simulation(savefilepath=None):
    """ visualizes/plots the current state of the simulation"""
    fig, axes = plt.subplots(1,1, figsize=(int(GRID_SIZE[0]/2), int(GRID_SIZE[1]*np.sqrt(3))))

    # draw particles and virtual particles
    for p in particles:
        
        # original particles
        c = plt.Circle((p.x, p.y), R, facecolor="navy", edgecolor="black", linestyle="-", linewidth=2)
        axes.add_artist(c)

        # particle traces:
        #axes.plot(particles_traces_x[p.id], particles_traces_y[p.id], ".--", alpha=0.5) # trace

        # virtual particles
        for E in [[1,0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]:

            pos = p.r + np.multiply(E, BOUNDARY_BOX)
            c = plt.Circle((pos[0], pos[1]), R, facecolor="orange", edgecolor="black", linestyle="-", linewidth=2)
            axes.add_artist(c)

    axes.plot([BOUNDARY_BOX[0]/2, BOUNDARY_BOX[0]/2, -1*BOUNDARY_BOX[0]/2, -1*BOUNDARY_BOX[0]/2, BOUNDARY_BOX[0]/2], [-1*BOUNDARY_BOX[1]/2, BOUNDARY_BOX[1]/2, BOUNDARY_BOX[1]/2, -1*BOUNDARY_BOX[1]/2, -1*BOUNDARY_BOX[1]/2], color="black")

    axes.set_xlim([-1.6*BOUNDARY_BOX[0], 1.6*BOUNDARY_BOX[0]])
    axes.set_ylim([-1.6*BOUNDARY_BOX[1], 1.6*BOUNDARY_BOX[1]])

    if savefilepath:
        fig.savefig(savefilepath)

    plt.show()



# ########################################################################################################################
# ########################################################################################################################
# MAIN CODE:


# initialize and place particels on hexgri
for nx in range(0, GRID_SIZE[0]):
    for ny in range(0, GRID_SIZE[1]):

        px:float = a/2*(nx - GRID_SIZE[0]/2) + a/1000
        py:float = a*np.sqrt(3)*(ny - GRID_SIZE[1]/2) + a/1000

        if nx % 2 == 0:
            py += a*np.sqrt(3)/2


        # check if rotation trafo for grain applies:
        if np.sqrt(px**2 + py**2) < ROTATION_RADIUS:
            px, py = rotate_point(px, py, ROTATION_ANGLE)
        

        p:Particle2D = Particle2D(px, py) # initialise particle
        particles.append(p) # add particle to particle list




# visualize initial condition:
visualize_simulation("data/n_particle/particle_initial.png")


# simulation:
time_start = time.time()
for i in range(1, NUMBER_STEPS):
    print(f'{i}') # print current simulation step

    #capture_particle_snapshot(f"{i}.png")

    print("calculating forces")
    for p in particles:
        #p.move(*borwnian_move(p, p.x, p.y, t))
        p.F = calculate_force_on_particle(p, t) # calculate the force on given particle


    print("performing position updates")
    for p in particles:
        p.move_by_force()


time_end = time.time()
print(f'time elapsed: {time_end - time_start}sec.')



print(f"mean movement: {np.mean(avg_move)}")
print(f"max movement: {np.max(avg_move)}")
print(f"min movement: {np.min(avg_move)}")


# visualice results:
visualize_simulation("data/n_particle/particle_traces.png")


# SAVE PARTICLE TRACES:

#print(particles_traces_x)
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

    fig.savefig("data/n_particle/r2-ensemble.png")
    plt.show()



    fig, axes = plt.subplots(1,1, figsize=(10, 10))

    axes.plot(t[1:], Darr[1:], label="<D(t)>", color="navy")

    axes.set_ylim([0, 2])
    axes.set_ylabel("<D(t)>")
    axes.set_xlabel("Time t [a.u.]")
    axes.grid()

    fig.savefig("data/n_particle/D-ensemble.png")

    plt.show()

