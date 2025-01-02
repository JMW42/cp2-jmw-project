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
NUMBER_STEPS:int = 50 # number of simulation steps
NUMBER_PARTICLES:int = 100 # number of prticles
EVALUATE_ENSEMBLE:bool = False
BOUNDARY_BOX:list = [40, 40]

# nature constants
kb:float=1 # bolzmann constant

# environment / system constant
T:float = 1 # K, temperature
R:float = 1 # particle radius
friction:float = 1 # friction constant

# interaction parameters:
k_int:float = 100 # interaction constant, spring constant
int_exp:int = 2 # interaction exponential

# lattice parameters:
a1 = np.asarray([np.sqrt(3)/2, -1/2])*2.2 # lattice basis vector
a2 = np.asarray([np.sqrt(3)/2, 1/2])*2.2 # lattice basis vector

# derived variables/constants
dt:float = R**2/(kb*T)/100 # timescale
rng_width:float = np.sqrt(2*friction*kb*T/dt) # width of the normal distributed rng

# further variables:
particles:list = [] # list of all particles
particles_traces_x:list = [[]]*NUMBER_PARTICLES # x component for the traces of all particles
particles_traces_y:list = [[]]*NUMBER_PARTICLES # y component for the traces of all particles

t:np.ndarray = np.arange(dt/100, NUMBER_STEPS*dt, dt) # list of all simulated time values:



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
        particles_traces_x[self.id] = [x0]
        particles_traces_y[self.id] = [y0]

    
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
        return np.sign(self.r)


    # method: move particle
    def move(self, dx:float, dy:float) -> None:
        """ method used to move a given particle"""
        self.x +=dx
        self.y +=dy

        # boundary check left and right boundary
        if np.abs(self.x) > BOUNDARY_BOX[0]/2:
            self.x -= np.sign(self.x)*BOUNDARY_BOX[0]

        # boundary check upper and lower boundary
        if np.abs(self.y) > BOUNDARY_BOX[1]/2:
            self.y -= np.sign(self.y)*BOUNDARY_BOX[1]
        
        # add new position to trace lists
        particles_traces_x[self.id].append(self.x)
        particles_traces_y[self.id].append(self.y)

    
    def move_by_force(self):
        """ updates the particle position based on the stored force values. May the force be with you!!!!"""
        dr = dt/friction*self.F
        self.move(*dr)


# ########################################################################################################################
# ########################################################################################################################
# METHODS:

def rng_normal()->np.ndarray:
    """ rng used for the random kicks by the thermal force."""
    return np.random.normal(scale=rng_width)



def repulsive_force(rA: np.ndarray, rB: np.ndarray) -> np.ndarray:
    """ calculates the repulsive force between two particle position A and B acting on A"""
    F:np.ndarray = np.asarray([0.0, 0.0])

    # 1. calculate vectorial distance and corresponding length
    vec:np.ndarray = rA - rB # distance vector between rA and rB (rB -> rA), acting on rA
    length:np.ndarray = np.linalg.norm(vec, ord=2) # length of interaction vector, distance between mass centers of pA and pB

    vec = vec/length # normalize vector

    # 2. check if positions are within interaction distance:
    if length < 2*R:
        F += (vec/length**int_exp)*k_int # repulsive interaction force
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
        if ((BOUNDARY_BOX[0]/2 - np.abs(p1.x)) < 2*R) or ((BOUNDARY_BOX[1]/2 - np.abs(p1.y)) < 2*R): # check for proximity of boundary:
            
            # 1. calculate trafo vector based on quandrant vectors
            Eab = (p1.Q - p2.Q)/2
            
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



# ########################################################################################################################
# ########################################################################################################################
# MAIN CODE:

plt.ioff()


# create particles:
for i in range(NUMBER_PARTICLES):
    
    n1:int = int(i/np.sqrt(NUMBER_PARTICLES)) - np.sqrt(NUMBER_PARTICLES)/2 # n1 gitter coordinate
    n2:int = int(i%np.sqrt(NUMBER_PARTICLES)) - np.sqrt(NUMBER_PARTICLES)/2 # n2 gitter coordinate

    pos = np.add(np.multiply(a1, n1), np.multiply(a2, n2)) + np.asarray([1,1]) # position in global coordinate grid

    p:Particle2D = Particle2D(*pos, i) # initialise particle
    particles.append(p) # add particle to particle list



# visualize initial condition:
fig, axes = plt.subplots(1,1, figsize=(10, 10))

for p in particles:
    c = plt.Circle((p.x, p.y), R, color="navy")
    axes.add_artist(c)

axes.plot([BOUNDARY_BOX[0]/2, BOUNDARY_BOX[0]/2, -1*BOUNDARY_BOX[0]/2, -1*BOUNDARY_BOX[0]/2, BOUNDARY_BOX[0]/2], [-1*BOUNDARY_BOX[1]/2, BOUNDARY_BOX[1]/2, BOUNDARY_BOX[1]/2, -1*BOUNDARY_BOX[1]/2, -1*BOUNDARY_BOX[1]/2], color="black")

plt.show()


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




# evaluate results:
## show all particle tarces:

fig, axes = plt.subplots(1,1, figsize=(10, 10))

for xarr, yarr in zip(particles_traces_x, particles_traces_y):
    axes.plot(xarr, yarr, ".--", alpha=0.5)
    #axes.plot(xarr[0], yarr[0], "o", alpha=1, color="navy")
    axes.plot(xarr[-1], yarr[-1], "o", alpha=1, color="red")

    c = plt.Circle((xarr[-1], yarr[-1]), R, color="navy")
    axes.add_artist(c)



axes.set_xlim([-1.2*BOUNDARY_BOX[0]/2, 1.2*BOUNDARY_BOX[0]/2])
axes.set_ylim([-1.2*BOUNDARY_BOX[1]/2, 1.2*BOUNDARY_BOX[1]/2])


fig.savefig("data/n_particle/particle_traces.png")
plt.show()



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

