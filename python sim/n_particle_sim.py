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
NUMBER_PARTICLES:int = 14 # number of prticles
EVALUATE_ENSEMBLE:bool = False
BOUNDARY_BOX:list = [40, 40]

# nature constants
kb:float=1 # bolzmann constant

# environment / system constant
T:float = 1 # K, temperature
R:float = 1 # particle radius
friction:float = 1 # friction constant

# interaction parameters:
k_int:float = 1000 # interaction constant, spring constant

# lattice parameters:
a1 = np.asarray([0, 1])*4 # lattice basis vector
a2 = np.asarray([1, 0])*4 # lattice basis vector

# derived variables/constants
dt:float = R**2/(kb*T)/100 # timescale
rng_width:float = np.sqrt(2*friction*kb*T/dt) # width of the normal distributed rng

# further variables:
particles:list = [] # list of all particles
particles_traces_x:list = [[]]*NUMBER_PARTICLES # x component for the traces of all particles
particles_traces_y:list = [[]]*NUMBER_PARTICLES # y component for the traces of all particles

t:np.ndarray = np.arange(dt/100, NUMBER_STEPS*dt, dt) # list of all simulated time values:

current_repulsive_force_x = np.zeros((NUMBER_PARTICLES, NUMBER_PARTICLES)) # matrix for temporal storing the force acting between individual particles
current_repulsive_force_y = np.zeros((NUMBER_PARTICLES, NUMBER_PARTICLES))

print(current_repulsive_force_x)



# ########################################################################################################################
# ########################################################################################################################
# CLASSES:

class Particle2D:
    """ Particle2D, class used to simulate the particles."""
    def __init__(self, x0:float=0, y0:float=0, id:int=None):
        
        self.r = np.asarray([x0, y0])

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


# ########################################################################################################################
# ########################################################################################################################
# METHODS:

def rng_normal()->np.ndarray:
    """ rng used for the random kicks by the thermal force."""
    return np.random.normal(scale=rng_width)


def external_force(x:float, y:float, t:float)->np.ndarray:
    """ external force, defined by position and time"""
    return np.array([0, 0])


def thermal_force(x:float, y:float, t:float)->np.ndarray:
    """ thermal force, simulates the random kicks, done for brownian motion."""
    return (friction * kb*T)*np.array([rng_normal(), rng_normal()])


def repulsive_force(p1:Particle2D):
    """ Method for calculating the repulsve force a given particle p1 experiences. Takes normal interaction withing abd over boundaries into account"""
    F:np.ndarray = np.asarray([0.0, 0.0])

    # standart repulsive force for particle p1 interactions with p2

    for p2 in particles:
        if p2 is p1: continue # ignore, because p2 is self

        # casual repulsive interaction interaction based on inverse hookes law
        vec:np.ndarray = p1.r - p2.r # distance vector between p1 and p2 (p2 -> p1)
        length:np.ndarray = np.linalg.norm(vec, ord=2) # length of interaction vector, distance between mass centers of p1 and p2

        vec = vec/length # normalize vector

        # check if interaction is happening:
        if length < 2*R:
            F += (vec/length)*k_int # repulsive interaction force
        

        # boundary interaction:
        if ((BOUNDARY_BOX[0]/2 - np.abs(p1.x)) < 2*R) or ((BOUNDARY_BOX[1]/2 - np.abs(p1.y)) < 2*R): # check for proximity of boundary:
            
            # 1. calculate trafo vector based on quandrant vectors
            Eab = (p1.Q - p2.Q)/2
            
            # 2. calculate virtual position for interaction over boundary:
            r2:np.ndarray = np.asarray([p2.x, p2.y]) + np.multiply(Eab, BOUNDARY_BOX) # position of virtual boundary particle
            vec2:np.ndarray = p1.r - r2 # interaction vector for virtual boundary particle
            length2:np.ndarray = np.linalg.norm(vec2, ord=2) # length of interaction vector for virtual boundary particle


            if length2 < 2*R:
                F += vec2/length2*k_int
                print("BOUNDARY INTERACTION!!!", p1.x, p1.y)
                    
    return F


def repulsive_interaction(x: float, y:float, t:float) -> np.ndarray:
    # deprecated, will be removed soon!
    """ repulsive force, following an inverse hooks law if the particles overlapp for the given position and time"""
    F = np.asarray([0.0,0.0])

    for p in particles:
        # standart repulsice interaction:
        if np.abs((x-p.x)) < R and (np.abs((y-p.y)) <R):
            if (x == p.x) and (y == p.y): continue

            d = np.sqrt((x-p.x)**2 + (y-p.y)**2) # distance between particles
            vec = np.asarray([(x-p.x), (y-p.y)])/d # interaction vector
            F += vec*(1/d)*k_int
            #print("C", x, y)

        # boundary interaction:

        ## 1. check if close to boundary:
        if ((BOUNDARY_BOX[0]/2 - np.abs(x)) < 2*R) or ((BOUNDARY_BOX[1]/2 - np.abs(y)) < 2*R):

            # 2. calculate trafo vector based on quandrant vectors
            Qa = np.asarray([np.sign(x), np.sign(y)])
            Qb = np.asarray([np.sign(p.x), np.sign(p.y)])
            Eab = np.subtract(Qa, Qb)/2

            # 3. calculate virtual position for interaction over boundary:
            rb = np.asarray([p.x, p.y]) + np.multiply(Eab, BOUNDARY_BOX)


            if np.abs((x-rb[0])) < 2*R and (np.abs((y-rb[1])) < 2*R):
                d = np.sqrt((x-rb[0])**2 + (y-rb[1])**2)
                vec = np.asarray([(x-rb[0]), (y-rb[1])])/d
                F += vec*(1/d)*k_int
                print("boundary interaction")
                for i in range(1):
                    print("BOUNDARY INTERACTION!!!", x, y)
                    


        
    return F



def borwnian_move(p:Particle2D, x:float, y:float, t:float)->np.ndarray:
    """ method to calculate the change of a particles position done by a brownian movement."""
    Fex:np.ndarray = external_force(x, y, t)  # exteenal additional force
    Ft:np.ndarray = thermal_force(x, y, t) # thermal force, borwnian motion
    Fint:np.ndarray = repulsive_force(p) # repulsive_interaction(x, y, t) # interactin force between particles
    
    return dt/friction*(Fex + Ft + Fint)


def capture_particle_snapshot(name):
    """ Draws the current position of the particles as matplotlib plot"""

    fig, axes = plt.subplots(1,1, figsize=(10, 10))

    for p in particles:
        axes.plot(p.x, p.y, "o", alpha=0.5, color="black")

    plt.savefig(f'data/n_particle/snapshot/{name}')
    plt.close(fig)


# ########################################################################################################################
# ########################################################################################################################
# MAIN CODE:

plt.ioff()


# create particles:
#for i in range(NUMBER_PARTICLES):
#    
#    n1:int = int(i/np.sqrt(NUMBER_PARTICLES)) - np.sqrt(NUMBER_PARTICLES)/2 # n1 gitter coordinate
#    n2:int = int(i%np.sqrt(NUMBER_PARTICLES)) - np.sqrt(NUMBER_PARTICLES)/2 # n2 gitter coordinate
#
#    pos = np.add(np.multiply(a1, n1), np.multiply(a2, n2)) # position in global coordinate grid
#
#    p:Particle2D = Particle2D(*pos, i) # initialise particle
#    particles.append(p) # add particle to particle list


for i in np.linspace(-10, 10, 4):


    p:Particle2D = Particle2D(19, i) # initialise particle
    particles.append(p)

    p:Particle2D = Particle2D(-19, i) # initialise particle
    particles.append(p)

    p:Particle2D = Particle2D(i, 19) # initialise particle
    particles.append(p)

    #p:Particle2D = Particle2D(i, -19) # initialise particle
    #particles.append(p)

p:Particle2D = Particle2D(2, 6.1) # initialise particle
particles.append(p)

p:Particle2D = Particle2D(2, 5) # initialise particle
particles.append(p)


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

    for p in particles:
        p.move(*borwnian_move(p, p.x, p.y, t))


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

