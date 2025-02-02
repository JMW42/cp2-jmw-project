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
import threading
import time

# ########################################################################################################################
# ########################################################################################################################
# GLOBAL VARIABLES:

# simulation related
a:float = 2 # lattice parameter
GRID_SIZE = [20, 10]

NUMBER_STEPS:int = 10 # number of simulation steps
NUMBER_PARTICLES:int = np.prod(GRID_SIZE) # number of prticles, calculated later
EVALUATE_ENSEMBLE:bool = False
BOUNDARY_BOX:list = [GRID_SIZE[0]*a/2, GRID_SIZE[1]*a*np.sqrt(3)] # size of the simulation box, calculated later

# rotation:
ROTATION_ANGLE = np.pi/2*0
ROTATION_RADIUS = a*3



# threading:
NUMBER_THREADS = 2


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
boundary_box_half = np.divide(BOUNDARY_BOX, 2) # half size of the boundary box, max coordinate values

# further variables:
particles:list = [] # list of all particles
particles_traces_x:list = [[]]*NUMBER_PARTICLES # x component for the traces of all particles
particles_traces_y:list = [[]]*NUMBER_PARTICLES # y component for the traces of all particles

tarr:np.ndarray = np.arange(dt/100, NUMBER_STEPS*dt, dt) # list of all simulated time values:
print(f"NUMBER OF TIMESTEPS: {len(tarr)}")
print(f'TIME: {tarr[0]}, .... , {tarr[-1]}')
print(f'TIMESTEP: {dt}')


# threading
threads = []
flag_threads_start = False
flag_threads_clculate_force = False
flag_threads_calculate_position = False

n = 0 # simulation step


# tmp:
avg_moves = []



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

        avg_moves.append(np.sqrt(np.sum(np.power([dx, dy], 2))))

    
    def move_by_force(self):
        """ updates the particle position based on the stored force values. May the force be with you!!!!"""
        dr = dt/friction*self.F
        self.move(*dr)


class LoggingThread(threading.Thread):
    def __init__(self, group = None, target = None, name = None, args = ..., kwargs = None, *, daemon = None):
        print("Initializing logging thread")
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self._queue = []
        self._running = True

    def log(self, msg):
        self._queue.append(msg)

    
    def close(self):
        self._running = False


    def run(self):
        print("Starting logging thread")

        while self._running:
            if len(self._queue) > 0:
                print(self._queue[0])
                self._queue.pop(0)
        
        print("Closing logging thread")





class SimulationThread(threading.Thread):
    def __init__(self, id:int, index_start:int, index_end:int, group = None, target = None, name = None, args = ..., kwargs = None, *, daemon = None):
        super().__init__(group, target, name, args, kwargs, daemon=daemon)

        print(f"initializing thread: {id} - particles: {index_start} to {index_end}")

        self.id = id
        self.index_start = index_start
        self.index_end = index_end

        self.finished_calculation_forces = False
        self.finished_calculation_positions = False
        self.finished_all = False
    

    def run(self):
        logging.log(f"starting thread: {self.id}")

        # initial wait:
        logging.log(f" t{self.id}: waiting")
        while not flag_threads_start:
            pass
        

        while n < NUMBER_STEPS:

            # wait for force calculation flag:
            while not flag_threads_clculate_force:

                # if n is to late updated, capture it by break
                if n >= NUMBER_STEPS:
                    break
            
            # if n is to late updated, capture it by break
            if n >= NUMBER_STEPS:
                    break

            self.finished_calculation_positions = False # update flag

            logging.log(f" {self.id} step: {n}")


            # calculate forces:
            logging.log(f" {self.id} calculating forces")
            for i in range(self.index_start, self.index_end+1):
                particles[i].F = calculate_force_on_particle(particles[i], tarr[n])

            self.finished_calculation_forces = True # force calculation finished

            # wait for position update flag:
            while not flag_threads_calculate_position:
                pass
            
            self.finished_calculation_forces = False # update flag

            # update positions
            logging.log(f" {self.id} calculating positions")
            for i in range(self.index_start, self.index_end+1):
                particles[i].move_by_force()

            self.finished_calculation_positions = True # position updates done

        logging.log(f"closing thread: {self.id}")
        self.finished_all = True


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
    fig, axes = plt.subplots(1,1, figsize=(10, 10))

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
            if (np.abs(pos[0]) < BOUNDARY_BOX[0]/2 + 4.5*R) and (np.abs(pos[1]) < BOUNDARY_BOX[1]/2 + 4.5*R):
                c = plt.Circle((pos[0], pos[1]), R, facecolor="orange", edgecolor="black", linestyle="-", linewidth=2)
                axes.add_artist(c)

    axes.plot([BOUNDARY_BOX[0]/2, BOUNDARY_BOX[0]/2, -1*BOUNDARY_BOX[0]/2, -1*BOUNDARY_BOX[0]/2, BOUNDARY_BOX[0]/2], [-1*BOUNDARY_BOX[1]/2, BOUNDARY_BOX[1]/2, BOUNDARY_BOX[1]/2, -1*BOUNDARY_BOX[1]/2, -1*BOUNDARY_BOX[1]/2], color="black")

    axes.set_xlim([-BOUNDARY_BOX[0]/2-5*R, BOUNDARY_BOX[0]/2+5*R])
    axes.set_ylim([-BOUNDARY_BOX[1]/2-5*R, BOUNDARY_BOX[1]/2+5*R])

    axes.set_xlabel("X position [a.u.]", fontsize=20)
    axes.set_ylabel("Y position [a.u.]", fontsize=20)


    if savefilepath:
        fig.savefig(savefilepath)

    plt.show()



def get_threads_finished_all():
    status = []
    for t in threads:
        status.append(t.finished_all)

    return status


def get_threads_finished_calculating_forces():
    status = []
    for t in threads:
        status.append(t.finished_calculation_forces)

    return status


def get_threads_finished_calculating_position():
    status = []
    for t in threads:
        status.append(t.finished_calculation_positions)

    return status

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


# initialize and start logging thread
logging = LoggingThread()
logging.start()

time.sleep(1)

# initialize threads
for i in range(NUMBER_THREADS):
    i0 = int(np.multiply(*GRID_SIZE)/NUMBER_THREADS)*i
    i1 = int(np.multiply(*GRID_SIZE)/NUMBER_THREADS)*(i+1)-1
    thread = SimulationThread(i, i0, i1)
    threads.append(thread)


# start threads:
for t in threads:
    t.start()

# start with simulating:
logging.log("starting simulation")
flag_threads_start = True
time.sleep(1)


# simulation:
while n < NUMBER_STEPS:

    logging.log(f"-"*60 + f"> STEP: {n}")
    logging.log(f"FORCE CALCULATION: {n}")
    flag_threads_clculate_force = True # allow threads to calculate forces now

    # calculate forces, wait for threads
    while not np.all(get_threads_finished_calculating_forces()):
        pass

    flag_threads_clculate_force = False # so all threads will wait in the next iteration

    logging.log(f"POSITION UPDATE: {n}")
    flag_threads_calculate_position = True # allow threads to update positions

    # calculate positions, wait for threads
    while not np.all(get_threads_finished_calculating_position()):
        pass
    
    flag_threads_calculate_position = False # so all threads will wait in the next iteration


    # done itteration:
    n += 1



# waiting forthreads to close:
while not np.all(get_threads_finished_all()):
    pass


# simulation finished:
logging.log("done simulation")

time.sleep(1)

# close logging thread:
logging.close()

time.sleep(1)

time_end = time.time()
dt = time_end - time_start
print(f" time elapsed : {int(dt/3600)%60} hours. {int(dt/60)%60} min. {dt%60} sec.")


# visualice results:
#visualize_simulation("data/n_particle/particle_traces.png")

# SAVE PARTICLE TRACES:

print("saving data ....")
np.savetxt("data/n_particle/x_traces.txt", particles_traces_x)
np.savetxt("data/n_particle/y_traces.txt", particles_traces_y)
print("... saved")


print(f" avg. movement: {np.mean(avg_moves)} +/- {np.std(avg_moves)}")
print(f" max movement: {np.max(avg_moves)}")
print(f" min movement: {np.min(avg_moves)}")

print(f"DONE ...")