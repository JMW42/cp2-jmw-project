import numpy as np
import matplotlib.pyplot as plt


kb=1
T = 1
friction = 1


hist = []



def rng_normal2():
    res = 0

    for i in range(20):
        res += np.random.randint(-1, 1)

    return res

    
    #


def rng_normal():
    return np.random.normal()


for i in range(10000):
    hist.append(rng_normal())


def thermal_force(x:float, y:float, t:float):
    """ thermal force, simulates the random kicks, done for brownian motion."""
    return np.array([rng_normal(), rng_normal()])



# visualice:
fig, axes = plt.subplots(1,1, figsize=(10, 5))

axes.hist(hist, bins=100)

plt.show()



# evaluate rng force
farr = []
for i in range(10000):
    f = thermal_force(0, 0, 0)
    farr.append(np.dot(f, f))


# second moment of force:
fsm=np.mean(farr)

dt = 4*friction * kb * T / fsm

print(f'dt={dt}')