import numpy as np
import matplotlib.pyplot as plt

a1 = np.asarray([np.sqrt(3)/2, -1/2])
a2 = np.asarray([np.sqrt(3)/2, 1/2])

#a1 = np.asarray([1, 0])
#a2 = np.asarray([0, 1])

BOUNDARY_BOX:list = [10, 10]
NUMBER_PARTICLES = 300


# further variables
lat_xarr = []
lat_yarr = []


for i in range(NUMBER_PARTICLES):
    n1:int = int(i/np.sqrt(NUMBER_PARTICLES)) - np.sqrt(NUMBER_PARTICLES)/2 # n1 gitter coordinate # collumn number
    n2:int = int(i%np.sqrt(NUMBER_PARTICLES)) - np.sqrt(NUMBER_PARTICLES)/2 # n2 gitter coordinate # row number

    pos = np.add(np.multiply(a1, n1), np.multiply(a2, n2)) # position in global coordinate grid

    if not ( (np.abs(pos[0]) > BOUNDARY_BOX[0]/2) or (np.abs(pos[1]) > BOUNDARY_BOX[1]/2)):
        lat_xarr.append(pos[0])
        lat_yarr.append(pos[1])

print(f'lattce points within box: {len(lat_xarr)}')


fig, axes = plt.subplots(1,1, figsize=(10, 10))

axes.scatter(lat_xarr, lat_yarr, s=40)

axes.plot([BOUNDARY_BOX[0]/2, BOUNDARY_BOX[0]/2, -1*BOUNDARY_BOX[0]/2, -1*BOUNDARY_BOX[0]/2, BOUNDARY_BOX[0]/2], [-1*BOUNDARY_BOX[1]/2, BOUNDARY_BOX[1]/2, BOUNDARY_BOX[1]/2, -1*BOUNDARY_BOX[1]/2, -1*BOUNDARY_BOX[1]/2], color="black")

axes.grid()

axes.set_xlim([-10, 10])
axes.set_ylim([-10, 10])

plt.show()
