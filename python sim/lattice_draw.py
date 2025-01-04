import numpy as np
import matplotlib.pyplot as plt

a1 = np.asarray([np.sqrt(3)/2, -1/2])
a2 = np.asarray([np.sqrt(3)/2, 1/2])

#a1 = np.asarray([1, 0])
#a2 = np.asarray([0, 1])

BOUNDARY_BOX:list = [40, 40]
NUMBER_PARTICLES = 300


# further variables
lat_xarr = []
lat_yarr = []


""" 

i = 0
j = 0
while True:
    if (j >= NUMBER_PARTICLES) or i > (10*NUMBER_PARTICLES):
        break
    
    n1:int = int(i/np.sqrt(NUMBER_PARTICLES)) - np.sqrt(NUMBER_PARTICLES)/2 # n1 gitter coordinate # collumn number
    n2:int = int(i%np.sqrt(NUMBER_PARTICLES)) - np.sqrt(NUMBER_PARTICLES)/2

    print(i, n1, n2)

    pos = np.add(np.multiply(a1, n1), np.multiply(a2, n2)) +a1/4 + a2/4 # position in global coordinate grid

    i += 1
    if (np.abs(pos[0]) > 0.98*BOUNDARY_BOX[0]/2) or (np.abs(pos[1]) > 0.98*BOUNDARY_BOX[1]/2):
        print("skip")
        continue
    else:
        lat_xarr.append(pos[0])
        lat_yarr.append(pos[1])
        j += 1
        

    print(i, j)
    print(f'lattce points within box: {len(lat_xarr)}')
"""
    
R = 1
posarr = []
a = 2.5


countx = int(BOUNDARY_BOX[0] / a)
county = int(BOUNDARY_BOX[0] / a)

for x in range(-int(countx/2), int(countx/2)):
    for y in range(-int(county/2), int(county/2)):

        px = x*a + a*0.01
        py = y*a + a*0.01

        if x % 2 == 0:
            py += a/2

        dx = a/2
        dy = dx
        if (np.abs(px) >= BOUNDARY_BOX[0]/2 - dx) or (np.abs(py) >= BOUNDARY_BOX[1]/2 - dy):
            continue
        else:
            posarr.append([px, py])


print(f'total: {len(posarr)} particles within pox')



fig, axes = plt.subplots(1,1, figsize=(10, 10))

#axes.scatter(lat_xarr, lat_yarr, s=40)

for pos in posarr:
    c = plt.Circle((pos[0], pos[1]), R, color="navy")
    axes.add_artist(c)


axes.plot([BOUNDARY_BOX[0]/2, BOUNDARY_BOX[0]/2, -1*BOUNDARY_BOX[0]/2, -1*BOUNDARY_BOX[0]/2, BOUNDARY_BOX[0]/2], [-1*BOUNDARY_BOX[1]/2, BOUNDARY_BOX[1]/2, BOUNDARY_BOX[1]/2, -1*BOUNDARY_BOX[1]/2, -1*BOUNDARY_BOX[1]/2], color="black")


axes.set_xlabel("x position in [a.u.]")
axes.set_ylabel("y position in [a.u.]")
axes.grid()


fig.savefig("data/n_particle/simbox.png")


plt.show()
