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
# lattice parameter
a:float = 2.1

# calculate PARTICLE NUMBER:
# calculate particle positions
posarr:list = []
countx:int = int(BOUNDARY_BOX[0] / a*2)
county:int = int(BOUNDARY_BOX[0] / a*2)

for nx in range(-int(countx/2), int(countx/2) + 1):
    for ny in range(-int(county/2), int(county/2)):

        px:float = nx*a/2 + a*0.01
        py:float = ny*a*np.sqrt(3) + a*0.01

        if nx % 2 == 0:
            py += a*np.sqrt(3)/2
            pass

        dx = a/6
        dy = dx
        if (np.abs(px - dx) >= np.abs( BOUNDARY_BOX[0]/2)) or (np.abs(py - dy) >= np.abs(BOUNDARY_BOX[1]/2)):
            continue
        else:
            posarr.append([px, py])


print(f'total: {len(posarr)} particles within pox')



fig, axes = plt.subplots(1,1, figsize=(10, 10))

#axes.scatter(lat_xarr, lat_yarr, s=40)

for pos in posarr:
    c = plt.Circle((pos[0], pos[1]), R, facecolor="navy", edgecolor="black", linestyle="-", linewidth=2)
    axes.add_artist(c)


axes.plot([BOUNDARY_BOX[0]/2, BOUNDARY_BOX[0]/2, -1*BOUNDARY_BOX[0]/2, -1*BOUNDARY_BOX[0]/2, BOUNDARY_BOX[0]/2], [-1*BOUNDARY_BOX[1]/2, BOUNDARY_BOX[1]/2, BOUNDARY_BOX[1]/2, -1*BOUNDARY_BOX[1]/2, -1*BOUNDARY_BOX[1]/2], color="black")


axes.set_xlabel("x position in [a.u.]", fontsize=20)
axes.set_ylabel("y position in [a.u.]", fontsize=20)
axes.grid()


fig.savefig("data/n_particle/simbox.png")


plt.show()
