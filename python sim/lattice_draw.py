import numpy as np
import matplotlib.pyplot as plt

a = np.asarray([np.sqrt(3)/2, 1/2])
b = np.asarray([-1*np.sqrt(3)/2, 1/2])


lat_xarr = []
lat_yarr = []


NUMBER_PARTICLES = 100


for i in range(NUMBER_PARTICLES):
    
    m1:int = int(i/np.sqrt(NUMBER_PARTICLES)) # n1 gitter coordinate
    m2:int = int(i%np.sqrt(NUMBER_PARTICLES)) # n2 gitter coordinate

    p = np.multiply(a, m1) + np.multiply(b, m2)    
    lat_xarr.append(p[0])
    lat_yarr.append(p[1])




fig, axes = plt.subplots(1,1, figsize=(10, 10))

axes.scatter(lat_xarr, lat_yarr, s=40)
axes.grid()

plt.show()
