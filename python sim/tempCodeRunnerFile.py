R = 1
posarr = []
a = 2.5


countx = int(BOUNDARY_BOX[0] / a)
county = int(BOUNDARY_BOX[0] / a)

for x in range(-int(countx/2), int(countx/2)):
    for y in range(-int(county/2), int(county/2)):

        px = x*a
        py = y*a

        if x % 2 == 0:
            py += a/2

        dx = a/2
        dy = dx
        if (np.abs(px) > BOUNDARY_BOX[0]/2 - dx) or (np.abs(py) > BOUNDARY_BOX[1]/2 - dy):
            continue
        else:
            posarr.append([px, py])