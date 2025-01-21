from uncertainties import ufloat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



xarr = np.linspace(0, 100, 100)
yarr = xarr


fig, axes = plt.subplots(1,1, figsize=(10, 10))

axes.plot(xarr, yarr, label="<r(t)^2>", color="navy")

axes.set_yscale('log')
axes.set_xscale('log')

axes.grid()
plt.show()