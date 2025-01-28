import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

def theta(R, a):
    return np.divide(a, R)*3/np.pi



Rarr = np.linspace(2.8, 14, 200)


fig, axes = plt.subplots(1,1, figsize=(20, 10))


axes.plot(Rarr, theta(Rarr, 2.1)/np.pi*180, label=r"$\theta_c(R_0)$, a=2.1")
axes.plot(Rarr, theta(Rarr, 2.0)/np.pi*180, label=r"$\theta_c(R_0)$, a=2.0")
axes.plot(Rarr, theta(Rarr, 1.9)/np.pi*180, color="navy", label=r"$\theta_c(R_0)$, a=1.9")
axes.plot(Rarr, theta(Rarr, 1.8)/np.pi*180, label=r"$\theta_c(R_0)$, a=1.8")

axes.plot([18*1.9/np.pi**2]*2, [0, 30], "--", color="green", label=r"$R_{min}(a=1.9)\approx 3.47$")
axes.plot([np.min(Rarr), np.max(Rarr)], [30, 30], "--", color="red", label=r"$\theta_{max}$")



axes.set_xlabel(r"Initial pertubation radius $R_0$ in [a.u.]", fontsize=20)
axes.set_ylabel(r"Angle $\theta$ in [deg]", fontsize=20)
axes.set_ylim([0, 40])
axes.legend(fontsize=20)
axes.grid()

fig.savefig("data/stability.png", bbox_inches='tight')

plt.show()




df = pd.read_excel("python sim/animation_eval.xlsx")
df_grain = df[df["NOTES"] == "grain"]
df_elastic = df[df["NOTES"] == "elastic"]
print(df_grain)



#df.to_excel("file.xlsx")


fig, axes = plt.subplots(1,1, figsize=(20, 10))


axes.plot(Rarr, theta(Rarr, 1.9)/np.pi*180, color="navy", label=r"$\theta_c(R_0)$, a=1.9")
axes.plot([18*1.9/np.pi**2]*2, [0, 30], "--", color="green", label=r"$R_{min}(a=1.9)\approx 3.47$")
axes.plot([np.min(Rarr), np.max(Rarr)], [30, 30], "--", color="red", label=r"$\theta_{max}$")

#axes.plot(df["R"], df["THETA [DEG]"], "o", color="gray", label="Simulations: Not clear")
axes.plot(df_elastic["R"], df_elastic["THETA [DEG]"], "o", color="green", label="Simulation: elastic restoration")
axes.plot(df_grain["R"], df_grain["THETA [DEG]"], "o", color="red", label="Simulation: grain boundary loop")





axes.set_xlabel(r"Initial pertubation radius $R_0$ in [a.u.]", fontsize=20)
axes.set_ylabel(r"Angle $\theta$ in [deg]", fontsize=20)
axes.set_ylim([0, 40])
axes.legend(fontsize=20)
axes.grid()

fig.savefig("data/stability_markers.png", bbox_inches='tight')

plt.show()


