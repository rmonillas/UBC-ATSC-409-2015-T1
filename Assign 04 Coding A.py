###############################################################################
# ATSC 409 Assignment 04 Problem Coding A
###############################################################################

#%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
from numlabs.lab4.example.test import read_init, derivs4

##### Prepare for harmonic oscillator.
coeff=read_init('Assign_04.json')
time=np.arange(coeff.t_beg,coeff.t_end,coeff.dt)
y=coeff.yinitial
nsteps=len(time)
soln=np.empty([nsteps], 'float')

##### Write a routine that solves the harmonic oscillator with Heun's method.
def heun(coeff, y, derivs):
    k1 = coeff.dt * derivs(coeff,y)
    k2 = coeff.dt * derivs(coeff,y + ((2.0/3.0) * k1))
    ynew = y + (1.0/4.0) * (k1 + (3.0 * k2))
    return ynew

##### Compute the solution.
for i in range(nsteps):
    y=heun(coeff,y,derivs4)
    soln[i]=y[0]

##### Construct plots.
theFig,theAx=plt.subplots(1,1)
theAx.plot(time,soln,'o-')
theAx.set_title(coeff.plot_title)
theAx.set_xlabel('Time (seconds)')
theAx.set_ylabel('Y-values')
plt.show()

###############################################################################