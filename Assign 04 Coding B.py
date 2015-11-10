###############################################################################
# ATSC 409 Assignment 04 Problem Coding B
###############################################################################

#%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
from numlabs.lab4.lab4_functions import initinter41, derivsinter41, midpointinter41

##### Write a function that computes the solution using Heun's method.
def heun(coeff, y, theTime):
    k1 = coeff.dt * derivsinter41(coeff,y,theTime)
    k2 = coeff.dt * derivsinter41(coeff,y + (2.0/3.0 * k1), theTime + (2.0/3.0)*coeff.dt)
    y = y + (1.0/4.0) * (k1 + (3.0 * k2))
    return y
                                        
##### Prepare for example eq:test.
initialVals={'yinitial': 0,'t_beg':0.,'t_end':1.,'dt':0.05,'c1':-1.,'c2':1.,'c3':1.}
coeff = initinter41(initialVals)
timeVec=np.arange(coeff.t_beg,coeff.t_end,coeff.dt)
nsteps=len(timeVec)

##### Set up lists for approximated values (m=midpoint, h=heun).
ym=[]
yh=[]
y=coeff.yinitial
ym.append(coeff.yinitial)
yh.append(coeff.yinitial)

##### Obtain approximated y values.
for i in np.arange(1,nsteps):
    ynew=midpointinter41(coeff,y,timeVec[i-1])
    ym.append(ynew)
    ynew=heun(coeff,y,timeVec[i-1])
    yh.append(ynew)
    y=ynew

##### Construct plots.
theFig=plt.figure(0)
theAx=theFig.add_subplot(111)
l1=theAx.plot(timeVec,ym,'b-',label='midpoint')
l2=theAx.plot(timeVec,yh,'g-',label='heun')
theAx.set_xlabel('time (seconds)')
theAx.legend(loc='best')
theAx.set_title('Test equation with midpoint and heun')

##### Find the difference between midpoint and heun approximations.
diff = np.ndarray(20)
for i in range(0,20):
    diff[i] = yh[i] - ym[i]
diff = np.mean(diff**2)
print('Difference between midpoint and heun is ' + str(diff))

###############################################################################