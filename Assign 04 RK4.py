###############################################################################
# ATSC 409 Assignment 04 Problem RK4
###############################################################################

#%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
from numlabs.lab4.lab4_functions import initinter41,eulerinter41,\
                                        midpointinter41,rk4ODEinter41
                                        
##### Prepare for example eq:test.
initialVals={'yinitial': 1,'t_beg':0.,'t_end':1.,'dt':0.05,'c1':-1.,'c2':1.,'c3':1.}
coeff = initinter41(initialVals)
timeVec=np.arange(coeff.t_beg,coeff.t_end,coeff.dt)
nsteps=len(timeVec)

##### Set up lists for approximated values (e=euler, m=midpoint, rk=rk4).
ye=[]
ym=[]
yrk=[]
y=coeff.yinitial
ye.append(coeff.yinitial)
ym.append(coeff.yinitial)
yrk.append(coeff.yinitial)

##### Obtain approximated y values.
for i in np.arange(1,nsteps):
    ynew=eulerinter41(coeff,y,timeVec[i-1])
    ye.append(ynew)
    ynew=midpointinter41(coeff,y,timeVec[i-1])
    ym.append(ynew)
    ynew=rk4ODEinter41(coeff,y,timeVec[i-1])
    yrk.append(ynew)
    y=ynew
    
##### Obtain exact values.
analytic=timeVec + np.exp(-timeVec)

##### Construct plots.
theFig=plt.figure(0)
theFig.clf()
theAx=theFig.add_subplot(111)
l1=theAx.plot(timeVec,analytic,'b-',label='analytic')
theAx.set_xlabel('time (seconds)')
l2=theAx.plot(timeVec,ye,'r-',label='euler')
l3=theAx.plot(timeVec,ym,'g-',label='midpoint')
l4=theAx.plot(timeVec,yrk,'m-',label='rk4')
theAx.legend(loc='best')
theAx.set_title('interactive 4.2')

##### Take mean squared error between approximation and exact values.
me = np.ndarray(20)
mm = np.ndarray(20)
mr = np.ndarray(20)
for i in range(0,20):
    me[i] = analytic[i] - ye[i]
    mm[i] = analytic[i] - ym[i]
    mr[i] = analytic[i] - yrk[i]
mspe_e = np.mean(me**2)
print('MSE for Euler method is ' + str(mspe_e))
mspe_m = np.mean(mm**2)
print('MSE for midpoint method is ' + str(mspe_m))
mspe_r = np.mean(mr**2)
print('MSE for RK4 method is ' + str(mspe_r))

###############################################################################