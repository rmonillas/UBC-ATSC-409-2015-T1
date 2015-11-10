###############################################################################
# ATSC 409 Assignment 04 Problem Embedded
###############################################################################

#%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
from numlabs.lab4.lab4_functions import initinter41,rk4ODEinter41,rkckODEinter41

##### Prepare for example eq:test.
initialVals={'yinitial': 1,'t_beg':0.,'t_end':1.,'dt':0.025,'c1':-1.,'c2':1.,'c3':1.}
coeff = initinter41(initialVals)
timeVec=np.arange(coeff.t_beg,coeff.t_end,coeff.dt)
nsteps=len(timeVec)

##### Set up lists for approximated values (rk=rk4, rkck=embedded rk).
yrk=[]
yrkck=[]
y1=coeff.yinitial
y2=coeff.yinitial
yrk.append(coeff.yinitial)
yrkck.append(coeff.yinitial)

##### Obtain approximated y values.
for i in np.arange(1,nsteps):
    ynew=rk4ODEinter41(coeff,y1,timeVec[i-1])
    yrk.append(ynew)
    y1=ynew 
    ynew=rkckODEinter41(coeff,y2,timeVec[i-1])
    yrkck.append(ynew)
    y2=ynew 
    
##### Obtain exact values.
analytic=timeVec + np.exp(-timeVec)

##### Construct plots.
theFig,theAx=plt.subplots(1,1)
l1=theAx.plot(timeVec,analytic,'b-',label='analytic')
theAx.set_xlabel('time (seconds)')
l2=theAx.plot(timeVec,yrkck,'g-',label='rkck')
l3=theAx.plot(timeVec,yrk,'m-',label='rk')
theAx.legend(loc='best')
theAx.set_title('interactive 4.3')

##### Take mean squared error between approximation and exact values.
mrk = np.ndarray(20)
mrkck = np.ndarray(20)
for i in range(0,20):
    mrk[i] = analytic[i] - yrk[i]
    mrkck[i] = analytic[i] - yrkck[i]
mspe_rk = np.mean(mrk**2)
print('MSE for RK4 method is ' + str(mspe_rk))
mspe_rkck = np.mean(mrkck**2)
print('MSE for embedded RK4 method is ' + str(mspe_rkck))

###############################################################################