"""
  this script show how to plot the heat conduction equation

"""
import matplotlib.pyplot as plt
from numlabs.lab2.lab2_functions import euler,beuler,leapfrog
import numpy as np


theFuncs={'euler':euler,'beuler':beuler,'leapfrog':leapfrog}

if __name__=="__main__":
    tend=100.
    Ta=20.
    To=30.
    theLambda=-8.
    funChoice='beuler'
    npts=1000
    # 75 for deltat = 0.20
    # 62 for deltat = 0.24
    # 60 for deltat = 0.25
    # 50 for deltat = 0.30
    approxTime,approxTemp=theFuncs[funChoice](npts,tend,To,Ta,theLambda)
    exactTime=np.empty([npts,],float)
    exactTemp=np.empty_like(exactTime)
    for i in np.arange(0,npts):
       exactTime[i] = tend*i/npts
       exactTemp[i] = Ta + (To-Ta)*np.exp(theLambda*exactTime[i])
    plt.close('all')
    plt.figure(1)
    plt.clf()
    plt.plot(exactTime,exactTemp,'r+')
    plt.hold(True)
    plt.plot(approxTime,approxTemp)
    theAx=plt.gca()
    theAx.set_xlim([0,15])
    theAx.set_ylim([10,30])
    outdict=dict(deltat=tend/npts,func=funChoice)
    theAx.set_title("stability check for leapfrog with deltat={deltat:5.2g}".format_map(outdict))
    plt.show()
   
