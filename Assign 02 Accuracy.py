import matplotlib.pyplot as plt
from numlabs.lab2.lab2_functions import euler,leapfrog,runge
import numpy as np

theFuncs={'euler':euler,'leapfrog':leapfrog,'runge':runge}

if __name__=="__main__":
    Ta= 20
    To= 30
    tend = 10.0
    theLambda= 0.8
    npts=25
    # 100 for deltat = 0.1
    #  50 for deltat = 0.2
    #  25 for deltat = 0.4
    
    funChoice='euler'
    #funChoice='leapfrog'
    #funChoice='runge'
    #
    #find the method in the theFuncs dictionary and call it
    #
    approxTime,approxTemp=theFuncs[funChoice](npts,tend,To,Ta,theLambda)
    
    plt.close('all')
    
    fig1,ax1=plt.subplots(1,1)
    ax1.plot(approxTime,approxTemp,label=funChoice)
    exactTime=np.empty([npts,],np.float)
    exactTemp=np.empty_like(exactTime)
    for i in np.arange(0,npts):
        exactTime[i] = tend*i/npts
        exactTemp[i] = Ta + (To-Ta)*np.exp(theLambda*exactTime[i])
    ax1.plot(exactTime,exactTemp,'r+',label='exact')
    outdict=dict(deltat=tend/npts,func=funChoice)
    title="exact and approx using {func} with deltat={deltat:5.2g}".format_map(outdict)
    ax1.set(title=title)
    ax1.legend(loc='best')
    
    fig2,ax2=plt.subplots(1,1)
    difference = exactTemp - approxTemp
    ax2.plot(exactTime,difference)
    title="exact - approx using {func} with deltat={deltat:5.2g}".format_map(outdict)
    ax2.set(title=title)
    
    localError = np.empty([npts,], np.float)
    timeStep = tend/npts
    for i in np.arange(1, npts):
        localError[i] = abs(exactTemp[i] - exactTemp[i-1] - \
            timeStep*theLambda*(To-Ta)*np.exp(theLambda*exactTime[i]))
    fig3,ax3=plt.subplots(1,1)
    ax3.plot(exactTime,localError)
    title="local error in euler approximation with deltat={deltat:5.2g}".format_map(outdict)
    ax3.set(title=title)
    
    globalError = np.empty([npts,], np.float)
    for i in np.arange(1, npts):
        err = 0
        for k in np.arange(0, i):
            err += localError[k]
        globalError[i] = err
    fig4,ax4=plt.subplots(1,1)
    ax4.plot(exactTime,globalError)
    title="global error in euler approximation with deltat={deltat:5.2g}".format_map(outdict)
    ax4.set(title=title)
    
    plt.show()








