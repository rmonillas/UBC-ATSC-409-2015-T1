import numlabs.lab5.lab5_funs
from importlib import reload
reload(numlabs.lab5.lab5_funs)
from numlabs.lab5.lab5_funs import Integrator
from collections import namedtuple
import numpy as np


class Integ55(Integrator):

    def set_yinit(self):
        #
        # read in 'c1 c2 c3'
        #
        uservars = namedtuple('uservars', self.config['uservars'].keys())
        self.uservars = uservars(**self.config['uservars'])
        #
        # read in initial yinit
        #
        initvars = namedtuple('initvars', self.config['initvars'].keys())
        self.initvars = initvars(**self.config['initvars'])
        self.yinit = np.array([self.initvars.yinit])
        self.nvars = len(self.yinit)
        return None

    def __init__(self, coeff_file_name):
        super().__init__(coeff_file_name)
        self.set_yinit()

    def derivs5(self, y, theTime):
        """
           y[0]=fraction white daisies
        """
        user=self.uservars
        f=np.empty_like(self.yinit)
        f[0]=user.c1*y[0] + user.c2*theTime + user.c3;
        return f

import matplotlib.pyplot as plt

theSolver=Integ55('expon.yaml')

timeVals,yVals,yErrors =theSolver.timeloop5Err()
timeVals=np.array(timeVals)
exact=timeVals + np.exp(-timeVals)
yVals=np.array(yVals)
yVals=yVals.squeeze()
yErrors=np.array(yErrors)

thefig,theAx=plt.subplots(1,1)
line1=theAx.plot(timeVals,yVals,label='adapt')
line2=theAx.plot(timeVals,exact,'r+',label='exact')
theAx.set_title('lab 5 interactive 5')
theAx.set_xlabel('time')
theAx.set_ylabel('y value')
theAx.legend(loc='center right')

#
# we need to unpack yvals (a list of arrays of length 1
# into an array of numbers using a list comprehension
#

thefig,theAx=plt.subplots(1,1)
realestError = yVals - exact
actualErrorLine=theAx.plot(timeVals,realestError,label='actual error')
estimatedErrorLine=theAx.plot(timeVals,yErrors,label='estimated error')
theAx.legend(loc='best')


timeVals,yVals,yErrors =theSolver.timeloop5fixed()

np_yVals=np.array(yVals).squeeze()
yErrors=np.array(yErrors)
np_exact=timeVals + np.exp(-timeVals)


thefig,theAx=plt.subplots(1,1)
line1=theAx.plot(timeVals,np_yVals,label='fixed')
line2=theAx.plot(timeVals,np_exact,'r+',label='exact')
theAx.set_title('lab 5 interactive 5 -- fixed')
theAx.set_xlabel('time')
theAx.set_ylabel('y value')
theAx.legend(loc='center right')

thefig,theAx=plt.subplots(1,1)
realestError = np_yVals - np_exact
actualErrorLine=theAx.plot(timeVals,realestError,label='actual error')
estimatedErrorLine=theAx.plot(timeVals,yErrors,label='estimated error')
theAx.legend(loc='best')
theAx.set_title('lab 5 interactive 5 -- fixed errors')

