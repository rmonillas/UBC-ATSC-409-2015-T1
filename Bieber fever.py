###############################################################################
# BIOL 301 Bieber Fever
###############################################################################

from numlabs.lab5.lab5_funs import Integrator
from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)


class Integ61(Integrator):
   
    def __init__(self, coeff_file_name,initvars=None,uservars=None,
                timevars=None):
        super().__init__(coeff_file_name)
        self.set_yinit(initvars,uservars,timevars)
        
    def set_yinit(self,initvars,uservars,timevars):
        #
        # read in 'sigma beta rho', override if uservars not None
        #
        if uservars:
            self.config['uservars'].update(uservars)
        uservars = namedtuple('uservars', self.config['uservars'].keys())
        self.uservars = uservars(**self.config['uservars'])
        #
        # read in 'x y z'
        #
        if initvars:
            self.config['initvars'].update(initvars)
        initvars = namedtuple('initvars', self.config['initvars'].keys())
        self.initvars = initvars(**self.config['initvars'])
        #
        # set dt, tstart, tend if overiding base class values
        #
        if timevars:
            self.config['timevars'].update(timevars)
        timevars = namedtuple('timevars', self.config['timevars'].keys())
        self.timevars = timevars(**self.config['timevars'])
        self.yinit = np.array([self.initvars.x, self.initvars.y])
        self.nvars = len(self.yinit)
    
    def derivs5(self, coords, t):
        x,y = coords
        u = self.uservars
        f = np.empty_like(coords)
        f[0] = (-1*u.n2-u.b-u.p1-u.p2)*x + (u.n1-u.p2)*y + u.p2
        f[1] = (u.b+u.p1)*x + (-1*u.n1-u.c)*y
        return f
        


#
# make a nested dictionary to hold parameters
#
timevars=dict(tstart=0,tend=100,dt=0.01)
uservars=dict(b=0.5, c=0.2, n1=0.4, n2=0.5, p1=0.3, p2=0.9)
initvars=dict(x=1,y=0)
params=dict(timevars=timevars,uservars=uservars,initvars=initvars)
#
# expand the params dictionary into key,value pairs for
# the Integ61 constructor using dictionary expansion
#
theSolver = Integ61('lorenz.yaml',**params)
timevals, coords, errorlist = theSolver.timeloop5fixed()
xvals,yvals=coords[:,0],coords[:,1]




fig,ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(timevals,xvals,label='x')
ax.plot(timevals,yvals,label='y')
ax.set(title='x, y for trajectory',xlabel='time')
out=ax.legend()