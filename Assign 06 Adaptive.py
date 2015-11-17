###############################################################################
# ATSC 409 Assignment 06 Problem Adaptive
###############################################################################

from numlabs.lab5.lab5_funs import Integrator
from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import time
from scipy import integrate



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
        self.yinit = np.array([self.initvars.x, self.initvars.y, self.initvars.z])
        self.nvars = len(self.yinit)
    
    def derivs5(self, coords, t):
        x,y,z = coords
        u=self.uservars
        f=np.empty_like(coords)
        f[0] = u.sigma * (y - x)
        f[1] = x * (u.rho - z) - y
        f[2] = x * y - u.beta * z
        return f
    
    def timeloop5Err(self):
        """return errors as well as values
        """
        t = self.timevars
        a = self.adaptvars
        i = self.initvars
        nvars = self.nvars
        oldTime = t.tstart
        olddt = t.dt
        yold = self.yinit
        yerror = np.zeros_like(yold)
        num = 0
        badsteps = 0
        goodsteps = 0
        timeVals = []
        yvals = []
        errorList = []
        integrationTime = []
        t_start = time.time()
        while(oldTime < t.tend):
            tic = time.time()
            timeVals.append(oldTime)
            yvals.append(yold)
            errorList.append(yerror)
            if(num > a.maxsteps):
                raise Exception('num > maxsteps')
            # start out with goodstep false and
            # try different sizes for the next step
            # until one meets the error conditions
            # then move onto next step by setting
            # goodstep to true
            goodStep = False
            failSteps = 0
            while(not goodStep):
                # to exit this loop, need to
                # get the estimated error smaller than
                # the desired error set by the relative
                # tolerance
                if(failSteps > a.maxfail):
                    raise Exception('failSteps > a.maxfail')
                #
                # try a timestep, we may need to reverse this
                #
                ynew, yerror, timeStep = self.rkckODE5(yold, oldTime, olddt)
                # print("try a step: : ", ynew)
                #
                # lab 5 section 4.2.3
                # find the desired tolerance by multiplying the relative
                # tolerance (RTOL) times the value of y
                # compare this to the error estimate returnd from rkckODE5
                # atol takes care of the possibility that y~0 at some point
                #
                errtest = 0.
                for i in range(nvars):
                    errtest = errtest + \
                        (yerror[i] / (a.atol + a.rtol * np.abs(ynew[i])))**2.0
                errtest = np.sqrt(errtest / nvars)
                #
                # lab5 equation 4.13, S
                #
                dtchange = a.s * (1.0 / errtest)**0.2
                # print("dtchange, errtest, timeStep: ",
                #       dtchange, errtest, timeStep, ynew, yerror)
                if (errtest > 1.0):
                    # estimated error is too big so
                    # reduce the timestep and retry
                    # dtFailMax ~ 0.5, which guarantees that
                    # the new timestep is reduced by at least a
                    # factor of 2
                    # dtFailMin~0.1, which means that we don't trust
                    # the estimate to reduce the timestep by more
                    # than a factor of 10 in one loop
                    if(dtchange > a.dtfailmax):
                        olddt = a.dtfailmax * olddt
                    elif (dtchange < a.dtfailmin):
                        olddt = a.dtfailmin * olddt
                    else:
                        olddt = dtchange * olddt
                    if (timeStep + olddt == timeStep):
                        raise Exception('step smaller than machine precision')
                    failSteps = failSteps + 1
                    #
                    # undo the timestep since the error wasn't small enough
                    #
                    ynew = yold
                    timeStep = oldTime
                    # go back to top and see if this olddt produices
                    # a better yerrror
                else:
                    # errtest < 1, so we're happy
                    # try to enlarge the timestep by a factor of dtChange > 1
                    # but keep it smaller than dtpassmax
                    # try enlarging the timestep bigger for next time
                    # dtpassmin ~ 0.1 and dtpassmax ~ 5
                    if (abs((1.0 - dtchange)) > a.dtpassmin):
                        if(dtchange > a.dtpassmax):
                            dtnew = a.dtpassmax * olddt
                        else:
                            dtnew = dtchange * olddt
                    else:
                        # don't bother changing the step size if
                        # the change is less than dtpassmin
                        dtnew = olddt
                    goodStep = True
                    #
                    # overwrite the old timestep with the new one
                    #
                    oldTime = timeStep
                    yold = ynew
                    # go back up to top while(timeStep < t.tend)
                    goodsteps = goodsteps + 1
                #
                # this is number of times we decreased the step size without
                #  advancing
                #
                badsteps = badsteps + failSteps
            # special case if we're within one ortwo timesteps of the end
            # otherwise, set dt to the new timestep size
            if(timeStep + dtnew > t.tend):
                olddt = t.tend - timeStep
            elif(timeStep + 2.0 * dtnew > t.tend):
                olddt = (t.tend - timeStep) / 2.0
            else:
                olddt = dtnew
            elapsed = time.time() - tic
            integrationTime.append(elapsed)
        t_total = time.time() - t_start
        timeVals = np.array(timeVals).squeeze()
        yvals = np.array(yvals).squeeze()
        errorVals = np.array(errorList).squeeze()
        integTime = np.array(integrationTime).squeeze()
        self.timevals = timeVals
        self.yvals = yvals
        self.errorVals = errorVals
        self.integTime = integTime
        self.t_total = t_total
        return (timeVals, yvals, errorVals, integTime, t_total)

    def timeloop5fixed(self):
        """fixed time step with
           estimated errors
        """
        t = self.timevars
        yold = self.yinit
        yError = np.zeros_like(yold)
        yvals = [yold]
        errorList = [yError]
        timeSteps = np.arange(t.tstart, t.tend, t.dt)
        integrationTime = np.ndarray((len(timeSteps),), float)
        t_start = time.time()
        for theTime in timeSteps[:-1]:
            tic = time.time()
            yold, yError, newTime = self.rkckODE5(yold, theTime, t.dt)
            yvals.append(yold)
            errorList.append(yError)
            elapsed = time.time() - tic
            integrationTime[theTime] = elapsed
        t_total = time.time() - t_start
        yvals = np.array(yvals).squeeze()
        errorVals = np.array(errorList).squeeze()
        return (timeSteps, yvals, errorVals, integrationTime, t_total)
    

#
# make a nested dictionary to hold parameters
#
timevars=dict(tstart=0,tend=27,dt=0.01)
uservars=dict(sigma=10,beta=2.6666,rho=28)
initvars=dict(x=5,y=5,z=5)
params=dict(timevars=timevars,uservars=uservars,initvars=initvars)
#
# expand the params dictionary into key,value pairs for
# the Integ61 constructor using dictionary expansion
#
theSolver = Integ61('lorenz.yaml',**params)
timevals, coords, errorlist, integTime, t = theSolver.timeloop5fixed()
timevals2, coords2, errorlist2, integTime2, t2 = theSolver.timeloop5Err()

##### Solution using the fixed scheme:
thefig, theAx = plt.subplots(1,1)
theLines = theAx.plot(timevals, integTime)
theAx.set_title('Fixed timeloop')
theAx.set_xlabel('time steps')
theAx.set_ylabel('integration time')

##### Solution using the adaptive scheme:
thefig, theAx = plt.subplots(1,1)
theLines = theAx.plot(timevals2, integTime2)
theAx.set_title('Adaptive timeloop')
theAx.set_xlabel('time steps')
theAx.set_ylabel('integration time')

##### Solution using the odeint module:
def lorentz_deriv(coords, t0, sigma=10., beta=8./3, rho=28.0):
    x,y,z = coords
    out = [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    return out
x0 = np.array([5,5,5])
tt = np.linspace(0, 50, 1000)
tic = time.time()
x_t = integrate.odeint(lorentz_deriv, x0, tt)
t3 = time.time() - tic


print("The total elapsed time for the fixed scheme is " + str(t) + " seconds.")
print("The total elapsed time for the adaptive scheme is " + str(t2) + " seconds.")
print("The total elapsed time for the odeint module is " + str(t3) + " seconds.")


###############################################################################