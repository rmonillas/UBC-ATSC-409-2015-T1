###############################################################################
# ATSC 409 Assignment 01 Question 01
###############################################################################

from scipy.interpolate import interp1d

##### Initialize given info.
x = [-5, 0, 5, 8]
y = [-1, 0, 1, 4]

##### Use linear interpolation to find the y-value at x = 3.
fitLinear = interp1d(x, y, kind = 'linear')
print(fitLinear(3))
# 0.600

##### Use cubic interpolation to find the y-value at x = 3.
fitCubic = interp1d(x, y, kind = 'cubic')
print(fitCubic(3))
# 0.231

###############################################################################