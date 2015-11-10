###############################################################################
# ATSC 409 Assignment 03 Question 02
###############################################################################

import numpy as np
import scipy.stats as sps

##### Write a function that creates the differential system matrix of size N+1
##### x N+1 given N.
def dir(N):
    m = np.zeros((N+1, N+1))
    m[0, 0] = m[N, N] = 1
    for i in range(1,N):
        m[i, i-1] = m[i, i+1] = 1
        m[i, i] = -2
    return m
    
##### Compute the condition number for the differential system matrix for 
##### several values of N between 5 and 50.
K = np.ndarray((46,), float)
for i in range(5,51):
    K[i-5] = np.linalg.cond(dir(i))
    
##### Take the N-1 x N-1 submatrix of dir(N) and compare the condition number
##### for the new matrix to the condition number for the original matrix.
def dire(N):
    m = np.zeros((N-1, N-1))
    for i in range(0,N-1):
        m[i, i] = -2
    for i in range(1,N-1):
        m[i, i-1] = m[i-1, i] = 1
    return m
    
C = np.ndarray((46,), float)
for i in range(5,51):
    C[i-5] = np.linalg.cond(dire(i))

##### Is the difference between the condition numbers significant?
abs(C - K)

##### Perform a one-sample t-test on the differences.
sps.ttest_1samp(abs(C-K), 0)
    
###############################################################################