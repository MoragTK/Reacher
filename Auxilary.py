import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error as mse

import getpass


''' 
    *Auxilary Functions*
'''


#  Out of 50 random u[k] actions, returns an
#  action u[k] that minimizes the speed of the arm.
def StopArm(model,x,N=50):
    x_ = np.reshape(np.copy(x), (1, 8))
    xtarget = np.copy(x_)
    xtarget[0, 6] = 0
    xtarget[0, 7] = 0
    min = 100
    uStop = np.zeros((1, 2))
    for i in range(N):
        u = np.random.random((1, 2))
        x1=model.predict(x_,u)
        MSE = mse(xtarget, x1)
        if MSE < min:
            min = MSE
            uStop = u
    return uStop


# Receives a vector x and a matrix M, and returns the result of x.T * M * x
def xMx(x, M):
    xt = np.transpose(x)
    xt_M = np.matmul(xt, M)
    xt_M_x = np.matmul(xt_M, x)
    return xt_M_x


# receives an action vector u[k] and cuts it in the limits lowerLim
# and upperLim according to the contraints on the action u[k].
def constrained(u, lowerLim=-1, upperLim=1):
    if u[0] > upperLim:
        u[0] = upperLim
    if u[0] < lowerLim:
        u[0] = lowerLim

    if u[1] > upperLim:
        u[1] = upperLim
    if u[1] < lowerLim:
        u[1] = lowerLim

    return u


# This function is not used anymore, but helped us verify that
# the state-space matrices A,B were correct.
def evaluateLTIError(controller, xk, uk, xk1):
    xk_ = np.reshape(np.copy(xk), (8, 1))
    uk_ = np.reshape(np.copy(uk), (2, 1))
    xk1_ = np.reshape(np.copy(xk1), (8, 1))
    A, B = controller.deriveAB(xk, uk)
    xk1_lti = np.matmul(A, xk_) + np.matmul(B, uk_)
    ltiErr = mse(xk1_lti, xk1_)
    print ltiErr





