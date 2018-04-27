import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
import scipy.linalg

#CalculateAB
# Receives the model, the input vector and delta
# Returns the state space matrices A and B.
#TODO: This function needs to be verified
def deriveAB(xk_in, uk_in, model, eps=1e-4):
    xkDim = 8
    ukDim = 2
    A = np.ones((xkDim, xkDim))
    B = np.ones((xkDim, ukDim))

    for i in range(0, xkDim):
        xk = np.copy(xk_in)
        xk[i, 0] += eps
        state_inc = model.predict(xk, uk_in)
        xk = np.copy(xk_in)
        xk[i, 0] -= eps
        state_dec = model.predict(xk, uk_in)
        A[:, i] = (state_inc[:, 0] - state_dec[:, 0]) / (2 * eps) # TODO: Is this how A should be? or transpose?

    for i in range(0, ukDim):
        uk = np.copy(uk_in)
        uk[i, 0] += eps
        state_inc = model.predict(xk_in, uk)
        uk = np.copy(uk_in)
        uk[i, 0] -= eps
        state_dec = model.predict(xk_in, uk)
        B[:, i] = (state_inc[:, 0] - state_dec[:, 0]) / (2 * eps)

    return A, B


def xMx(x, M):
    xt = np.transpose(x)
    xt_M = np.matmul(xt, M)
    xt_M_x = np.matmul(xt_M, x)
    return xt_M_x


def solveRiccati(A, B, Pk, Q, R):

    At = np.transpose(A)
    Bt = np.transpose(B)
    AtPk = np.matmul(At, Pk)
    BtPk = np.matmul(Bt, Pk)

    AtPkA = np.matmul(AtPk, A)
    AtPkB = np.matmul(AtPk, B)
    BtPkA = np.matmul(BtPk, A)
    BtPkB = np.matmul(BtPk, B)

    inv_BtPkB_R = inv(BtPkB + R)  # (bt*Pk*B + R)^-1
    t_ = np.matmul(AtPkB, inv_BtPkB_R)  # (At*Pk*B) * (bt*Pk*B + R)^-1
    t = np.matmul(t_, BtPkA)           # (At*Pk*B) * (bt*Pk*B + R)^-1 * (Bt*Pk*A)

    result = AtPkA - t + Q
    return np.copy(result)

def getError(A, B, model, uk, xnu, xreal):
    x = xnu[0, 0:6]
    x = np.hstack((x, uk))
    netOut = model.predict(x)
    x = np.transpose(x[0, 0:6])
    ABOout = np.matmul(A, x)
    ABOout = ABOout + np.matmul(B, np.transpose(uk))
    ABOout = np.transpose(ABOout)
    print "net-real: " + str(mean_squared_error(netOut, xreal))
    print "AB-real: " + str(mean_squared_error(ABOout, xreal))
   # print "AB-net: " + str(mean_squared_error(ABOout, netOut))
    return


'''

def LqrFhD(A,B,Q,R,N=10):
    PN=Q
    At=np.transpose(A)
    Bt=np.transpose(B)
    for i in range(0,N-1):
        a1=np.matmul(At,PN)
        a2=np.matmul(a1,A)+Q # At*Pk*A+Q
        a3=np.matmul(At,PN)
        a4=np.matmul(a3,B)  # At*Pk*B
        a5=np.matmul(Bt,PN)
        a6=np.matmul(a5,B)+R # Bt*Pk*B+R
        a7=inv(a6)            # inv(R+Bt*Pk*B)
        a8=np.matmul(Bt,PN)
        a9=np.matmul(a8,A)    # Bt*Pk*A

        b1=np.matmul(a4,a7) # (At*Pk*B)*inv(R+Bt*Pk*B)
        b2=-np.matmul(b1,a9)+a2 # At*Pk*A+Q-(At*Pk*B)*inv(R+Bt*Pk*B)*(Bt*Pk*A)

        PN=b2

    c1=np.matmul(Bt,PN)
    c2=np.matmul(c1,B)+R # Bt*Pk*B+R
    c3=inv(c2)
    c4=np.matmul(Bt,PN)
    c5=np.matmul(c4,A)  # Bt*Pk*A
    F=np.matmul(c3,c5) # inv(Bt*Pk*B+R)*(Bt*Pk*A)
    return F





'''



def plot_curve(X, ball, cost=None, step=0,t=2):
    from numpy import *
    import matplotlib
    matplotlib.use('qt4agg')
    import matplotlib.pyplot as plt
    x=X[:,4]
    y=X[:,5]
    plt.ion()
    plt.plot(x,y)
    plt.title("Step: {}".format(step))
    plt.xlim([-0.25, 0.25])
    plt.ylim([-0.25, 0.25])
    plt.plot(ball[0],ball[1],'go')
    plt.text(ball[0],ball[1],'ball')
    plt.plot(X[0,4],X[0,5],'ro')
    plt.text(X[0,4],X[0,5],'start')

    plt.plot(X[-1, 4], X[-1, 5], 'ro')
    plt.text(X[-1, 4], X[-1, 5], 'End')
    plt.text(X[-1, 4], X[-1, 5]+0.03, str(cost))
    plt.show(block=False)
    #print "cost: " +str(cost)
    #print(X[0,4],X[0,5])
    plt.pause(1)
    plt.clf()

def plot_next_pos(xk1_pred,xk1_real):
    xk1_real_=np.reshape(np.copy(xk1_real), (1,8))
    #from numpy import *
    import matplotlib
    matplotlib.use('qt4agg')
    import matplotlib.pyplot as plt
    plt.ion()
    plt.xlim([-0.25, 0.25])
    plt.ylim([-0.25, 0.25])
    plt.plot(xk1_pred[0,4],xk1_pred[0,5],'ro')
    plt.text(xk1_pred[0,4],xk1_pred[0,5],'xk_pred')
    plt.plot(xk1_real_[0, 4], xk1_real_[0, 5], 'ro')
    plt.text(xk1_real_[0, 4]+0.02, xk1_real_[0, 5]+0.02, 'xk1_real')
    plt.show(block=False)
    plt.pause(4)
    plt.clf()
    #plt.close(fig)

