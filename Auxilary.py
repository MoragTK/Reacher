import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
import cvxpy
import matplotlib.pyplot as plt

import cvxopt
from cvxopt import matrix
import scipy.linalg


#CalculateAB
# Receives the model, the input vector and delta
# Returns the state space matrices A and B.
def CalculateAB( xu, model, delta=0.01):
    A = np.ones((6, 6))
    B = np.ones((6, 2))
    net = model
    for i in range(0, 8):
        x = np.copy(xu)
        x[0, i] = np.copy(xu[0, i] + delta)
        d2 = net.predict(x)
        x = np.copy(xu)
        x[0, i] = np.copy(xu[0, i] - delta)
        d1 = net.predict(x)
        if i < 6:
            A[:, i] = (d2 - d1) / (2*delta)
        else:
            B[:, i - 6] = (d2 - d1) / (2*delta)
    return [A, B]

# set Q function
def setQ(Q):
        Q[0,0]=0 # theta 1
        Q[1,1]=0 # theta 2
        Q[2,2]=1 # v1
        Q[3,3]=1 # v2
        Q[4,4]=1 # dx (finger-ball)
        Q[5,5]=1 # dy
        return Q

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


def LqrFhD2(A,B,Q,R,N=10):
    PN=Q
    Bt=np.transpose(B)
    for i in range(0,N-1):
        PN=scipy.linalg.solve_discrete_are(A,B,PN,R)

    c1 = np.matmul(Bt, PN)
    c2 = np.matmul(c1, B) + R  # Bt*Pk*B+R
    c3=inv(c2)
    c4 = np.matmul(Bt, PN)
    c5 = np.matmul(c4, A)  # Bt*Pk*A
    F = np.matmul(c3, c5)  # inv(Bt*Pk*B+R)*(Bt*Pk*A)
    return F






def getError(A, B, model, uk, xnu, xreal):
    x = xnu[0,0:6]
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
def getAll(Q,size):
    out = np.copy(Q.pop())
    Q.appendleft(np.copy(out))
    input = np.copy(out[:, :8])
    target = np.copy(out[:, 8:])
    for i in range(0,size-1):
        out = np.copy(Q.pop())
        Q.appendleft(np.copy(out))
        inp = np.copy(out[:, :8])
        tar = np.copy(out[:, 8:])
        input = np.vstack((input, inp))
        target = np.vstack((target, tar))
    return [input,target]


def scanUopt(model,xu,ball):
    x=np.copy(xu[0,:6])
    xd =np.copy(xu[0,:6])
    u1=-1.1
    MSE=1000
    ud=np.matrix([[0.,0.]])
    xd[0,4]=ball[0]        #dx
    xd[0,5]=ball[1]  #dy
    for i in range(0,20):
        u2=-1.1
        u1=u1+0.1
        for j in range(0,20):
            u2=u2+0.1
            u=np.matrix([[u1,u2]])
            xnu=np.hstack((x,u))
            xn1=model.predict(xnu)
            mse=mean_squared_error(xn1,xd)
            #print(mse,j,i)
            if mse<MSE:
                MSE=mse
                ud=u
    return ud


