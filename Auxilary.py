import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
import scipy.linalg


#CalculateAB
# Receives the model, the input vector and delta
# Returns the state space matrices A and B.
#TODO: This function needs to be verified
def deriveAB(xk_in, uk_in, model, eps=1e-4):
    xk_ = np.reshape(np.copy(xk_in), (1, 8))
    uk_ = np.reshape(np.copy(uk_in), (1, 2))
    xkDim = 8
    ukDim = 2
    A = np.ones((xkDim, xkDim))
    B = np.ones((xkDim, ukDim))

    for i in range(0, xkDim):
        xk = np.copy(xk_)
        xk[:, i] += eps
        state_inc = model.predict(xk, uk_)
        xk = np.copy(xk_)
        xk[:, i] -= eps
        state_dec = model.predict(xk, uk_)
        A[:, i] = (state_inc - state_dec) / (2 * eps)

    for i in range(0, ukDim):
        uk = np.copy(uk_)
        uk[:, i] += eps
        state_inc = model.predict(xk_, uk)
        uk = np.copy(uk_)
        uk[:, i] -= eps
        state_dec = model.predict(xk_, uk)
        B[:, i] = (state_inc - state_dec) / (2 * eps)

    return A, B


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


'''
# set Q function
def setQ(Q):
        Q[0, 0] = 0  # cos(theta) of outer arm
        Q[1, 1] = 0  # cos(theta) of inner arm
        Q[2, 2] = 1  # sin(theta) of outer arm
        Q[3, 3] = 1  # sin(theta) of inner arm
        Q[4, 4] = 1  # velocity of outer arm
        Q[5, 5] = 1  # velocity of outer arm
        Q[6, 6] = 1  # fingertip location x
        Q[7, 7] = 1  # fingertip location y

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


def lqrFhD(A,B,Q,R,N=10):

    Pk = Q

    for i in range(0, N-1):
        temp = scipy.linalg.solve_discrete_are(A, B, Pk, R)
        Pk = np.copy(temp)
        print "calculating p"+i+"\n"

    Bt = np.transpose(B)
    Bt_Pk = np.matmul(Bt, Pk)
    Bt_Pk_B = np.matmul(Bt_Pk, B)
    Bt_Pk_A = np.matmul(Bt_Pk, A)
    F = np.matmul(inv(Bt_Pk_B+R), Bt_Pk_A)  # inv(Bt*Pk*B+R)*(Bt*Pk*A)

    return F

'''




'''
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
'''

def scanUopt(model,xu,ball):
    u1=-0.002
    MSE=1000
    ud=np.matrix([[0.,0.]])
    for i in range(0,3):
        u2=-0.002
        u1=u1+0.001
        for j in range(0,3):
            u2=u2+0.001
            u=np.copy(np.matrix([[u1,u2]]))
            xnu=np.hstack((xu,u))
            xn1=np.matrix(model.predict(xnu))
            xy=xn1[0,4:]
            mse=mean_squared_error(xy,ball)
            if mse<MSE:
                MSE=mse
                ud=u
    return ud

