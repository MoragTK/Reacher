import numpy as np
from numpy.linalg import inv

#CalculateAB
# Receives the model, the input vector and delta
# Returns the state space matrices A and B.
def CalculateAB( xu, model, delta=0.00001):
    A = np.ones((8, 8))
    B = np.ones((8, 2))
    net = model
    for i in range(0, 10):
        x = xu
        x[0, i] = xu[0, i] + delta
        d2 = net.predict(x)
        x = xu
        x[0, i] = xu[0, i] - delta
        d1 = net.predict(x)
        if i < 8:
            A[:, i] = (d2 - d1) / (2*delta)
        else:
            B[:, i - 8] = (d2 - d1) / (2*delta)
    return [A, B]

# set Q function
def setQ(Q):
    Q[0,0]=0.1
    Q[1,1]=0.1
    Q[2,2]=1
    Q[3,3]=1
    Q[4,4]=1
    Q[5,5]=1
    Q[6,6]=1
    Q[6,6]=1
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
