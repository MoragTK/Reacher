import scipy
import numpy as np
from numpy.linalg import inv

xkDim = 8
ukDim = 2

class Controller:

    def __init__(self):
        self.R = np.identity(ukDim)*0.01
        self.Q = self.setQ()

    def calculateNextAction(self, A, B, xk):
        Fk = self.lqrFhd(A, B)
        uk = -np.matmul(Fk, xk)
        return uk

    def lqrFhd(self, A, B, N=10):
        Pk = self.Q
        for i in range(0, N - 1):
            temp = scipy.linalg.solve_discrete_are(A, B, Pk, self.R)
            Pk = np.copy(temp)
        Bt = np.transpose(B)
        Bt_Pk = np.matmul(Bt, Pk)
        Bt_Pk_B = np.matmul(Bt_Pk, B)
        Bt_Pk_A = np.matmul(Bt_Pk, A)
        F = np.matmul(inv(Bt_Pk_B + self.R), Bt_Pk_A)  # inv(Bt*Pk*B+R)*(Bt*Pk*A)
        return F

    # set Q function
    def setQ(self):
        Q = np.zeros((xkDim,xkDim))
        Q[0, 0] = 0  # cos(theta) of outer arm
        Q[1, 1] = 0  # cos(theta) of inner arm
        Q[2, 2] = 1  # sin(theta) of outer arm
        Q[3, 3] = 1  # sin(theta) of inner arm
        Q[4, 4] = 1  # velocity of outer arm
        Q[5, 5] = 1  # velocity of outer arm
        Q[6, 6] = 1  # fingertip location x
        Q[7, 7] = 1  # fingertip location y

        return Q