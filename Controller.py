import scipy
import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_discrete_are
from Auxilary import deriveAB, xMx, solveRiccati

xkDim = 8
ukDim = 2


class Controller:

    def __init__(self, model):
        self.R = np.identity(ukDim)*0.1
        self.Q = self.setQ()
        self.numOfLQRSteps = 10
        self.threshold = 1e-5
        self.model = model

    # Calculate next step using the iLQR algorithm
    def calculateNextAction(self, A, B, x0):

        arrayA = []
        arrayB = []

        for i in range(self.numOfLQRSteps):
            arrayA.append(A)
            arrayB.append(B)

        prevCost = 1e+100
        minCost = 1e+100
        simNewTrajectory = True
        i = 0
        while i < 100:

            if simNewTrajectory is True:
                Fk, Pk = self.lqrFhd(arrayA, arrayB)
            else:
                Fk = Fk*0.8  # TODO: Not Correct


            # Forward Pass: Calculate X and U - the state-space and control signal trajectories.
            U, X = self.calculateTrajectories(Fk, x0)

            # Backward Pass: Estimate the dynamics for each (xk, uk) in the state-space and control signal trajectories.
            newArrayA, newArrayB = self.calculateSystemDynamics(U, X)

            cost = self.calculateCost(X, U, Pk)
            #print "Cost: {}".format(cost)  # TODO: Delete
            if cost < prevCost:
                # TODO: Save stats when cost was minimum, and give those!!!!!
               # if (abs(prevCost - cost))/cost < self.threshold:
               #     Fk_x0 = np.matmul(Fk[0], x0)
               #     nextAction = np.asarray(np.negative(Fk_x0))  # TODO: Add to previous uk?...
               #     print "cost difference is below threshold! Returning next action: {}".format(nextAction)
               #     return nextAction
                if cost < minCost:
                    Fk_x0 = np.matmul(Fk[0], x0)
                    nextAction = np.asarray(np.negative(Fk_x0))
                    minCost = cost
                    print "New Minimum Cost! {} ".format(minCost)

                arrayA = newArrayA
                arrayB = newArrayB
                simNewTrajectory = True
            else:
                simNewTrajectory = False
            prevCost = cost
            i += 1
        return nextAction

    # LQR
    def lqrFhd(self, arrayA, arrayB):
        Pk = [self.Q]
        Fk = []
        for i in range(self.numOfLQRSteps):
            A = arrayA[i]
            B = arrayB[i]
            Pk_m1 = solveRiccati(A, B, Pk[0], self.Q, self.R)
            Pk.insert(0, np.copy(Pk_m1))
            Bt = np.transpose(B)
            Bt_Pk = np.matmul(Bt, Pk[0])
            Bt_Pk_B = np.matmul(Bt_Pk, B)
            Bt_Pk_A = np.matmul(Bt_Pk, A)
            F = np.matmul(inv(Bt_Pk_B + self.R), Bt_Pk_A)  # inv(Bt*Pk*B+R)*(Bt*Pk*A)
            Fk.insert(0, F)
        Fk = np.asarray(Fk)
        Pk = np.asarray(Pk)
        return Fk, Pk

    def calculateTrajectories(self, Fk, x0):
        U = []
        X = []
        xk = x0
        for i in range(self.numOfLQRSteps):
            uk = -np.matmul(Fk[i], xk)
            X.append(xk)
            U.append(uk)
            xk = self.model.predict(xk, uk)
        return U, X

    # TODO: Document
    def calculateSystemDynamics(self, U, X):
        arrayA = []
        arrayB = []
        for i in range(self.numOfLQRSteps):
            A, B = deriveAB(X[i], U[i], self.model)
            #print "Deriving A, B from X[{}] and U[{}]".format(i, i)
            #print X[i]
            #print U[i]
            arrayA.append(A)
            arrayB.append(B)
        return np.copy(arrayA), np.copy(arrayB)

    #TODO: Document
    def calculateCost(self, X, U, Pk):
        cost = 0
        for i in range(self.numOfLQRSteps):
            cost += xMx(x=X[i], M=Pk[i]) + xMx(x=U[i], M=self.R)
        cost += xMx(x=X[i], M=self.Q)
        return cost

    # set Q function
    def setQ(self):
        Q = np.zeros((xkDim, xkDim))
        Q[0, 0] = 0   # cos(theta) of outer arm
        Q[1, 1] = 0   # cos(theta) of inner arm
        Q[2, 2] = 0   # sin(theta) of outer arm
        Q[3, 3] = 0   # sin(theta) of inner arm
        Q[4, 4] = 10  # distance between ball and fingertip - X axis
        Q[5, 5] = 10  # distance between ball and fingertip - Y axis
        Q[6, 6] = 5   # velocity of inner arm
        Q[7, 7] = 5   # velocity of outer arm

        return Q

