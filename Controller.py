import scipy
import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_discrete_are
from Auxilary import deriveAB

xkDim = 8
ukDim = 2


class Controller:

    def __init__(self, model):
        self.R = np.identity(ukDim)*0.01
        self.Q = self.setQ()
        self.numOfLQRSteps = 2  # TODO: Return to 10
        self.threshold = 5
        self.model = model

    # Calculate next step using the iLQR algorithm
    def calculateNextAction(self, A, B, x0):

        arrayA = []
        arrayB = []

        for i in range(self.numOfLQRSteps):
            arrayA.append(A)
            arrayB.append(B)

        prevCost = 0
        while True:
            Fk = self.lqrFhd(arrayA, arrayB)
            nextAction = -np.matmul(Fk[0], x0)
            newArrayA, newArrayB = self.calculateSystemDynamics(Fk, x0)
            print "Fk is: {}".format(Fk)  # TODO: Delete
            print "nextAction is: {}".format(nextAction)  # TODO: Delete
            print "arrayA is: {}".format(arrayA)  # TODO: Delete
            print "newArrayA is: {}".format(newArrayA)  # TODO: Delete
            exit()
            cost = self.calculateCost(arrayA, arrayB, newArrayA, newArrayB)
            print "Cost: {} Difference: {}".format(cost, abs(prevCost-cost))  # TODO: Delete
            if cost <= prevCost:
                if abs(prevCost - cost) < self.threshold:
                    return nextAction
                else:
                    prevCost = cost
                    arrayA = newArrayA
                    arrayB = newArrayB
            else: # cost > prevCost
                print "cost is growing!!! fix this" #TODO: add code here


    # LQR
    def lqrFhd(self, arrayA, arrayB):
        Pk = self.Q
        Fk = []
        for i in range(self.numOfLQRSteps):
            A = arrayA[i]
            B = arrayB[i]
            temp = solve_discrete_are(A, B, Pk, self.R)
            Pk = np.copy(temp)
            Bt = np.transpose(B)
            Bt_Pk = np.matmul(Bt, Pk)
            Bt_Pk_B = np.matmul(Bt_Pk, B)
            Bt_Pk_A = np.matmul(Bt_Pk, A)
            F = np.matmul(inv(Bt_Pk_B + self.R), Bt_Pk_A)  # inv(Bt*Pk*B+R)*(Bt*Pk*A)
            Fk.insert(0, F)

        return Fk

    def calculateSystemDynamics(self, Fk, x0):
        arrayA = []
        arrayB = []
        xk = x0
        for i in range(self.numOfLQRSteps):
            uk = -np.matmul(Fk[i], xk)
            A, B = deriveAB(xk, uk, self.model)
            arrayA.append(A)
            arrayB.append(B)

        return np.copy(arrayA), np.copy(arrayB)

    def calculateCost(self, arrayA, arrayB, newArrayA, newArrayB):  # TODO: Possibly calculate cost in a different way - with Xk+1 = AXk + BUk

        numOfElemsA = arrayA[0].shape[0]*arrayA[0].shape[1]
        numOfElemsB = arrayB[0].shape[0]*arrayB[0].shape[1]

        lineA = []
        lineB = []
        newLineA = []
        newLineB = []
        for i in range(self.numOfLQRSteps):
            lineA = np.hstack((lineA, arrayA[i].reshape(numOfElemsA)))
            lineB = np.hstack((lineB, arrayB[i].reshape(numOfElemsB)))
            newLineA = np.hstack((newLineA, newArrayA[i].reshape(numOfElemsA)))
            newLineB = np.hstack((newLineB, newArrayB[i].reshape(numOfElemsB)))

        costA = costB = 0
        for i in range(numOfElemsA):
            costA += (lineA[i] - newLineA[i])/self.numOfLQRSteps
            costB += (lineB[i] - newLineB[i])/self.numOfLQRSteps
        totalCost = costA + costB

        return totalCost

    # set Q function
    def setQ(self):
        Q = np.zeros((xkDim, xkDim))
        Q[0, 0] = 0  # cos(theta) of outer arm
        Q[1, 1] = 0  # cos(theta) of inner arm
        Q[2, 2] = 1  # sin(theta) of outer arm
        Q[3, 3] = 1  # sin(theta) of inner arm
        Q[4, 4] = 1  # velocity of outer arm
        Q[5, 5] = 1  # velocity of outer arm
        Q[6, 6] = 1  # fingertip location x
        Q[7, 7] = 1  # fingertip location y

        return Q