from collections import deque
import numpy as np

'''
    This class implements the data base for all the samples that are recorded.
    The samples that are saved in this data base are later used for training the network.
    A sample is saved like so: <x[k],u[k],x[k+1]>.
    Every new sample that is added, deletes an old one, so that the data base 
    always holds the newest samples for the model to be trained on.
'''


class DataSet:

    # Constructor
    def __init__(self, size):
        self.size = size
        self.Q = deque(maxlen=self.size)
        self.numOfElements = 0
        return

    # append - receives Xk, Uk, Xk+1 and inserts the sample into the data set. (<xn,un,xn+1>)
    def append(self, xk_uk=None, xk_1=None):
        xk_uk_c = np.copy(xk_uk)
        xk_1_c = np.copy(xk_1)

        sample = np.vstack((xk_uk_c, xk_1_c))
        self.Q.append(sample)
        if self.numOfElements < self.size:
            self.numOfElements += 1

    # getAll - returns all samples for training purposes.
    def getAll(self):  # TODO: Consider changing the emulator input
        xuIn = np.zeros((self.numOfElements, 10))
        xOut = np.zeros((self.numOfElements, 8))
        for i in range(0, self.numOfElements):
            tmpOut = self.Q.pop()
            xuIn[i, :] = np.copy(tmpOut[:10, 0])
            xOut[i, :] = np.copy(tmpOut[10:, 0])
        for i in range(0, self.numOfElements):
            tmpIn = np.vstack((np.reshape((xuIn[i, :]), (10, 1)), np.reshape((xOut[i, :]), (8, 1))))
            self.Q.appendleft(np.copy(tmpIn))
        return [np.copy(xuIn), np.copy(xOut)]

