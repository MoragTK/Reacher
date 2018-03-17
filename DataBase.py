from collections import deque
import numpy as np


class DataBase():

    # Constructor
    def __init__(self, N = 10):
        self.size = N
        self.Q = deque(maxlen=N) #xn = 6, un = 2, xn1 = 6
        return

    # receives Xk, Uk, and Xk+1, and adds them as another data sample to the data base.
    def append(self, xu=None, xn1=None):
        self.Q.append(np.hstack((xu, xn1))) #TODO: Doesnt need to be [xu, xn1]?
        return

    # returns the newest sample at the data base
    def getLast(self): #TODO: I dont think it needs "self" inside... does it?
        xn = self.Q.pop()
        self.Q.append(xn)
        tmpX=xn[0,8:]   # xn1 #TODO: Xk+1?
        tmpU=xn[0,6:8]  # un. We assume the next u will be close to the previous #TODO: Where is the original Xk?
        r=np.hstack((tmpX,tmpU))
        return r

    # returns all of the data in the data base.  ???
    def getAll(self):   # this is for training #TODO: Do we need to send self?
        input = np.zeros((self.size, 8))
        target = np.zeros((self.size, 6))
        for i in range(0, self.size):
            tmpOut = self.Q.pop()
            input[i, :] = tmpOut[0, :8]
            target[i, :] = tmpOut[0, 8:]
        for i in range(0, self.size):
            tmpIn = np.hstack((input[i,:],target[i,:]))
            tmpIn = np.matrix(tmpIn)
            self.Q.appendleft(tmpIn)
        return [input,target]
        #TODO: Question... Is this the cycle thing?