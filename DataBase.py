from collections import deque
import numpy as np
class DataBase():
    def __init__(self,N=10):
        self.size=N
        self.Q=deque(maxlen=N) # xn=6,un=2,xn1=6
        return
    def append(self,xu=None,xn1=None):
        self.Q.append(np.hstack((xu,xn1)))
        return
    def getLast(self):
        xn=self.Q.pop()
        self.Q.append(xn)
        tmpX=xn[0,8:]   # xn1
        tmpU=xn[0,6:8]  #un  we assume the next u will be close to the previos
        r=np.hstack((tmpX,tmpU))
        return r
    def getAll(self):   # this is for training
        input=np.zeros((self.size,8))
        target=np.zeros((self.size,6))
        for i in range(0,self.size):
            tmpOut=self.Q.pop()
            input[i,:]=tmpOut[0,:8]
            target[i,:]=tmpOut[0,8:]
        for i in range(0,self.size):
            tmpIn=np.hstack((input[i,:],target[i,:]))
            tmpIn=np.matrix(tmpIn)
            self.Q.appendleft(tmpIn)
        return [input,target]
