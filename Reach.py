from DataBase import DataBase
from DataClass import DataClass
from Auxilary import CalculateAB, setQ, LqrFhD, getError
import numpy as np
import keras
import time
import os

# for LQR controller
R = np.identity(2) * 0.01
Q = np.identity(6)
Q = setQ(Q)
size = 100  # size of samples data base

filepath = '/home/project/PycharmProjects/Reacher/net/6xy21.3-9_37'

# create Database and random data.
data = DataBase(size)
sample = DataClass()
sample.makeData(size+1)
d = sample.getData()
X = d[0]
Y = d[1]
# now we first, fill the Database  with sample who come from random Action
for i in range(0, size):
    data.append(X[size-i-1], Y[size-i-1])

model = keras.models.load_model(filepath)
timeout = time.time()  # limit time of the script
t = timeout
while True:
    #input, target = data.getAll()   # we get the all data for training
    #model.fit(input, target, batch_size=10, epochs=150, verbose=0)

    # this is [xn,un-1]
    xnu = data.getLast()
    ball=DataClass.ball
    x = xnu  # -xuTarget
    x[0,4]=abs(x[0,4]-ball[0])
    x[0,5] = abs(x[0,5] - ball[1])
    # print mean_squared_error(x[0,:6],[[x[0,0],x[0,1],0,0,0,0]])
    # print (x[0,4],x[0,5])
    A, B = CalculateAB(x, model)  # we get A & B by deriving the Emulator

    K = LqrFhD(A, B, Q, R, 20)

    # we act u, check the reaction ,and update Database
    x = np.transpose(x[0, :6])
    uk = -np.matmul(K,x) * 0.001  # to reduce volume of action because of linearization
    uk = np.transpose(uk)
    xn1 = np.matrix(sample.actUk(uk))
    getError(A, B, model, uk, xnu, xn1)
    xn = np.hstack((xnu[0, :6], uk))
    data.append(xn, xn1)

    if time.time() > t + 60 * 4:
        newState = sample.reset()
        data.append(newState[0], newState[1])
        t = time.time()

    if time.time() > timeout + 60 * 30:
        d = time.gmtime()
        time_stamp = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + ":" + str(d[4])
        model.save('/home/project/PycharmProjects/Reacher/net/'+'cont'+time_stamp)
        timeout = time.time()
