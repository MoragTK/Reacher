from DataBase import DataBase
from DataClass import DataClass
from Auxilary import CalculateAB, setQ, LqrFhD, getError,  LqrFhD2,scanUopt
import numpy as np
import keras
import time
import os
from sklearn.metrics import mean_squared_error

# for LQR controller
R = np.identity(2)
R=R*0.01
Q = np.identity(6)
Q = setQ(Q)
size = 150  # size of samples data base

filepath = '/home/london/PycharmProjects/Reacher/net/cont22.3-10_31'

# create Database and random data.
data = DataBase(size)
sample = DataClass()
sample.makeData(size+1)
d = sample.getData()
X =np.copy(d[0])
Y =np.copy(d[1])
# now we first, fill the Database  with sample who come from random Action
for i in range(0, size):
    data.append(np.copy(X[size-i-1]), np.copy(Y[size-i-1]))

model = keras.models.load_model(filepath)
timeout = time.time()  # limit time of the script
t = timeout
while True:
    input, target = data.getAll()   # we get the all data for training
    model.fit(input, target, batch_size=10, epochs=100, verbose=2)

    # this is [xn,un-1]
    #xnu = data.getLast()
    xn = np.copy(np.matrix(sample.x_n))
    un = np.copy(sample.action)
    xnu = np.hstack((xn, np.matrix(un)))
    ball=DataClass.ball

    x=np.copy(xnu)
    x[0,4]=abs(x[0,4]-ball[0])
    x[0,5] = abs(x[0,5] - ball[1])
    A, B = CalculateAB(x, model)  # we get A & B by deriving the Emulator
    K = LqrFhD2(A, B, Q, R,30)
    x = np.copy(np.transpose(x[0, :6]))
    uk = np.copy(-np.matmul(K,np.transpose(xnu[0,:6])))   # to reduce volume of action because of linearization
    uk = np.transpose(uk)*0.1
    #Mse1 = np.matmul((np.matmul(np.transpose(x), Q)), x)
    #Mse2 = np.matmul((np.matmul(uk, R)), np.transpose(uk))
    #print "MSE: " +str(Mse1+Mse2)

    #uk=np.copy(scanUopt(model,xnu,ball))
    xn1 = np.copy(np.matrix(sample.actUk(uk)))
    #getError(A, B, model, np.copy(uk), np.copy(xnu), np.copy(xn1))
    xn = np.hstack((xnu[0, :6], uk))
    data.append(xn, xn1)

    if time.time() > t + 60*4:
        newState = sample.reset()
        data.append(np.copy(newState[0]), np.copy(newState[1]))
        t = time.time()

    if time.time() > timeout + 60*30:
        d = time.gmtime()
        time_stamp = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + ":" + str(d[4])
        model.save('/home/london/PycharmProjects/Reacher/net'+'copy'+time_stamp)
        timeout = time.time()
