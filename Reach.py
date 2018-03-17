import os
#import control as control
import numpy as np
from DataBase import DataBase
from DataClass import Rand_data
import keras
from CalculateAB import CalculateAB
import time
#import slycot0
from LQR import dlqr
import controlpy

#Parameters
R=np.matrix([[1, 0], [0, 1]])
Q=np.matrix([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 5, 0, 0, 0], [0, 0, 0, 5, 0, 0], [0, 0, 0, 0, 10, 0], [0, 0, 0, 0, 0, 10]])
N=np.zeros((6, 2))
file = '~/PycharmProjects/Reacher/net/'
fileEnd = os.listdir('~/PycharmProjects/Reacher/net/')[0]
print fileEnd
size = 100
filepath = file + fileEnd
data=DataBase(size)
sample = Rand_data()
sample.make_data(size+1)
d = sample.get_data()
X = d[0]
Y = d[1]
for i in range(0, size):
    data.append(X[size-i-1], Y[size-i-1])
flag = 0
while flag == 0:

    model = keras.models.load_model(filepath)

    timeout = time.time()
    while True:
        input, target = data.getAll()
        model.fit(input, target, batch_size=10, epochs=100, verbose=2)
        xnu = data.getLast()
        A, B = CalculateAB(xnu, model)
        try:
            K, X, eigVals = controlpy.synthesis.controller_lqr_discrete_time(A, B, Q, R)
        except:
            os.remove(filepath)
            d = time.gmtime()
            fileEnd = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + ":" + str(d[4])
            filepath = file + fileEnd
            model.save(filepath)
            newState = sample.reset()
            data.append(newState[0], newState[1])
            print "Value error"
            break
        x = np.transpose(xnu[0, :6])
        uk = -(K*x)
        uk = np.transpose(uk)
        xn1 = np.matrix(sample.actUk(uk*1))
        xn = np.hstack((xnu[0,:6],uk))
        data.append(xn, xn1)
        if time.time() > timeout + 60 * 5:
            flag = 1
            break
