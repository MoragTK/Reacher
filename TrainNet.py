import time
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from DataClass import DataClass
from Auxilary import getAll
from collections import deque

size = 30
sample = 100
Q = deque(maxlen=size)
data = DataClass()
for i in range(0, size):
    data.makeData(sample+1)
    d = data.getData()

    X = d[0]
    Y = d[1]
    Q.append(np.hstack((X, Y)))

model = Sequential()
model.add(Dense(50, input_dim=8, kernel_initializer='normal', activation='relu'))
model.add(Dense(14, input_dim=25, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal'))
optimizer = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(loss='mean_squared_error',optimizer=optimizer)

timeout = time.time()
t = timeout
while True:
    input, target = getAll(Q, size)
    model.fit(input, target, batch_size=10, epochs=50, verbose=2)
    data.makeData(sample + 1)
    d = data.getData()
    X = d[0]
    Y = d[1]
    Q.append(np.hstack((X, Y)))

    if time.time() > t + 60 * 60:
        d = time.gmtime()
        time_stamp = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + ":" + str(d[4])
        model.save('/home/' + username + '/PycharmProjects/Reacher/net/' + '6xy' + time_stamp)
        t=time.time()
        continue

    if time.time() > timeout+60*5:
        d = time.gmtime()
        time_stamp = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + ":" + str(d[4])
        model.save('/home/' + username + '/PycharmProjects/Reacher/net/' + '6xy' + time_stamp)