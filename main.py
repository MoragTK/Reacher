from Auxilary import CalculateAB, setQ, LqrFhD, getError,  LqrFhD2,scanUopt
import numpy as np
import keras
import time

from DataBase import DataBase
from SYS import SYS

size=100
data=DataBase(size)
R = np.identity(2)*0.01
Q = np.identity(6)
Q = setQ(Q)

filepath = '/home/london/PycharmProjects/Reacher/net/6xy21.3-9_37'

sys =SYS()
model = keras.models.load_model(filepath)
timeout = time.time()  # limit time of the script
t = timeout
k=0
while True:
    ''''
    if k>size:
        input,target = data.getAll()  # we get the all data for training
        model.fit(input, target, batch_size=10, epochs=100, verbose=2)
    k=k+1
    '''
    xn = sys.xn
    un = sys.un
    ball=sys.ball
    '''''
    x=np.matrix(np.copy(xn))
    x[0,4]=np.copy(xn[0,4])-np.copy(ball[0,0])
    x[0, 5] = np.copy(xn[0, 5]) - np.copy(ball[0, 1])
    A, B = CalculateAB(np.hstack((x,un)), model)  # we get A & B by deriving the Emulator
    K = LqrFhD2(A, B, Q, R, 8)
    un1=-np.matmul(K,np.transpose(x))
    '''
    un1=scanUopt(model,xn,ball)
    xn1 =sys.actUk(np.copy(np.transpose(un1)))
    data.append(np.hstack((xn,un1)),xn1)
    if time.time() > t + 100:
        newState = sys.reset()
        t = time.time()

    if time.time() > timeout + 60*30:
        d = time.gmtime()
        time_stamp = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + ":" + str(d[4])
        model.save('/home/london/PycharmProjects/Reacher/net'+'copy'+time_stamp)
        timeout = time.time()
