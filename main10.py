from Auxilary10 import CalculateAB, setQ, LqrFhD, getError, LqrFhD2, scanUopt, LqrFhD3
import numpy as np
import keras
import time
from sklearn.metrics import mean_squared_error
from DataBase10 import DataBase10
from SYS10 import SYS10
import getpass


username = getpass.getuser()

size=200
data=DataBase10(size)
R = np.identity(2)*0.1
Q = np.identity(8)
Q = setQ(Q)

filepath = '/home/london/PycharmProjects/Reacher/net/18.3-10_16' #6xy21.3-9_37'

sys =SYS10()
model = keras.models.load_model(filepath)
timeout = time.time()  # limit time of the script
t = timeout
k=0
while True:

    '''
    if k>size:
        input,target = data.getAll()  # we get the all data for training
        model.fit(input, target, batch_size=10, epochs=100, verbose=0)
    k=k+1
    '''
    sys.env.render()
    xn = sys.xn
    un = sys.un
    ball=sys.ball
    A, B = CalculateAB(np.hstack((xn,un)), model)  # we get A & B by deriving the Emulator
    K = LqrFhD(A, B, Q, R, 8)
    un1=-np.matmul(K,np.transpose(xn))

    xn1 =sys.actUk(np.copy(np.transpose(un1)))
    xn1pred=model.predict(np.hstack((xn,np.transpose(un1))))
    print (format(mean_squared_error(xn1,xn1pred),'.4f'))
    xn1Lti=(np.matmul(A,np.transpose(xn))+np.matmul(B,un1))
    print (format(mean_squared_error(xn1Lti,np.transpose(xn1)),'.4f'))
    data.append(np.hstack((xn,np.transpose(un1))),xn1)
    if time.time() > t + 100:
        newState = sys.reset()
        t = time.time()

    if time.time() > timeout + 60*30:
        d = time.gmtime()
        time_stamp = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + ":" + str(d[4])
        model.save('/home/london/PycharmProjects/Reacher/net/'+'sys'+time_stamp)
        timeout = time.time()
