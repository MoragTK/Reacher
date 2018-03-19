#from Database import Database
#from DataClass import Rand_data
#import control as control
#import slycot0
#from func2 import dlqr
#import controlpy #TODO: Do we need this

import os
import numpy as np
import keras
import time
from DataClass import DataClass
from DataBase import DataBase
from Auxilary import CalculateAB, setQ, LqrFhD

# for LQR controller
R = np.identity(2) * 0.2
Q = np.identity(8)
#Q=setQ(Q)
eps = 0.1
size = 100  # size of samples data base


file = '~/PycharmProjects/Reacher/net/'
fileEnd = os.listdir('~/PycharmProjects/Reacher/net/')[0]
print fileEnd
filepath = file+fileEnd

# create Databae and random data.
data=DataBase(size)
sample = DataClass()
sample.makeData(size+1)
d = sample.getData()
X = d[0]
Y = d[1]
# now we first, fill the Database  with sample who come from randon Action
for i in range(0,size):
    data.append(X[size-i-1],Y[size-i-1])


flag=0
while flag==0:
    model = keras.models.load_model(filepath)
    from keras.utils import plot_model
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    timeout=time.time()  # limit time of the script
    while True:
        input,target=data.getAll()   # we get the all data for training
        model.fit(input, target, batch_size=10, epochs=100, verbose=0)

        # this is [xn,un]
        xnu=data.getLast()
        xuTarget = np.matrix([xnu[0,0]-eps,xnu[0,1]-eps,0,0,0,0,xnu[0,6]-eps,xnu[0,7]-eps,0,0])
        x=xnu#-xuTarget
        A,B= CalculateAB(x,model) # we get AB by deriviate of the Emulator
        #try:
            #K, X, eigVals=controlpy.synthesis.controller_lqr_discrete_time(A,B,Q,R)
        K=LqrFhD(A,B,Q,R,15)
        #except ValueError:
        '''''
            os.remove(filepath)
            d = time.gmtime()
            fileEnd = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + ":" + str(d[4])
            filepath=file+fileEnd
            model.save(filepath)
            newState=sample.reset()
            data.append(newState[0],newState[1])
            print "Value error"
            break
        '''
        # we act u, check the reaction ,and update Database
        x=np.transpose(x[0,:8])
        uk=-(K*x)
        uk=np.transpose(uk)
        xn1=np.matrix(sample.actUk(uk*1))
        xn=np.hstack((xnu[0,:8],uk))
        data.append(xn,xn1)

        if time.time()>timeout+60*5:
            flag=1
            break