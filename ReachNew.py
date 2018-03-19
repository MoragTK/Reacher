
import os
import numpy as np
from Dataclass10 import Rand_data10
import keras
from Database10 import Database10
from function import ABfinder10, setQ, LqrFhD
import time


# for LQR controller
R=np.identity(2)*0.01
Q=np.identity(8)
Q=setQ(Q)
eps=0.1
size=100 # size of Queue


file='/home/control29/PycharmProjects/Reacher/net/'
fileEnd=os.listdir('/home/control29/PycharmProjects/Reacher/net/')[0]
print fileEnd
filepath=file+fileEnd

# create Databae and random data.
data=Database10(size)
sample = Rand_data10()
sample.make_data(size+1)
d = sample.get_data()
X = d[0]
Y = d[1]
# now we first, fill the Database  with sample who come from randon Action
for i in range(0,size):
    data.append(X[size-i-1],Y[size-i-1])


model = keras.models.load_model(filepath)
from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
timeout=time.time()  # limit time of the script
t=timeout
while True:
    input,target=data.getAll()   # we get the all data for training
    model.fit(input, target, batch_size=10, epochs=100, verbose=2)

    # this is [xn,un]
    xnu=data.getLast()
    x=xnu#-xuTarget
    A,B= ABfinder10(x,model) # we get AB by deriviate of the Emulator
    K = LqrFhD(A, B, Q, R, 20)

    # we act u, check the reaction ,and update Database
    x=np.transpose(x[0,:8])
    uk=-(K*x)
    uk=np.transpose(uk)
    xn1=np.matrix(sample.actUk(uk*1))
    xn=np.hstack((xnu[0,:8],uk))
    data.append(xn,xn1)

    if time.time()>t+60*5:
        newState = sample.reset()
        data.append(newState[0], newState[1])
        t=time.time()

    elif time.time()>timeout+60*30.5:
        d = time.gmtime()
        fileEnd = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + ":" + str(d[4])
        model.save('/home/control29/PycharmProjects/Reacher/net/'+'cont'+fileEnd)