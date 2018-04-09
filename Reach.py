from RealWorldSimulator import RealWorldSimulator
from Emulator import Emulator
from DataSet import DataSet
from Auxilary import deriveAB, generateRandomSamples
from Controller import Controller
import numpy as np
from sklearn.metrics import mean_squared_error
import time

# Mode
states = ('INITIALIZE', 'TRAIN', 'RUN')
# Choose state HERE:
state = states[0]

# Algorithm main building blocks
db = DataSet(size=2000)               # Initializing an empty data base
simulator = RealWorldSimulator()    # Initialize the RealWorldSimulator
emulator = Emulator()               # Initialize emulator
controller = Controller(emulator)   # Initialize Controller


if state == 'INITIALIZE':

    # Generate random samples for the initial training and insert them to the data set.
    # Train model with available samples in the data base.
    initial_time=time.time()
    t=initial_time
    while time.time()<15*60*60+initial_time:
        generateRandomSamples(db.size, db, simulator)
        emulator.train(db, state)
        if time.time()>t+(30*60):
            emulator.saveModel()
            t=time.time()
    #state = states[1]
    '''
    for i in range(10):
        xk = simulator.xk
        uk = simulator.generateRandomAction()
        xk_1_real = simulator.actUk(uk)
        xk_1_net = emulator.predict(np.matrix(xk),np.matrix(uk))
        mse = mean_squared_error(np.matrix(xk_1_real),np.matrix(xk_1_net))
        print "MSE: {}".format(mse)
    '''
if state == 'TRAIN':
    # In this state, the algorithm performs both ANN Training alongside LQR Control.

    emulator.restoreModel()
    i = 0
    while i < 100:
        if db.numOfElements == db.size:
            emulator.train(db, state)

        xTarget=np.copy(simulator.xk)
        xTarget[4] = abs(xTarget[4]-simulator.ball[0])
        xTarget[5] = abs(xTarget[5]-simulator.ball[1])

        A, B = deriveAB(simulator.xk, simulator.uk, emulator)
        #print xTarget
        print "###############   A    ###############"
        print A
        F = controller.lqrFhd(A, B)
        uk = -np.matmul(F, xTarget)


        #uk = controller.calculateNextAction(A, B, simulator.xk)
        xk_uk = np.hstack((np.copy(simulator.xk), np.copy(uk)))
        xk_1 = simulator.actUk(uk)
        db.append(xk_uk, xk_1)
        i += 1
'''
else state == 'RUN':
    print "Nothing now, predictions later"
'''



