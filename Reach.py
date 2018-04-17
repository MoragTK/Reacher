from RealWorldSimulator import RealWorldSimulator
from Emulator import Emulator
from DataSet import DataSet
from Auxilary import deriveAB
from Controller import Controller
import numpy as np
import time

# Mode
states = ('INITIALIZE', 'TRAIN', 'RUN')
# Choose state HERE:
state = states[1]

# Algorithm main building blocks
db = DataSet(size=1000)               # Initializing an empty data base
simulator = RealWorldSimulator()    # Initialize the RealWorldSimulator
emulator = Emulator()               # Initialize emulator
controller = Controller(emulator)   # Initialize Controller


if state == 'INITIALIZE':

    # Generate random samples for the initial training and insert them to the data set.


    # Train model with available samples in the data base.
    initial_time = time.time()
    t=initial_time
    while time.time() < 1*60*60+initial_time:
        simulator.generateRandomSamples(db.size, db)
        emulator.train(db, state)
        if time.time() > t+(30*60):
            emulator.saveModel()
            t = time.time()
    #state = states[1]

if state == 'TRAIN':
    # In this state, the algorithm performs both ANN Training alongside LQR Control.

    emulator.restoreModel()
    i = 0
    start = time.time()
    t1 = start
    t2 = start
    while True:  # time.time() < 10 * 60 * 60 + start:
        if db.numOfElements == db.size:
            emulator.train(db, state)

        A, B = deriveAB(simulator.getXk(), simulator.getUk(), emulator)
        xTarget = simulator.getXk()
        ball = simulator.getBall()
        xTarget[4, 0] = abs(xTarget[4, 0] - ball[0])
        xTarget[5, 0] = abs(xTarget[5, 0] - ball[1])
        uk = controller.calculateNextAction(A, B, xTarget)
        xk_uk = np.vstack((simulator.getXk(), np.copy(uk)))
        xk_1 = simulator.actUk(uk)
        simulator.simulate()
        #print "Action Taken: {}".format(uk)
        #simulator.printState()
        print "Distance (X,Y): ({},{})".format(xTarget[4, 0], xTarget[5, 0])
        db.append(xk_uk, xk_1)
        if time.time() > t1 + (10 * 60):
            simulator.reset()
            t1 = time.time()

        if time.time() > t2 + (60 * 60):
            emulator.saveModel(property="_LQR_")
            t2 = time.time()
    emulator.saveModel(property="_LQR_")
'''
else state == 'RUN':
    print "Nothing now, predictions later"
'''