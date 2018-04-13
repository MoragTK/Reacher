from RealWorldSimulator import RealWorldSimulator
from Emulator import Emulator
from DataSet import DataSet
from Auxilary import deriveAB
from Controller import Controller
import numpy as np


# Mode
states = ('INITIALIZE', 'TRAIN', 'RUN')
# Choose state HERE:
state = states[0]

# Algorithm main building blocks
db = DataSet(size=10)               # Initializing an empty data base
simulator = RealWorldSimulator()    # Initialize the RealWorldSimulator
emulator = Emulator()               # Initialize emulator
controller = Controller(emulator)   # Initialize Controller


if state == 'INITIALIZE':

    # Generate random samples for the initial training and insert them to the data set.


    # Train model with available samples in the data base.
    for i in range(3):
        simulator.generateRandomSamples(db.size, db)
        emulator.train(db, state)
    state = states[1]

if state == 'TRAIN':
    # In this state, the algorithm performs both ANN Training alongside LQR Control.
    i = 0
    while i < 6:
        emulator.train(db, state)
        A, B = deriveAB(simulator.getXk(), simulator.getUk(), emulator)
        xTarget = simulator.getXk()
        ball = simulator.getBall()
        xTarget[4, 0] = abs(xTarget[4, 0] - ball[0])
        xTarget[5, 0] = abs(xTarget[5, 0] - ball[1])
        uk = controller.calculateNextAction(A, B, xTarget)
        xk_uk = np.vstack((simulator.getXk(), np.copy(uk)))
        xk_1 = simulator.actUk(uk)
        db.append(xk_uk, xk_1)
        i = i + 1
'''
else state == 'RUN':
    print "Nothing now, predictions later"
'''