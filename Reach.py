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
db = DataSet(size=50)               # Initializing an empty data base
simulator = RealWorldSimulator()    # Initialize the RealWorldSimulator
emulator = Emulator()               # Initialize emulator
controller = Controller()           # Initialize Controller


if state == 'INITIALIZE':

    # Generate random samples for the initial training and insert them to the data set.
    simulator.generateRandomSamples(db.size, db)

    # Train model with available samples in the data base.
    emulator.train(db, state)    #TODO: Need to train untill accuracy is higher.
    emulator.train(db, state)
    state = 'TRAIN'

if state == 'TRAIN':
    # In this state, the algorithm performs both ANN Training alongside LQR Control.
    while True:
        emulator.train(db, state)
        A, B = deriveAB(simulator.xk, simulator.uk, emulator)
        uk = controller.calculateNextAction(A, B, simulator.xk)
        xk_uk = np.hstack((np.copy(simulator.xk), np.copy(uk)))
        xk_1 = simulator.actUk(uk)
        db.append(xk_uk, xk_1)
        break
'''
else state == 'RUN':
    print "Nothing now, predictions later"
'''