from RealWorldSimulator import RealWorldSimulator
from Emulator import Emulator
from DataSet import DataSet
from Controller import Controller
import numpy as np
import time
import getpass
import glob
import os

username = getpass.getuser()
modelDir = '/home/' + username + '/PycharmProjects/Reacher/Networks/'
list_of_files = glob.glob(modelDir + "*")  # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print latest_file

# Algorithm main building blocks
db = DataSet(size=100000)               # Initializing an empty data base
simulator = RealWorldSimulator()        # Initialize the RealWorldSimulator
emulator = Emulator(new=False, filePath=latest_file)               # Initialize emulator
controller = Controller(emulator, simulator)   # Initialize Controller

# Mode
states = ('INITIALIZE', 'TRAIN', 'RUN')
# Choose state HERE:
state = states[1]

if state == 'INITIALIZE':
    # emulator.restoreModel()
    # Train model with available samples in the data base.
    start = time.time()
    t = time.time()
    while time.time() < start + (1 * 60):

        ## ???
        '''
        xk=simulator.getXk()
        uk=simulator.env.action_space.sample()
        xk1_pred=emulator.predict(xk,uk)
        xk1_real=simulator.actUk(uk)
        plot_next_pos(xk1_pred,xk1_real)
        '''

        # Generate random samples for the initial training and insert them to the data set.
        simulator.generateRandomSamples(db.size, db)
        emulator.train(db, state)
        if time.time() > t + (15 * 60):
            t = time.time()
            d = time.gmtime()
            time_stamp = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + "-" + str(d[4])
            emulator.saveModel(modelDir + "emulator_" + time_stamp)

    state = states[1]


if state == 'TRAIN':
    # In this state, the algorithm performs both ANN Training alongside LQR Control.

    start = time.time()
    t1 = start  # For resetting the environment
    t2 = start  # For saving the model
    t3 = start  # For generating random samples
    sampleGroupSize = 40
    samplesAdded = 0
    while True:
        #print "Sampled added : {}".format(samplesAdded)
        #if samplesAdded == sampleGroupSize:
        #    emulator.train(db, state)
        #    samplesAdded = 0

        xk = simulator.getXk()
        uk = controller.calculateNextAction(xk)
        #print 'uk: ' + str(uk)
        uk = np.reshape(uk, (2, 1))

        xk_uk = np.vstack((simulator.getXk(), np.copy(uk)))
        xk_1 = simulator.actUk(uk)

        simulator.simulate()
        db.append(xk_uk, xk_1)
        samplesAdded += 1

        if time.time() > t1 + (5 * 60):
            simulator.reset()
            t1 = time.time()

        if time.time() > t2 + (60 * 60):
            d = time.gmtime()
            time_stamp = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + "-" + str(d[4])
            emulator.saveModel(modelDir + "emulator_" + time_stamp)
            t2 = time.time()

       # if time.time() > t3 + (5 * 60):
       #     simulator.generateRandomSamples(sampleGroupSize, db)
       #     t3 = time.time()
       #     print "Generating Random Samples"

'''
else state == 'RUN':
    print "Nothing now, predictions later"
'''