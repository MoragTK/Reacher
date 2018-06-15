from RealWorldSimulator import RealWorldSimulator
from Emulator import Emulator
from DataSet import DataSet
from DataPlotter import DataPlotter
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
print "Using Latest network: {}".format(latest_file)


'''Initialize all the algorithms classes'''

db = DataSet(size=10000)
plotter = DataPlotter()
simulator = RealWorldSimulator()
# Initialize the emulator. The 'new' parameter determines whether we're
# starting a completely new network or training on an existing one.
emulator = Emulator(db, plotter, new=False, filePath=latest_file)
controller = Controller(emulator, simulator, plotter)   # Initialize Controller


# Modes
states = ('INITIALIZE', 'RUN')
# Choose state HERE:
state = states[0]

if state == 'INITIALIZE':
    # In this state, the model is trained with random samples from data base.
    start = time.time()
    t = time.time()
    #while time.time() < start + (5 * 60):
    while True:
        # Generate random samples for the initial training and insert them to the data set.
        simulator.generateRandomSamples(db.size, db)
        emulator.train()
        plotter.plot()

        if time.time() > t + (15 * 60):
            t = time.time()
            d = time.gmtime()
            time_stamp = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + "-" + str(d[4])
            emulator.saveModel(modelDir + "emulator_" + time_stamp)

        if (emulator.minTrainError < 1e-4):
            break

    t = time.time()
    d = time.gmtime()
    time_stamp = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + "-" + str(d[4])
    emulator.saveModel(modelDir + "emulator_" + time_stamp)
    state = states[1]

simulator.reset()

if state == 'RUN':
    # In this state, the algorithm uses the learned model and
    # performs iLQR Control in order to get the arm to reach the ball.

    start = time.time()
    t1 = start  # For resetting the environment
    t2 = start  # For saving the model

    sampleGroupSize = 100
    samplesAdded = 0
    enoughSamples = False
    simulator.reset()

    while True:
        if enoughSamples is True:
            if samplesAdded >= sampleGroupSize:
                emulator.train()
                samplesAdded = 0

        xk = simulator.getXk()
        uk = controller.calculateNextAction(xk)
        uk = np.reshape(uk, (2, 1))
        xk_uk = np.vstack((simulator.getXk(), np.copy(uk)))
        xk1 = simulator.actUk(uk)

        emulator.evaluatePredictionError(xk, uk, xk1)
        plotter.plot()

        simulator.simulate()
        db.append(xk_uk, xk1)
        samplesAdded += 1

        if samplesAdded == 1000:
            enoughSamples = True

        # Every 30 seconds, the ball changes location.
        if time.time() > t1 + (30):
            simulator.reset()
            t1 = time.time()
            plotter.reset()

        # Every 30 minutes, the model is saved with a time stamp.
        if time.time() > t2 + (30 * 60):
            d = time.gmtime()
            time_stamp = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + "-" + str(d[4])
            emulator.saveModel(modelDir + "emulator_" + time_stamp)
            t2 = time.time()
