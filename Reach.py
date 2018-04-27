from RealWorldSimulator import RealWorldSimulator
from Emulator import Emulator
from DataSet import DataSet
from Controller import Controller
from Auxilary import plot_next_pos
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
# Mode
states = ('INITIALIZE', 'TRAIN', 'RUN')
# Choose state HERE:
state = states[1]

# Algorithm main building blocks
db = DataSet(size=300)               # Initializing an empty data base
simulator = RealWorldSimulator()    # Initialize the RealWorldSimulator
emulator = Emulator(new=False,filePath=latest_file)               # Initialize emulator
controller = Controller(emulator, simulator)   # Initialize Controller


if state == 'INITIALIZE':
    # emulator.restoreModel()
    # Train model with available samples in the data base.
    t = time.time()
    while True:
        xk=simulator.getXk()
        uk=simulator.env.action_space.sample()
        xk1_pred=emulator.predict(xk,uk)
        xk1_real=simulator.actUk(uk)
        plot_next_pos(xk1_pred,xk1_real)


        '''
        # Generate random samples for the initial training and insert them to the data set.
        simulator.generateRandomSamples(db.size, db)
        emulator.train(db, state)
        if time.time() > t + (30 * 60):
            t = time.time()
            d = time.gmtime()
            time_stamp = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + "-" + str(d[4])
            emulator.saveModel(modelDir + "emulator_2lyaer" + time_stamp)
            state = states[1]
        '''

if state == 'TRAIN':
    # In this state, the algorithm performs both ANN Training alongside LQR Control.

    start = time.time()
    t1 = start
    t2 = start
    while True:
        if db.numOfElements == db.size:
           emulator.train(db, state)
        xk = simulator.getXk()
        uk = controller.calculateNextAction(xk)
        #print 'uk: ' +str(uk)
        uk = np.reshape(uk, (2, 1))

        xk_uk = np.vstack((simulator.getXk(), np.copy(uk)))
        xk_1 = simulator.actUk(uk)
        ##
        #xk1_pred=emulator.predict(xk,uk)
        #plot_next_pos(xk1_pred,xk_1)
        ##
        simulator.simulate()
        db.append(xk_uk, xk_1)
        if time.time() > t1 + ( 2* 60):
            simulator.reset()
            t1 = time.time()

        if time.time() > t2 + (60 * 60):
            d = time.gmtime()
            time_stamp = str(d[2]) + "." + str(d[1]) + "-" + str(d[3] + 2) + "-" + str(d[4])
            emulator.saveModel(modelDir + "emulator_adam_" + time_stamp)
            t2 = time.time()
'''
else state == 'RUN':
    print "Nothing now, predictions later"
'''