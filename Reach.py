from RealWorldSimulator import RealWorldSimulator
from Emulator import Emulator
from DataSet import DataSet
from DataPlotter import DataPlotter
from ResultsPlotter import ResultsPlotter
from Auxilary import epsilonGreedy
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

# Episode: a period of time that it's length is a constant number of steps defined below.
episodes = 1000
episodeLength = 20
evalReps = 5



'''INITIALIZE'''
# In this state, the model is initialized by training on random samples from the data-base.
db = DataSet(size=10000)
#plotter = DataPlotter()
resultsPlotter = ResultsPlotter()
simulator = RealWorldSimulator()

# Initialize the emulator. The 'new' parameter determines whether we're
# starting a completely new network or training on an existing one.
emulator = Emulator(db, new=True, filePath=latest_file)
controller = Controller(emulator, simulator)

# Generate random samples for the initial training and insert them to the data set.
simulator.generateRandomSamples(db.size, db)
emulator.train()
#plotter.plot()


def runEpisode(reps=1, evaluate=False, ballLocation='Random'):
    costArray = np.zeros(reps)

    for rep in range(reps):

        cost = 0
        simulator.reset(ballLocation)

        for step in range(episodeLength):
            xk = simulator.getXk()

            # if evaluating, do not explore. if not evaluating, explore with probability of 1-prob
            if evaluate or epsilonGreedy(prob=0.9):
                uk, trajectory = controller.calculateNextAction(xk)
            else:
                uk = simulator.generateRandomAction()

            #plotter.updateTrajectoryState(trajectory, simulator.getBall(), step, episodeLength, evaluate)
            #plotter.updateCostHistory(...)    
            #plotter.plot()

            uk = np.reshape(uk, (2, 1))
            xk_uk = np.vstack((simulator.getXk(), np.copy(uk)))
            xk1 = simulator.actUk(uk)

            err = emulator.evaluatePredictionError(xk, uk, xk1)
            #plotter.updateOnlineHistory(err)
            #plotter.plot()
            
            
            simulator.simulate()
            if evaluate is False:
                db.append(xk_uk, xk1)
            else:
                l, _, _, _, _, _ = controller.immediateCost(xk, uk)
                cost += l
        costArray[rep] = cost
    return np.average(costArray)


# In this state, the algorithm uses the learned model and
# performs iLQR Control in order to get the arm to reach the ball.
for episode in range(episodes):

    # Run a regular episode. the data that is generated is saved in the samples buffer.
    runEpisode()

    # Run an evaluation episode, to obtain the current cost. No samples are saved.
    halfwayCost = runEpisode(reps=evalReps, evaluate=True, ballLocation='Halfway')
    farCost = runEpisode(reps=evalReps, evaluate=True, ballLocation='Far')
    centerCost = runEpisode(reps=evalReps, evaluate=True, ballLocation='Center')

    resultsPlotter.updateHalfwayCostHistory(halfwayCost)
    resultsPlotter.updateFarCostHistory(farCost)
    resultsPlotter.updateCenterCostHistory(centerCost)
    resultsPlotter.plot()

    # Train the emulator with new data.
    if episode % 2 == 0:
        trainingErr = emulator.train()
        #plotter.updateTrainingHistory(trainingErr)
        #plotter.plot()


resultsPlotter.saveGraphs("Try")








'''start = time.time()
t1 = start  # For resetting the environment
t2 = start  # For saving the model

sampleGroupSize = 100
samplesAdded = 0
enoughSamples = False

while True:
if enoughSamples is True:
    if samplesAdded >= sampleGroupSize:
        emulator.train()
        samplesAdded = 0

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
    t2 = time.time()'''