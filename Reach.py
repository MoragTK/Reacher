from RealWorldSimulator import RealWorldSimulator
from Emulator import Emulator
from DataSet import DataSet
from DataPlotter import DataPlotter
from ResultsPlotter import ResultsPlotter
from Auxilary import epsilonGreedy
from Controller import Controller
import numpy as np
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
episodeLength = 25
evalReps = 4



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
#emulator.train()
#plotter.plot()


def runEpisode(reps=1, evaluate=False, ballLocation='Random', prob=1):
    costArray = np.zeros(reps)

    for rep in range(reps):

        cost = 0
        simulator.reset(ballLocation)

        for step in range(episodeLength):
            xk = simulator.getXk()

            # if evaluating, do not explore. if not evaluating, explore with probability of 1-prob
            if epsilonGreedy(prob=prob):
                uk, trajectory = controller.calculateNextAction(xk)
            else:
                uk = simulator.generateRandomAction()

            #plotter.updateTrajectoryState(trajectory, simulator.getBall(), step, episodeLength, evaluate)
            #plotter.updateCostHistory(...)    
            #plotter.plot()

            uk = np.reshape(uk, (2, 1))
            xk_uk = np.vstack((simulator.getXk(), np.copy(uk)))
            xk1 = simulator.actUk(uk)

            #err = emulator.evaluatePredictionError(xk, uk, xk1)
            #plotter.updateOnlineHistory(err)
            #plotter.plot()

            if rep == 0 and evaluate is True: #Simulate only one rep in each scenario
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
halfwayCost = []
farCost = []
centerCost = []

for episode in range(episodes):
    print "Episode: {}".format(episode)
    # Run a regular episode. the data that is generated is saved in the samples buffer.

    if episode < 10:
        p = 0.3
    elif (episode > 50 and episode < 100):
        p = 0.7
    else:
        p = 0.9

    runEpisode(prob=p)


    # Run an evaluation episode, to obtain the current cost. No samples are saved.
    halfwayCost.append(runEpisode(reps=evalReps, evaluate=True, ballLocation='Halfway', prob=1))
    farCost.append(runEpisode(reps=evalReps, evaluate=True, ballLocation='Far', prob=1))
    centerCost.append(runEpisode(reps=evalReps, evaluate=True, ballLocation='Center', prob=1))

    #resultsPlotter.updateHalfwayCostHistory(halfwayCost)
    #resultsPlotter.updateFarCostHistory(farCost)
    #resultsPlotter.updateCenterCostHistory(centerCost)
    #resultsPlotter.plot()

    # Train the emulator with new data.
    if episode % 5 == 0:
        trainingErr = emulator.train()
        print halfwayCost
        print farCost
        print centerCost
        #plotter.updateTrainingHistory(trainingErr)
        #plotter.plot()
        #resultsPlotter.saveGraphs("progress")

resultsPlotter.updateHalfwayCostHistory(halfwayCost)
resultsPlotter.updateFarCostHistory(farCost)
resultsPlotter.updateCenterCostHistory(centerCost)
resultsPlotter.plot()

resultsPlotter.saveGraphs("Final")

