
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import getpass

username = getpass.getuser()
figPath = '/home/' + username + '/PycharmProjects/Reacher/Figures/'


'''

This class is responsible for all plotting of the data.
It plots 4 graphs:

    1. Model's Training Error  
    2. Model's Online Error
    3. Planned Trajectory
    4. Cost of planned trajectory in the current step

'''

class DataPlotter:

    # Constructor
    def __init__(self):

        self.frameNumber = "%08d" % 0

        # For Errors:
        self.fig = plt.figure(figsize=(16, 8))
        self.fig = plt.gcf()
        self.fig.show()
        self.fig.canvas.draw()

        self.trainErrHistory = []
        self.onlineErrHistory = []
        self.ltiErrHistory = []
        self.costHistory = []
        self.trajectoryGraph = self.fig.add_subplot(2, 2, 1)
        self.trainErrGraph = self.fig.add_subplot(2, 2, 2)
        self.onlineErrGraph = self.fig.add_subplot(2, 2, 3)
        self.costGraph = self.fig.add_subplot(2, 2, 4)
        #self.ltiErrGraph = self.fig.add_subplot(2, 2, 4)

    # Updates the planned trajectory graph.
    def updateTrajectoryState(self, X, ball, step, N, evaluate):
        self.trajectoryGraph.cla()
        xLocations = X[:, 4]
        yLocations = X[:, 5]
        title = "Planned Trajectory, Step {} of {}. ".format(step, N)
        if evaluate is True:
            title = title + "(Evaluating)"
        self.trajectoryGraph.set_title(title, loc="left")
        self.trajectoryGraph.plot(xLocations, yLocations, '.-')
        self.trajectoryGraph.set_xlim(-1.1, 1.1)
        self.trajectoryGraph.set_ylim(-1.1, 1.1)
        self.trajectoryGraph.plot(ball[0], ball[1], 'go')
        self.trajectoryGraph.text(ball[0], ball[1], 'ball')
        self.trajectoryGraph.plot(X[0, 4], X[0, 5], 'ro')
        self.trajectoryGraph.text(X[0, 4], X[0, 5], 'start')
        self.trajectoryGraph.plot(X[-1, 4], X[-1, 5], 'ro')
        self.trajectoryGraph.text(X[-1, 4], X[-1, 5], 'End')
        self.frameNumber = "%08d" % step

    # Updates the model's training errors history graph.
    def updateTrainingHistory(self, newData):
        self.trainErrGraph.cla()
        self.trainErrHistory = self.trainErrHistory + newData
        X = range(len(np.asarray(self.trainErrHistory)))
        Y = np.asarray(self.trainErrHistory)
        self.trainErrGraph.plot(X, Y, '.-')
        self.trainErrGraph.set_yscale("log", nonposy='clip')
        self.trainErrGraph.set_ylim(bottom=10**-7, top=1)
        self.trainErrGraph.set_title('Training Error History', loc="left")
        self.trainErrGraph.set_xlabel('Epoch')
        self.trainErrGraph.set_ylabel('MSE Error')

    # Updates the model's online errors history graph.
    def updateOnlineHistory(self, newData):
        self.onlineErrGraph.cla()
        self.onlineErrHistory.append(newData)
        X = range(len(np.asarray(self.onlineErrHistory)))
        Y = np.asarray(self.onlineErrHistory)
        self.onlineErrGraph.plot(X, Y, '.-')
        self.onlineErrGraph.set_yscale("log", nonposy='clip')
        self.onlineErrGraph.set_ylim(bottom=10**-7, top=1)
        self.onlineErrGraph.set_title('Online Error History', loc="right")
        self.onlineErrGraph.set_xlabel('Step')
        self.onlineErrGraph.set_ylabel('MSE Error')

    # Updates the planned trajectory's cost graph.
    def updateCostHistory(self, newData):
        self.costGraph.cla()
        self.costHistory.append(newData)
        X = range(len(np.asarray(self.costHistory)))
        Y = np.asarray(self.costHistory)
        self.costGraph.plot(X, Y, '.-')
        #self.costGraph.set_yscale("log", nonposy='clip')
        #self.costGraph.set_ylim(bottom=10 ** -7, top=1)
        self.costGraph.set_title('Cost History', loc="right")
        self.costGraph.set_xlabel('Episode')
        self.costGraph.set_ylabel('Cost')

    # When called, plots all graphs, and saves them to a png file (Later all images can be turned into a gif)
    def plot(self):
        plt.savefig(figPath + self.frameNumber)
        self.fig.canvas.draw()

    # Resets all graphs and the frame number.
    def reset(self):
        self.trainErrHistory = []
        self.onlineErrHistory = []
        self.ltiErrHistory = []
        self.frameNumber = "%08d" % 0

    # An older graph plotting function that we used to verify the state space matrices A,B.
    '''def updateLTIHistory(self, newData):
        self.ltiErrGraph.cla()
        self.ltiErrHistory.append(newData)
        X = range(len(np.asarray(self.ltiErrHistory)))
        Y = np.asarray(self.ltiErrHistory)
        self.ltiErrGraph.plot(X, Y, '.-')
        self.ltiErrGraph.set_yscale("log", nonposy='clip')
        self.ltiErrGraph.set_ylim(bottom=10**-7, top=1)
        self.ltiErrGraph.set_title('LTI Error History', loc="right")
        self.ltiErrGraph.set_xlabel('Step')
        self.ltiErrGraph.set_ylabel('MSE Error')'''