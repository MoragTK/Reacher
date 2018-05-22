
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import getpass

username = getpass.getuser()
figPath = '/home/' + username + '/PycharmProjects/Reacher/Figures/'


class PlotData:

    def __init__(self):

        self.frameNumber = "%08d" % 0

        # For Errors:
        self.fig = plt.figure(figsize=(16, 8))
        self.fig=plt.gcf()
        self.fig.show()
        self.fig.canvas.draw()

        self.trainErrHistory = []
        self.onlineErrHistory = []
        self.ltiErrHistory = []
        self.trajectory = self.fig.add_subplot(2, 2, 1)
        self.trainErrGraph = self.fig.add_subplot(2, 2, 3)
        self.onlineErrGraph = self.fig.add_subplot(2, 2, 2)
        self.ltiErrGraph = self.fig.add_subplot(2, 2, 4)




    def updateTrajectoryState(self, X, ball, step, action):
        self.trajectory.cla()
        xLocations = X[:, 4]
        yLocations = X[:, 5]
        self.trajectory.set_title("Planned Trajectory", loc="left")
        self.trajectory.plot(xLocations, yLocations, '.-')
        self.trajectory.set_xlim(-1, 1)
        self.trajectory.set_ylim(-1, 1)
        self.trajectory.plot(ball[0], ball[1], 'go')
        self.trajectory.text(ball[0], ball[1], 'ball')
        self.trajectory.plot(X[0, 4], X[0, 5], 'ro')
        self.trajectory.text(X[0, 4], X[0, 5], 'start')
        self.trajectory.plot(X[-1, 4], X[-1, 5], 'ro')
        self.trajectory.text(X[-1, 4], X[-1, 5], 'End')
        #self.trajectory.text(0.15, 0.12, "Uk: {}".format(action))
        self.frameNumber = "%08d" % step

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

    def updateOnlineHistory(self, newData):
        self.onlineErrGraph.cla()
        self.onlineErrHistory.append(newData)
        X = range(len(np.asarray(self.onlineErrHistory)))
        Y = np.asarray(self.onlineErrHistory)
        self.onlineErrGraph.plot(X, Y, '.-')
        self.onlineErrGraph.set_yscale("log", nonposy='clip')
        self.onlineErrGraph.set_ylim(bottom=10**-7, top=1)
        self.onlineErrGraph.set_title('Online Error History', loc="right")
        #self.onlineErrGraph.set_xlabel('Step')
        self.onlineErrGraph.set_ylabel('MSE Error')

    def updateLTIHistory(self, newData):
        self.ltiErrGraph.cla()
        self.ltiErrHistory.append(newData)
        X = range(len(np.asarray(self.ltiErrHistory)))
        Y = np.asarray(self.ltiErrHistory)
        self.ltiErrGraph.plot(X, Y, '.-')
        self.ltiErrGraph.set_yscale("log", nonposy='clip')
        self.ltiErrGraph.set_ylim(bottom=10**-7, top=1)
        self.ltiErrGraph.set_title('LTI Error History', loc="right")
        self.ltiErrGraph.set_xlabel('Step')
        self.ltiErrGraph.set_ylabel('MSE Error')

    def plot(self):
        #plt.pause(2)
        plt.savefig(figPath + self.frameNumber)
        self.fig.canvas.draw()

    def reset(self):
        self.trainErrHistory = []
        self.onlineErrHistory = []
        self.ltiErrHistory = []
        self.frameNumber = "%08d" % 0