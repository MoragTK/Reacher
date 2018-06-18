
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


class ResultsPlotter:

    # Constructor
    def __init__(self):

        # For Errors:
        self.fig = plt.figure(figsize=(20, 4))
        self.fig = plt.gcf()
        self.fig.show()
        self.fig.canvas.draw()

        self.halfwayCostHistory = []
        self.centerCostHistory = []
        self.farCostHistory = []

        self.halfwayCostGraph = self.fig.add_subplot(1, 3, 1)
        self.centerCostGraph = self.fig.add_subplot(1, 3, 2)
        self.farCostGraph = self.fig.add_subplot(1, 3, 3)

    def updateHalfwayCostHistory(self, newData):
        self.halfwayCostGraph.cla()
        self.halfwayCostHistory.append(newData)
        X = range(len(np.asarray(self.halfwayCostHistory)))
        Y = np.asarray(self.halfwayCostHistory)
        self.halfwayCostGraph.plot(X, Y, '.-')
        self.halfwayCostGraph.set_title('Scenario I', loc="right")
        self.halfwayCostGraph.set_xlabel('Episode')
        self.halfwayCostGraph.set_ylabel('Cost')

    def updateFarCostHistory(self, newData):
        self.farCostGraph.cla()
        self.farCostHistory.append(newData)
        X = range(len(np.asarray(self.farCostHistory)))
        Y = np.asarray(self.farCostHistory)
        self.farCostGraph.plot(X, Y, '.-')
        self.farCostGraph.set_title('Scenario II', loc="right")
        self.farCostGraph.set_xlabel('Episode')
        self.farCostGraph.set_ylabel('Cost')

    # Updates the planned trajectory's cost graph.
    def updateCenterCostHistory(self, newData):
        self.centerCostGraph.cla()
        self.centerCostHistory.append(newData)
        X = range(len(np.asarray(self.centerCostHistory)))
        Y = np.asarray(self.centerCostHistory)
        self.centerCostGraph.plot(X, Y, '.-')
        self.centerCostGraph.set_title('Scenario III', loc="right")
        self.centerCostGraph.set_xlabel('Episode')
        self.centerCostGraph.set_ylabel('Cost')

    # When called, plots all graphs, and saves them to a png file (Later all images can be turned into a gif)
    def plot(self):
        self.fig.canvas.draw()

    # Resets all graphs and the frame number.
    def reset(self):
        self.halfwayCostHistory = []
        self.centerCostHistory = []
        self.farCostHistory = []

    #saves the graph into a png
    def saveGraphs(self, name):
        plt.savefig(figPath + name)
