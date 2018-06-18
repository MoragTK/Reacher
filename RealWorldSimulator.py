import numpy as np
import gym
import getpass

# Username for storing and loading purposes
username = getpass.getuser()

'''
    This class is the interface between our program and the OpenAI Simulator 
    for the "Reacher" environment.
    
    Definitions:

    Xk tensor includes:  
        1. cos(Theta) of inner arm
        2. cos(Theta) of outer arm
        3. sin(Theta) of inner arm
        4. sin(Theta) of outer arm
        5. Fingertip x location
        6. Fingertip y location
        7. Angular Velocity WRT inner arm 
        8. Angular Velocity WRT outer arm


    Uk tensor represents the action to be performed in the current state, and holds values:
        1. moment acted on inner arm
        2. moment acted on outer arm

    Ball tensor includes:
        1. x location of the ball
        2. y location of the ball
'''


class RealWorldSimulator:

    # Constructor
    def __init__(self):
        # define tensors
        self.uDim = 2
        self.xDim = 8
        self.xk = np.zeros(self.xDim)  # Current State
        self.uk = np.zeros(self.uDim)  # Current action taken
        self.ball = np.zeros(self.uDim)  # ball location

        # Initialize environment
        self.env = gym.make('ReacherRandomBall-v0')
        self.observation = np.array(self.env.reset())
        self.deriveXnFromObservation()
        self.ball[0] = self.observation[8]
        self.ball[1] = self.observation[9]

    # Generates random samples for the initial data set.
    def generateRandomSamples(self, numOfSamples, dataBase):
        for i in range(numOfSamples):
            # self.env.render()
            self.uk = self.env.action_space.sample()  # Create a random action uk
            xk_uk_input = np.vstack((self.getXk(), self.getUk()))  # The input vector for the neural network
            self.observation, reward, done, info = self.env.step(self.uk)  # Perform the random action uk
            self.deriveXnFromObservation()
            dataBase.append(xk_uk_input, self.getXk())

    # Generate a random action for the purpose of exploration.
    def generateRandomAction(self):
        self.uk = self.env.action_space.sample()
        return self.uk


    # Performs the action u[k] on the current state (x[k]) and returns the next state (x[k+1]).
    def actUk(self, uk):
        self.uk = uk
        self.observation, reward, done, info = self.env.step(self.uk)
        self.deriveXnFromObservation()
        # self.env.render()
        return self.getXk()

    # Resets the simulator state (in case of exception)
    def reset(self, ballLocation):  # TODO: Check return value dimensions

        if ballLocation == 'Random':
            self.env = gym.make('ReacherRandomBall-v0')

        elif ballLocation == 'Halfway':
            self.env = gym.make('ReacherHalfwayBall-v0')

        elif ballLocation == 'Far':
            self.env = gym.make('ReacherFarBall-v0')

        elif ballLocation == 'Center':
            self.env = gym.make('ReacherCenterBall-v0')

        self.observation = self.env.reset()  # reset the system to a new state
        self.deriveXnFromObservation()




    # Derives the state parameters that are relevant to our program from the current observation tensor.
    # this is done to make sure the values are up to date whenever we want to use them.
    def deriveXnFromObservation(self):
        self.xk[0] = np.copy(self.observation[0])           # cos(Theta) of inner arm
        self.xk[1] = np.copy(self.observation[1])           # cos(Theta) of outer arm
        self.xk[2] = np.copy(self.observation[2])           # sin(Theta) of inner arm
        self.xk[3] = np.copy(self.observation[3])           # sin(Theta) of outer arm
        self.xk[4] = np.copy(self.observation[4]) / 0.21    # fingertip location x
        self.xk[5] = np.copy(self.observation[5]) / 0.21    # fingertip location y
        self.xk[6] = np.copy(self.observation[6]) / 200     # v1 (Angular Velocity of inner arm)
        self.xk[7] = np.copy(self.observation[7]) / 200     # v2 (Angular Velocity of outer arm)
        self.ball[0] = np.copy(self.observation[8]) / 0.21  # ball location x
        self.ball[1] = np.copy(self.observation[9]) / 0.21  # ball location y

    # Returns current state x[k]
    def getXk(self):
        return np.reshape(np.copy(self.xk), (self.xDim, 1))

    # Returns the last action performed - u[k]
    def getUk(self):
        return np.reshape(np.copy(self.uk), (self.uDim, 1))

    # returns the location of the ball.
    def getBall(self):
        return np.copy(self.ball)

    # starts up a GUI with the simulation of the board, arm and ball.
    def simulate(self):
        self.env.render()

    '''def distance(self):
        from math import sqrt
        dis = sqrt(((abs(self.xk[4] - self.ball[0])) ** 2) + ((abs(self.xk[5] - self.ball[1])) ** 2))
        return dis'''