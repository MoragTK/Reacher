import numpy as np
import gym
import getpass

# Username for storing and loading purposes
username = getpass.getuser()

''' Definitions:

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
    
    ball tensor includes:
        1. x location
        2. y location
'''


# This class is the interface between our program and the OpenAI Simulator for the "Reacher" environment

class RealWorldSimulator:

    # Constructor
    def __init__(self):

        # define parameters:
        self.maxSpeed = 200

        # define tensors
        self.xk = np.zeros(8)    # Current State
        self.uk = np.zeros(2)    # Current action taken
        self.ball = np.zeros(2)  # ball location

        # Initialize environment
        self.env = gym.make('Reacher-v1')
        self.observation = np.array(self.env.reset())
        self.deriveXnFromObservation()
        self.ball[0] = self.observation[8]  # TODO: check the indices
        self.ball[1] = self.observation[9]

    # Generates random samples for the initial data set.
    def generateRandomAction(self):
            return self.env.action_space.sample()                        # Create a random action uk

    # Performs the action uk on the current state and returns the next state.
    def actUk(self, uk):
        self.uk = uk
        self.observation, reward, done, info = self.env.step(self.uk)
        self.deriveXnFromObservation()
        #self.env.render()
        return np.copy(self.xk)

    # Resets the simulator state (in case of exception)
    def reset(self):
        self.observation = self.env.reset()  # reset the system to a new state
        self.deriveXnFromObservation()
        self.env.render()
        return np.copy(self.xn)

    # Derives the state parameters that are relevant to our program from the current observation tensor.
    def deriveXnFromObservation(self):
        self.xk[0] = np.copy(self.observation[0])  # cos(Theta) of inner arm
        self.xk[1] = np.copy(self.observation[1])  # cos(Theta) of outer arm
        self.xk[2] = np.copy(self.observation[2])  # sin(Theta) of inner arm
        self.xk[3] = np.copy(self.observation[3])  # sin(Theta) of outer arm
        self.xk[4] = np.copy(self.observation[4])  # fingertip location x
        self.xk[5] = np.copy(self.observation[5])  # fingertip location y
        self.xk[6] = np.copy(self.observation[6])/self.maxSpeed  # v1 (Angular Velocity of lower arm)
        self.xk[7] = np.copy(self.observation[7])/self.maxSpeed  # v2 (Angular Velocity of upper arm)
        self.ball[0] = np.copy(self.observation[8])  # ball location x
        self.ball[1] = np.copy(self.observation[9])  # ball location y
        # TODO: do a sanity check that for a certain observation, the values make sense


