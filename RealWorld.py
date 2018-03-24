import mujoco_py
import pickle
import numpy as np
import gym
import math
import getpass

# Username for storing and loading purposes
username = getpass.getuser()

''' Definitions:

    Xk tensor includes:  
        1. Angle of lower arm
        2. Angle of upper arm
        3. Angular Velocity of lower arm
        4. Angular Velocity of upper arm
        5. Fingertip x location
        6. Fingertip y location
        
    Uk tensor includes:
        1. sd
        2. sd
    
    ball tensor includes:
        1. x location
        2. y location
'''


# This class is the interface between our program and the OpenAI Simulator for the "Reacher" environment

class RealWorld:


    # Constructor
    def __init__(self):

        # define tensors
        self.xk = np.zeros(shape=(1, 6))  # Current State
        self.uk = np.zeros(shape=(1, 2))  # Action to perform on the current state
        self.xk_next = np.zeros(shape=(1, 6))  # next state after action uk is performed
        self.ball = np.zeros(shape=(1, 2))  # ball location
        self.xk_uk_input = np.zeros(shape=(1, 8))  # input to the system

        # Initialize environment
        self.env = gym.make('Reacher-v1')
        self.observation = np.array(env.reset())
        self.ball[0] = self.observation[9]  #TODO: check the indices
        self.ball[1] = self.observation[10]

    # Generates random sampled for the initial data set.
    def generateRandomSamples(self, numOfSamples):
        for i in range(numOfSamples):
            self.uk = self.env.action_space.sample()
            print self.uk  # TODO: delete
            self.xk = self.env.observation_space.sample()
            print self.xk  # TODO: delete

            # TODO:
