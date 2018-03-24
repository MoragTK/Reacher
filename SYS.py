import mujoco_py
import numpy as np
import gym
import math


class SYS:
    env = gym.make('Reacher-v1')
    observation = np.array(env.reset())
    xn = np.matrix([0., 0., 0., 0., 0., 0.])
    un = np.matrix([0., 0.])  # Uk
    ball = np.matrix([0., 0.])
    ball[0, 0] = np.copy(observation[2])
    ball[0, 1] = np.copy(observation[3])
    xn[0, 0] = np.copy(observation[0] % (math.pi * 2))  # cos(theta1)
    xn[0, 1] = np.copy(observation[1] % (math.pi * 2))  # cos(theta2)
    xn[0, 2] = np.copy(observation[6])  # v1
    xn[0, 3] = np.copy(observation[7])  # v2
    xn[0, 4] = np.copy(observation[8])  # delta x
    xn[0, 5] = np.copy(observation[9])  # delta y

    def actUk(self, uk):
        self.un = uk
        self.observation, self.reward, done, info = self.env.step(self.un)
        self.xn[0, 0] = np.copy(self.observation[0] % (math.pi * 2))  # cos(theta1)
        self.xn[0, 1] = np.copy(self.observation[1] % (math.pi * 2))  # cos(theta2)
        self.xn[0, 2] = np.copy(self.observation[6])  # v1
        self.xn[0, 3] = np.copy(self.observation[7])  # v2
        self.xn[0, 4] = np.copy(self.observation[8])  # delta x
        self.xn[0, 5] = np.copy(self.observation[9])  # delta y
        self.env.render()
        return np.copy(self.xn)

    def reset(self):
        self.observation = self.env.reset()  # new xn
        self.xn[0, 0] = np.copy(self.observation[0] % (math.pi * 2))  # cos(theta1)
        self.xn[0, 1] = np.copy(self.observation[1] % (math.pi * 2))  # cos(theta2)
        self.xn[0, 2] = np.copy(self.observation[6])  # v1
        self.xn[0, 3] = np.copy(self.observation[7])  # v2
        self.xn[0, 4] = np.copy(self.observation[8])  # delta x
        self.xn[0, 5] = np.copy(self.observation[9])  # delta y
        self.env.render()
        return np.copy(self.xn)