import mujoco_py
import numpy as np
import gym
import math


class SYS10:
    env = gym.make('Reacher-v1')
    observation = np.array(env.reset())
    xn = np.matrix([0., 0., 0., 0., 0., 0.,0.,0.])
    un = np.matrix([0., 0.])  # Uk
    ball = np.matrix([0., 0.])
    ball[0, 0] = np.copy(observation[2])
    ball[0, 1] = np.copy(observation[3])
    xn[0, 0] = np.copy(observation[0] % (math.pi * 2))  # cos(theta1)
    xn[0, 1] = np.copy(observation[1] % (math.pi * 2))  # cos(theta2)
    xn[0, 2] = np.copy(observation[6])  # v1
    xn[0, 3] = np.copy(observation[7])  # v2
    xn[0, 4] = np.copy(observation[4])  # delta x
    xn[0, 5] = np.copy(observation[5])  # delta y
    xn[0, 6] = np.copy(observation[2])  # ballx
    xn[0, 7] = np.copy(observation[3])  # bally


    def actUk(self, uk):
        self.env.render()
        self.un = uk
        self.observation, self.reward, done, info = self.env.step(self.un)
        self.xn[0, 0] = np.copy(self.observation[0] % (math.pi * 2))  # cos(theta1)
        self.xn[0, 1] = np.copy(self.observation[1] % (math.pi * 2))  # cos(theta2)
        self.xn[0, 2] = np.copy(self.observation[6])  # v1
        self.xn[0, 3] = np.copy(self.observation[7])  # v2
        self.xn[0, 4] = np.copy(self.observation[4])  # delta x
        self.xn[0, 5] = np.copy(self.observation[5])  # delta y
        self.xn[0, 6] = np.copy(self.observation[2])  # ballx
        self.xn[0, 7] = np.copy(self.observation[3])  # bally
        return np.copy(self.xn)

    def reset(self):
        self.observation = self.env.reset()  # new xn
        self.xn[0, 0] = np.copy(self.observation[0] % (math.pi * 2))  # cos(theta1)
        self.xn[0, 1] = np.copy(self.observation[1] % (math.pi * 2))  # cos(theta2)
        self.xn[0, 2] = np.copy(self.observation[6])  # v1
        self.xn[0, 3] = np.copy(self.observation[7])  # v2
        self.xn[0, 4] = np.copy(self.observation[4])  # delta x
        self.xn[0, 5] = np.copy(self.observation[5])  # delta y
        self.xn[0, 6] = np.copy(self.observation[2])  # ballx
        self.xn[0, 7] = np.copy(self.observation[3])  # bally
        self.env.render()
        return np.copy(self.xn)