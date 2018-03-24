import mujoco_py
import pickle
import numpy as np
import gym
import math
import getpass

username = getpass.getuser()

class DataClass:

    env = gym.make('Reacher-v1')
    in_n = np.matrix([0., 0., 0., 0., 0., 0., 0., 0.])
    observation = np.array(env.reset())
    x_n = np.array([0., 0., 0., 0., 0., 0.])
    action = np.array([0., 0.])  # Uk
    ball = np.array([0.,0.])
    ball[0] = observation[2]
    ball[1] = observation[3]

    # makeData creates and saves a new sample <xn,un,xn+1> of a random action Uk
    def makeData(self, sample):
        self.in_n =np.matrix([0., 0., 0., 0., 0., 0., 0., 0.])
        for i in range(sample):
            self.env.render()
            # create random action
            self.action = self.env.action_space.sample()
            if i == 0:
                self.in_n[0, 0] = np.copy(self.observation[0] % (math.pi*2))  # theta of lower arm
                self.in_n[0, 1] = np.copy(self.observation[1] % (math.pi*2))  # theta of upper arm
                self.in_n[0, 2] = np.copy(self.observation[6])  # v1
                self.in_n[0, 3] = np.copy(self.observation[7])  # v2
                self.in_n[0, 4] = np.copy(self.observation[8])  # finger x
                self.in_n[0, 5] = np.copy(self.observation[9])  # finger y
                self.in_n[0, 6] = np.copy(self.action[0])  # act on lower arm
                self.in_n[0, 7] = np.copy(self.action[1])  # act on upper arm
                self.observation, self.reward, done, info = self.env.step(self.action)
            else:
                self.in_n = np.vstack((self.in_n, np.hstack((self.x_n, self.action))))
                self.observation, self.reward, done, info = self.env.step(self.action)

            self.x_n[0] = np.copy(self.observation[0] % (math.pi*2))  # cos(theta1)
            self.x_n[1] = np.copy(self.observation[1] % (math.pi*2))  # cos(theta2)
            self.x_n[2] = np.copy(self.observation[6])  # v1
            self.x_n[3] = np.copy(self.observation[7])  # v2
            self.x_n[4] = np.copy(self.observation[8])  # delta x
            self.x_n[5] = np.copy(self.observation[9])  # delta y


        file_in = open('/home/' + username + '/PycharmProjects/Reacher/var/in6_x_y', 'w')
        pickle.dump(self.in_n, file_in)

    def getData(self):
        file_out = open('/home/' + username + '/PycharmProjects/Reacher/var/in6_x_y', 'r')
        in_n = np.matrix(pickle.load(file_out))
        len = in_n.__len__()
        x_n_1 = np.matrix(in_n[1:, :6])
        in_n = np.delete(in_n, (len - 1), axis=0)
        return [np.copy(in_n), np.copy(x_n_1)]

    # actUk performs action Uk on the system and returns the new state, Xk+1
    def actUk(self, uk):
        self.action = uk
        self.env.render()
        self.observation, self.reward, done, info = self.env.step(self.action)
        self.x_n[0] = np.copy(self.observation[0] % (math.pi * 2))  # cos(theta1)
        self.x_n[1] = np.copy(self.observation[1] % (math.pi * 2))  # cos(theta2)
        self.x_n[2] = np.copy(self.observation[6])  # v1
        self.x_n[3] = np.copy(self.observation[7])  # v2
        self.x_n[4] = np.copy(self.observation[8])  # delta x
        self.x_n[5] = np.copy(self.observation[9])  # delta y
        return np.copy(self.x_n)

    # in case of an exception we start another env and return the first state and some random action
    # plus the new state we get from the action
    def reset(self):
        #self.env.render()
        self.observation = self.env.reset()  # new xn
        self.x_n[0] = np.copy(self.observation[0] % (math.pi * 2))  # cos(theta1)
        self.x_n[1] = np.copy(self.observation[1] % (math.pi * 2))  # cos(theta2)
        self.x_n[2] = np.copy(self.observation[6])  # v1
        self.x_n[3] = np.copy(self.observation[7])  # v2
        self.x_n[4] = np.copy(self.observation[8])  # delta x
        self.x_n[5] = np.copy(self.observation[9])  # delta y
        self.action = self.env.action_space.sample()  # new un
        self.observation, self.reward, done, info = self.env.step(self.action)  #new Xk+1
        x_n1 = np.array([0., 0., 0., 0., 0., 0.])
        self.x_n[0] = np.copy(self.observation[0] % (math.pi * 2))  # cos(theta1)
        self.x_n[1] = np.copy(self.observation[1] % (math.pi * 2))  # cos(theta2)
        self.x_n[2] = np.copy(self.observation[6])  # v1
        self.x_n[3] = np.copy(self.observation[7])  # v2
        self.x_n[4] = np.copy(self.observation[8])  # delta x
        self.x_n[5] = np.copy(self.observation[9])  # delta y
        x_n1 = np.copy(np.matrix(x_n1))
        xn = np.matrix(np.hstack((self.x_n, self.action)))
        return [np.copy(xn), np.copy(x_n1)]
