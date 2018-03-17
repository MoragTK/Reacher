import mujoco_py
import pickle
import numpy as np
import gym
import math

class Rand_data:
    env = gym.make('Reacher-v2')
    in_n=np.matrix([0.,0.,0.,0.,0.,0.,0.,0.]) #8
    observation=np.array(env.reset()) # 11
    x_n=np.array([0.,0.,0.,0.,0.,0.])
    action=np.array([0.,0.])
    def make_data(self,sample):
        for i in range(sample):
            self.env.render()
            action = self.env.action_space.sample()
            if (i==0):
                self.in_n[0,0] = self.observation[0]%(math.pi*2)
                self.in_n[0,1] = self.observation[1]%(math.pi*2)
                self.in_n[0,2] = self.observation[6]
                self.in_n[0,3] = self.observation[7]
                self.in_n[0,4] = self.observation[8] # delta x
                self.in_n[0,5] = self.observation[9] # delta y
                self.in_n[0,6] = self.action[0]
                self.in_n[0,7] = self.action[1]
                self.observation,self.reward, done, info = self.env.step(action)

            else:
                self.in_n = np.vstack((self.in_n, np.hstack((self.x_n, action))))
                self.observation,self.reward, done, info = self.env.step(action)
            self.x_n[0] = self.observation[0]%(math.pi*2)  # cost1
            self.x_n[1] = self.observation[1]%(math.pi*2)  # cost2
            self.x_n[2] = self.observation[6]  # v1
            self.x_n[3] = self.observation[7]  # v2
            self.x_n[4] = self.observation[8]  # delta x
            self.x_n[5] = self.observation[9]  # delta y

        file_in=open('~/PycharmProjects/Reacher/var/in','w')
        pickle.dump(self.in_n,file_in)

    def get_data(self):
        file_out = open('~/PycharmProjects/Reacher/var/in', 'r')
        in_n = np.matrix(pickle.load(file_out))
        len = in_n.__len__()
        x_n_1 = np.matrix(in_n[1:, :6])
        in_n = np.delete(in_n, (len - 1), axis=0)
        return [in_n,x_n_1]
    def actUk(self,uk):
        action=uk
        self.env.render()
        self.observation, self.reward, done, info = self.env.step(action)
        self.x_n[0] = self.observation[0] % (math.pi * 2)  # cost1
        self.x_n[1] = self.observation[1] % (math.pi * 2)  # cost2
        self.x_n[2] = self.observation[6]  # v1
        self.x_n[3] = self.observation[7]  # v2
        self.x_n[4] = self.observation[8]  # delta x
        self.x_n[5] = self.observation[9]  # delta y
        return self.x_n
    def reset(self):
        self.env.render()
        self.observation=self.env.reset()  # new xn
        self.x_n[0] = self.observation[0] % (math.pi * 2)  # cost1
        self.x_n[1] = self.observation[1] % (math.pi * 2)  # cost2
        self.x_n[2] = self.observation[6]  # v1
        self.x_n[3] = self.observation[7]  # v2
        self.x_n[4] = self.observation[8]  # delta x
        self.x_n[5] = self.observation[9]  # delta y
        action = self.env.action_space.sample()  # new un
        self.observation, self.reward, done, info = self.env.step(action) #new xn1
        x_n1=np.array([0.,0.,0.,0.,0.,0.])
        x_n1[0] = self.observation[0] % (math.pi * 2)  # cost1
        x_n1[1] = self.observation[1] % (math.pi * 2)  # cost2
        x_n1[2] = self.observation[6]  # v1
        x_n1[3] = self.observation[7]  # v2
        x_n1[4] = self.observation[8]  # delta x
        x_n1[5] = self.observation[9]  # delta y
        x_n1=np.matrix(x_n1)
        xn=np.matrix(np.hstack((self.x_n, action)))
        return [xn,x_n1]