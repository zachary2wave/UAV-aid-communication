import gym
from gym import spaces
import numpy as np
from gym.envs.wlan import env_simulated as env
from gym.envs.wlan import thought_out as tho
from gym.utils import seeding

class JointCP(gym.Env):

    def __init__(self):
        self.Num_AP = 9
        self.Num_UE = 100

        self.channel = [1, 2, 1, 2, 2, 3, 2, 1, 1]
        self.oriTHO = np.zeros([1, self.Num_AP])
        loss_cal = env.Scenario(self.Num_AP, self.Num_UE, freq=2, avr_ap=1)
        self.contact, self.placeAP, self.placeUE, self.Loss = loss_cal.sendout()

        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(self.Num_AP*3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-0, high=1, shape=(self.Num_AP,), dtype=np.float32)

        self.state = self.Num_AP * [0.5]
        envir = tho.ThoughtOutCal(self.channel, [i * 60 for i in self.state], self.Num_AP, self.Num_UE)
        RSSI, Speed, self.connection = envir.subspeed(self.Loss)
        thought_out_ue, P = envir.thomain(Speed, self.connection)
        # 将UE的转化为AP的
        thought_out_AP = np.zeros([self.Num_AP])
        for kki in range(0, self.Num_AP):
            tempN = np.argwhere(self.connection == kki)
            for kkj in tempN:
                thought_out_AP[kki] += thought_out_ue[kkj]
        self.oriTHO[:] = thought_out_AP[:]

    def step(self, u):
        reward = np.zeros([self.Num_AP])
        s_ = np.zeros([self.Num_AP])
        for kk in range(0, self.Num_AP):
            if self.state[kk] + u[kk] < 0:
                s_[kk] = 0
            elif self.state[kk] + u[kk] > 1:
                s_[kk] = 1
            else:
                s_[kk] = self.state[kk] + u[kk]
        envir = tho.ThoughtOutCal(self.channel, [i * 60 for i in s_], self.Num_AP, self.Num_UE)
        RSSI, Speed, connection = envir.subspeed(self.Loss)
        thought_out_ue, P = envir.thomain(Speed, connection)
        # 将UE的转化为AP的
        thought_out_AP = np.zeros([self.Num_AP])
        for kki in range(0, self.Num_AP):
            tempN = np.argwhere(connection == kki)
            for kkj in tempN:
                thought_out_AP[kki] += thought_out_ue[kkj]
        # 计算reward
        # print(thought_out_AP[0])
        for kk in range(0, self.Num_AP):
            if self.state[kk]+u[kk] < 0:
                reward[kk] = -100
            elif self.state[kk]+u[kk] > 1:
                reward[kk] = -100
            else:
                tempppppp = thought_out_AP[kk]
                reward[kk] = tempppppp*6
                # reward[kk] = (thought_out_AP[kk]-self.oriTHO[kk])*10
        # self.oriTHO[:] = thought_out_AP[:]
        # print(s_.shape)
        return s_, np.sum(reward), False, {}

    def reset(self):
        self.state = np.array(self.Num_AP*[0.5])
        # print(self.state.shape)
        return self.state
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def render(self, mode='human'):
        tho.showplot(self.placeAP, self.placeUE, self.state, self.channel, self.connection)
        return {}