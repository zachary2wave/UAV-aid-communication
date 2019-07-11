import gym
from gym import spaces
import numpy as np
from gym.envs.WLANPaper import Scenario as Env
from gym.utils import seeding

class MapEnv(gym.Env):

    def __init__(self):
        self.Num_AP = 9
        self.Num_UE = 100
        self.env = Env(self.Num_AP, self.Num_UE, 2, 1)
        self.action_space_P = spaces.Box(low=0, high=1, shape=(self.Num_AP,), dtype=np.float32)
        self.action_space_C = spaces.Box(low=0, high=1, shape=(self.Num_AP,), dtype=np.float32)
        self.action_space = [self.action_space_P, self.action_space_C]
        self.observation_space_P = spaces.Box(low=0, high=1, shape=(self.Num_AP,), dtype=np.float32)
        self.observation_space_C = spaces.Box(low=0, high=1, shape=(self.Num_AP,), dtype=np.float32)
        self.observation_space = [self.observation_space_P, self.observation_space_C]
    def reset(self):
        self.state = np.array(self.Num_AP * 2 * [0.5])
        # print(self.state.shape)
        return self.state

    def step(self, u):
        P = u[0:self.Num_AP]
        C = u[self.Num_AP:2*self.Num_AP]
        through = self.env.calculation_NP(P, C)
        reward = through
        s_ = u
        return s_, reward, False, {}
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def render(self, mode='human'):
        # tho.showplot(self.placeAP, self.placeUE, self.state, self.channel, self.connection)
        return {}