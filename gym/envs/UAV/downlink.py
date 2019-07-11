import gym
from gym import spaces
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from gym.utils import seeding
# import panda as pd

'''
本环境是利用下行链路
每一次计算强化学习动作是一个时隙。
这个时隙之内速度，位置不变？

此版本为最简单的版本
支持断点重传
始终是跟最大的相连
'''

class Downlink(gym.Env):
    def __init__(self):
        # UAV 及 用户参数
        self.NUAV = 1
        self.NSP = 5
        self.Vmax = 30                                 # 最大速度    in m/s
        self.Vmaxz = 10
        self.amax = 5                                  # 最大加速度  in m^2/s
        self.delta = 0.1                               # 时隙
        self.T = 1000                                  # 总时间

        self.N = self.T/self.delta
        self.Pmax = 10                                 #  dBm  功率
        self.choose = 'Urban'
        self.SNRlimit = 0
        self.K = 0.01                                  # 空气阻力系数
        self.alpha = 2
        # 环境信道参数

        self.B = 1e6                                   # 带宽 1Mhz
        self.N0 = -96                                  # dBm
        self.m = 1900                                  # in g
        self.R_th = 1.3*self.B                         #

        f = 3e9  # 载频
        c = 3e8  # 光速
        self.lossb = 20*math.log10(f*4*math.pi/c)

        # 初始参数  说明 向上加速度为正 向下 加速度为负
        self.a = np.array([0, 0, 0])                             # 加速度
        self.v = np.array([10, 10, 10])                          # 速度
        self.placex = 0                                          # 无人机位置x
        self.placey = 0                                          # 无人机位置y
        self.placez = 100                                         # 无人机位置z
        self.SPplacex = np.random.randint(-200,200,self.NSP)   # 节点位置x
        self.SPplacey = np.random.randint(-200,200,self.NSP)   # 节点位置y
        self.SPplacez = np.random.randint(0,10,self.NSP)         # 节点位置z
        self.G = np.random.randint(0, 100, self.NSP)            # 每个节点的数据量 M为单位
        self.P = 10                                              # 初始发射功率dBm
        self.P_data = 5                                          # 处理功率  单位W
        self.PLmax = self.PLoss()
        self.rate, self.SNR = self.Rate()
        self.cline = np.argmax(self.rate)
        if self.SNR[self.cline] <= self.SNRlimit:
            self.cline = -1
        # 定义状态空间
        ax = spaces.Box(low=-self.amax, high=self.amax, shape=(3,), dtype=np.float32)
        ay = spaces.Box(low=-self.amax, high=self.amax, shape=(3,), dtype=np.float32)
        az = spaces.Box(low=-self.amax, high=self.amax, shape=(3,), dtype=np.float32)
        p = spaces.Box(low=0, high=self.Pmax, shape=(1,), dtype=np.float32)
        self.action_space = np.array([ax, ay, az, p])
        v_spacex = spaces.Box(low=-self.Vmax, high=self.Vmax, shape=(1,), dtype=np.float32)
        v_spacey = spaces.Box(low=-self.Vmax, high=self.Vmax, shape=(1,), dtype=np.float32)
        v_spacez = spaces.Box(low=-self.Vmaxz, high=self.Vmaxz, shape=(1,), dtype=np.float32)
        p_spacex = spaces.Box(low=-200, high=200, shape=(1,), dtype=np.float32)
        p_spacey = spaces.Box(low=-200, high=200, shape=(1,), dtype=np.float32)
        p_spacez = spaces.Box(low=20, high=150, shape=(1,), dtype=np.float32)
        SNR = spaces.Box(low=-50, high=150, shape=(1,), dtype=np.float32)
        Gleft = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        self.observation_space = np.array([p_spacex, p_spacey, p_spacez, v_spacex, v_spacey,
                                           v_spacez, SNR, Gleft])
        self.data = [self.placex, self.placey, self.placez, self.v[0], self.v[1], self.v[2]
               , self.a[0], self.a[1], self.a[2], self.P, 0, self.cline, 0, 0]
    def reset(self):
        self.time = 0
        self.a = [0, 0, 0]  # 加速度
        self.v = [10, 10, 10]  # 速度
        self.placex = 0  # 无人机位置x
        self.placey = 0  # 无人机位置y
        self.placez = 100  # 无人机位置z
        self.SPplacex = np.random.randint(-200, 200, self.NSP)  # 节点位置x
        self.SPplacey = np.random.randint(-200, 200, self.NSP)  # 节点位置y
        self.SPplacez = np.random.randint(0, 10, self.NSP)        # 节点位置z
        self.G = np.random.randint(0, 100, self.NSP)  # 每个节点的数据量
        self.rate, self.SNR = self.Rate()
        cline = np.argmax(self.rate)
        if self.SNR[self.cline] <= self.SNRlimit:
            self.cline = -1
        S = [self.placex, self.placey, self.placez, self.v[0], self.v[1], self.v[2], self.SNR[cline], self.G[cline]]
        return S
    def step(self, a):
        acc = a[0:3]*self.amax*2-self.amax
        self.P = a[3]*self.Pmax
        P = 10**(self.P/10)/1000    #   W 为单位
        # 速度、位置变化
        # self.a = self.a + acc - self.K*self.v
        self.a = acc
        self.v += self.a*self.delta
        self.placex += self.v[0]*self.delta                              # 无人机位置x
        self.placey += self.v[1]*self.delta                                 # 无人机位置y
        self.placez += self.v[2]*self.delta                                 # 无人机位置z
        # 判断所链接的用户
        self.rate, self.SNR = self.Rate()
        self.cline = np.argmax(self.rate)
        # 判断信噪比
        PS = self.P_calfly() # 动力消耗功率
        if self.SNR[self.cline] <= self.SNRlimit:
            self.cline = -1
            reward = 0
        else:
            Gidea = self.rate[self.cline]*self.delta/1e6
            if Gidea < self.G[self.cline]:
                output = Gidea
                self.G[self.cline] -= output
                reward = output / (P + PS + self.P_data)
            else:
                output = self.G[self.cline]
                reward = output / self.delta / (P + PS + self.P_data)
        teskleft = np.sum(self.G)
        if teskleft == 0:
            done = 1
        else:
            done = 0
        self.time += 1
        self.record(a, PS, self.cline, self.rate[self.cline], reward, done)
        # if self.time == self.T:
        #     done = 1
        # trax = self.data[0, :]
        # tray = self.data[1, :]
        # traz = self.data[2, :]
        if self.placex>300 or self.placex<-300 or self.placey>300 or self.placey<-300 or self.placez>150 or self.placez<30:
            done = 1
            reward = -1000
        if self.v[0]>40 or self.v[0]<5 or self.v[1]>40 or self.v[1]<5 or self.v[2]>40 or self.v[2]<5:
            done = 1
            reward = -1000
        S_ = [self.placex, self.placey, self.placez, self.v[0], self.v[1], self.v[2]
              , self.SNR[self.cline], self.G[self.cline]]

        return S_, reward, done, {}


# 计算瞬时信噪比
    def Rate(self):
        PLmax = self.PLoss()
        rate = np.zeros(self.NSP)
        SNR = np.zeros(self.NSP)
        for i in range(self.NSP):
            SNR[i] = self.P-PLmax[i]-self.N0
            rate[i] = self.B*math.log2(1+self.IdB(SNR[i]))
        return rate, SNR

# 计算瞬时时间延迟 loss
    def PLoss(self):
        caij = np.zeros(shape=[4, 4])
        cbij = np.zeros(shape=[4, 4])
        caij[0, :] = [9.34e-1, 2.30e-1, -2.25e-3, 1.86e-5]
        caij[1, :] = [1.97e-2, 2.44e-3, 6.58e-6, 0]
        caij[2, :] = [-1.24e-4, -3.34e-6, 0, 0]
        caij[3, :] = [2.73e-7, 0, 0, 0]
        cbij[0, :] = [1.17, -7.56e-2, 1.98e-3, -1.78e-5]
        cbij[1, :] = [-5.79e-3, 1.81e-4, 1.65e-3, 0]
        cbij[2, :] = [1.73e-5, -2.02e-2, 0, 0]
        cbij[3, :] = [-2e-8, 0, 0, 0]

        subruban = [0.1, 750, 8]
        Urban = [0.3, 500, 15]
        DenseUrban = [0.5, 300, 20]
        HighUrban = [0.5, 300, 50]
        a, b, inta_los, inta_Nlos = 0, 0, 0, 0
        if self.choose == 'subruban':
                a, b = self.cal_a_b(subruban, caij, cbij)
                inta_los = 0.1
                inta_Nlos = 21
        elif self.choose == 'Urban':
                a, b = self.cal_a_b(Urban, caij, cbij)
                inta_los = 1
                inta_Nlos = 20
        elif self.choose == 'DenseUrban':
                a, b = self.cal_a_b(DenseUrban, caij, cbij)
                inta_los = 1.6
                inta_Nlos = 23
        elif self.choose == 'HighUrban':
                a, b = self.cal_a_b(HighUrban, caij, cbij)
                inta_los = 2.3
                inta_Nlos = 34

        PLmax = []
        for time in range(0, self.NSP):
                L = math.sqrt((self.placex - self.SPplacex[time]) ** 2 + (self.placey - self.SPplacey[time]) ** 2)
                H = self.placez - self.SPplacez[time]
                D = math.sqrt((self.placex - self.SPplacex[time]) ** 2 +
                              (self.placey - self.SPplacey[time]) ** 2 +
                              (self.placez - self.SPplacez[time]) ** 2)
                theta = 180*math.atan(H / L)/math.pi
                Plos = (1 / (1 + a * math.exp(-b * (theta - a))))
                PNlos = 1 - Plos
                PLmax.append(10 * math.log10(D**self.alpha) + self.lossb + Plos * inta_los + PNlos * inta_Nlos)
        return PLmax

    def cal_a_b(self, choose, caij, cbij):
            alpha = choose[0]
            belta = choose[1]
            gama = choose[2]
            a = 0
            b = 0
            for j in range(0, 4):
                for i in range(3 - j):
                    a += ((alpha * belta) ** i) * (gama ** j) * caij[i, j]
                    b += ((alpha * belta) ** i) * (gama ** j) * cbij[i, j]
            return a, b


# 计算顺时功率
    def P_calfly(self):
        C1 = 9.26e-4
        C2 = 2250
        g  = 9.8
        normV = np.linalg.norm(self.v)
        norma = np.linalg.norm(self.a)
        cos = np.sum(self.v*self.a)
        Ps = C1*normV**3+C2/normV*(1+(norma**2-cos**2/normV**2)/(g**2))
        return Ps


# 计算dB
    def dB(self,a):
        b = 10*math.log10(a/10)
        return b

    def IdB(self, a):
        b = math.pow(10,a/10)
        return b
# 画图三维
    def drawplot(self):
        fig = plt.figure(1)
        ax = Axes3D(fig)
        ax.scatter(self.placex, self.placey, self.placez)
        ax.scatter(self.SPplacex, self.SPplacey, self.SPplacez)
        ax.text(self.placex, self.placey, self.placez,
                'loc='+str([self.placex, self.placey, self.placez])+'\n'
                +'V='+str(self.v)+'\n'+'P='+str(self.P))
        if self.cline != -1:
            ax.plot([self.placex, self.SPplacex[self.cline]], [self.placey, self.SPplacey[self.cline]],
                    [self.placez, self.SPplacez[self.cline]], '--')
            ax.text((self.placex+self.SPplacex[self.cline])/2, (self.placey+self.SPplacey[self.cline])/2,
                    (self.placez+self.SPplacez[self.cline])/2, str(self.rate[self.cline]))
            ax.text(self.SPplacex[self.cline], self.SPplacex[self.cline], self.SPplacex[self.cline],
                    'loc='+str(self.SPplacex[self.cline])+str(self.SPplacex[self.cline])+str(self.SPplacex[self.cline])+'\n'
                     +'G='+str(self.G[self.cline])+'\n')
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        ax.set_zlim(0, 150)
        plt.show()
    def trajectory(self):
        fig = plt.figure(1)
        ax = Axes3D(fig)
        trax = self.data[:, 0]
        tray = self.data[:, 1]
        traz = self.data[:, 2]
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        ax.set_zlim(0, 150)
        ax.plot3D(trax,tray,traz,'r')
        ax.scatter3D(self.SPplacex, self.SPplacey, self.SPplacez,'g')
        for cline in range(self.NUAV):
            ax.text(self.SPplacex[cline], self.SPplacey[cline], self.SPplacez[cline],
                'loc=' + str(self.SPplacex[cline]) + str(self.SPplacey[cline]) + str(self.SPplacez[cline]) + '\n'
                + 'G=' + str(self.G[cline]) )
        plt.show()

    def render(self, mode='human'):
        return {}

    def record(self, a, ps, cline, rate, reward, done ):
        basic_data = [self.SPplacex,self.SPplacex,self.SPplacex,self.G]
        data = [self.placex, self.placey, self.placez, self.v[0], self.v[1], self.v[2]
               , a[0], a[1], a[2], self.P, ps, cline, rate/1e6, reward]
        self.data = np.vstack((self.data, data))
        if done == 1:
            np.savetxt("tarjectory.csv", self.data, delimiter=',')
            np.savetxt("basic.csv", basic_data, delimiter=',')
if __name__ == '__main__':
    env = Downlink()
    env.reset()
    loss = env.PLoss()
    env.drawplot()


