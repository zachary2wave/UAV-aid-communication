import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D



placex = 0  # 无人机位置x
placey = 0  # 无人机位置y

a = 9.5327
b = 0.4095


SPplacex = np.arange(-1000, 1000, step=6)  # 节点位置x
SPplacey = np.arange(-1000, 1000, step=6)  # 节点位置y
SPplacez = 0  # 节点位置z
print(np.shape(SPplacex))
ppl = int(np.shape(SPplacex)[0])
inta_los = 2
inta_Nlos = 20
lossb = 41.98
PLmax = np.zeros([ppl, ppl])
rate = np.zeros([ppl, ppl])
record=[]
for loop in range(0,100):
    placez = 50+loop  # 无人机位置z
    print(placez)
    for time1 in range(ppl):
        for time2 in range(ppl):
            L = math.sqrt((placex - SPplacex[time1])** 2 + (placey - SPplacey[time2]) ** 2)
            H = placez - SPplacez
            D = math.sqrt((placex - SPplacex[time1]) ** 2 +
                          (placey - SPplacey[time2]) ** 2 +
                          (placez - SPplacez) ** 2)
            theta = 180*math.asin(H / D)/math.pi
            Plos = (1 / (1 + a * math.exp(-b * (theta - a))))
            PNlos = 1 - Plos
            # if L<100:
            #     print(L,PNlos)
            PLmax[time1, time2] = (10 * math.log10(D**2) + lossb + Plos * inta_los + PNlos * inta_Nlos)
            rate [time1, time2] = math.log2(1+10**((10-PLmax[time1, time2]+96)/10))
    record.append(np.max(rate))

fig = plt.figure(2)
plt.plot(np.arange(50,150),record)
plt.show()
# X, Y = np.meshgrid(SPplacex, SPplacex)
#
# fig = plt.figure(1)
# ax = Axes3D(fig)
# ax.plot_surface(X, Y, rate)
# plt.show()





