"""
该程序仿真在单层 没有房间穿透 没有楼层穿透影响下
AP 均匀分布的时候 的情况
说明的是 freq = 2  为2.4G  freq = 5 为 5G条件下

"""
import numpy as np
import matplotlib.pyplot as plt


class Scenario:

    def __init__(self, Num_AP, Num_UE, freq, avr_ap):
        self.MAZE_H = 100
        self.MAZE_W = 100
        self.Num_AP = Num_AP
        self.Num_UE = Num_UE
        self.nsf2 = 1.67      # 2.4G 同层的穿透
        self.nsf5 = 1.87       # 5G   同城的穿透
        self.alpha2 = 0.01     # 2.4G 信道衰减指数
        self.alpha5 = 0.4     # 5G   信道衰减指数
        self.FAF2 = 13        # 2.4G 穿透
        self.FAF5 = 24        # 5G   穿透
        self.freq = freq      # 频率选择
        self.normalstd2 = 3   # 2.4G  标准差
        self.normalstd5 = 4   # 5G    标准差
        self.avr_ap = avr_ap

    def Enviroment_AP(self):
        if self.avr_ap == 1:
            APavenum = int(np.sqrt(self.Num_AP))
            avrlengH = self.MAZE_H / (APavenum + 1)
            avrlengW = self.MAZE_W / (APavenum + 1)
            APlocX = np.arange(0, self.MAZE_H, avrlengH)
            APlocY = np.arange(0, self.MAZE_W, avrlengW)
            APX=APlocX[1:]
            APY=APlocY[1:]
            outAPX = np.repeat(APX, APavenum)
            outAPY = np.zeros(self.Num_AP)
            # temp = np.repeat(APY, APavenum)
            # int()
            for loop1 in range(0, APavenum):
                temp = APY[np.arange(0-loop1, APavenum-loop1)]
                part = np.arange(0 + loop1 * APavenum, APavenum * (1 + loop1))
                for loop2 in range(0, APavenum):
                    outAPY[part[loop2]] = temp[loop2]
        else:
            outAPX = np.random.randint(1, self.MAZE_H, self.Num_AP)
            outAPY = np.random.randint(1, self.MAZE_W, self.Num_AP)

        return outAPX, outAPY

    def Enviroment_UE(self):
        UEX = np.random.randint(1, self.MAZE_H, self.Num_UE)
        UEY = np.random.randint(1, self.MAZE_W, self.Num_UE)
        return UEX, UEY

    def loss(self, UEX, UEY, APX, APY):
        distance = np.sqrt(pow(APX-UEX, 2)+pow(APY-UEY, 2))
        if self.freq == 2:
            shadefall = np.random.normal(0, self.normalstd2)
            Loss = 10*self.nsf2*np.log10(distance/2)+self.alpha2*distance+shadefall
        else:
            shadefall = np.random.normal(0, self.normalstd5)
            Loss = 10*self.nsf5*np.log10(distance/2)+self.alpha5*distance+shadefall
        return Loss

    def sendout(self):
        [APX, APY]=self.Enviroment_AP()
        [UEX, UEY]=self.Enviroment_UE()
        Loss = np.zeros([self.Num_UE, self.Num_AP])
        contactloss = np.zeros([self.Num_UE])
        contactAPnum= np.zeros([self.Num_UE])
        for UEloop in range(0, self.Num_UE):
            for APloop in range(0, self.Num_AP):
                Loss[UEloop, APloop] = self.loss(UEX[UEloop], UEY[UEloop], APX[APloop], APY[APloop])
            contactloss[UEloop] = min(Loss[UEloop, :])
            contactAPnum[UEloop] = list(Loss[UEloop, :]).index(contactloss[UEloop])
        contact = np.zeros([2, self.Num_UE]); placeAP = np.zeros([2, self.Num_AP])
        placeUE = np.zeros([2, self.Num_UE])
        contact[0, :] = contactloss
        contact[1, :] = contactAPnum
        placeAP[0, :] = APX
        placeAP[1, :] = APY
        placeUE[0, :] = UEX
        placeUE[1, :] = UEY
        return contact, placeAP, placeUE, Loss

    def connection(self, power, Loss):
        contact = np.zeros(self.Num_UE)
        loss_contact = []
        for UEloop in range(0, self.Num_UE):
            temploss = Loss[UEloop, :]
            tRSSI = power-temploss
            contact[UEloop] = np.argmax(tRSSI)
        return contact, len(loss_contact)
    ########################################################
    #  画图部分
    ########################################################



    def showplot(self, placeAP, placeUE, contact, channel):

        Loss = np.zeros(1000)
        for distance in range(1, 1000):
            shadefall = np.random.normal(0, self.normalstd5)
            Loss[distance] = 10*self.nsf2*np.log10(distance/2)+self.alpha2*distance
        r1 = np.argmin(abs(Loss - 5))
        r2 = np.argmin(abs(Loss - 30))
        plt.figure(1)
        pue = plt.scatter(placeUE[0, :], placeUE[1, :], marker=',')
        pap = plt.scatter(placeAP[0, :], placeAP[1, :], marker='v')
        for loop in range(0, self.Num_AP):
            # plt.text(placeAP[0, loop], placeAP[1, loop], str(loop), color='r')
            plt.text(placeAP[0, loop]+5, placeAP[1, loop]+5, str(channel[loop]), color='k')

            theta = np.arange(0, 2 * np.pi, 0.01)
            x1 = placeAP[0, loop] + r1 * np.cos(theta)
            y1 = placeAP[1, loop] + r1 * np.sin(theta)
            plt.plot(x1, y1)
            theta = np.arange(0, 2 * np.pi, 0.01)
            x2 = placeAP[0, loop] + r2 * np.cos(theta)
            y2 = placeAP[1, loop] + r2 * np.sin(theta)
            plt.plot(x2, y2)

            # plt.Circle(xy=(placeAP[0, loop], placeAP[1, loop]), radius=r1, alpha=0.6)
            # plt.Circle(xy=(placeAP[0, loop], placeAP[1, loop]), radius=r2, alpha=0.7)
            # 黑色为信道 红色为标号
        # color=['r', 'k', 'c', 'm', 'g', 'y', 'b', '#FF99FF', '#9999FF']
        # for UEloop in range(0, self.Num_UE):
        #     plt.plot([placeUE[0, UEloop], placeAP[0, int(contact[UEloop])]],
        #              [placeUE[1, UEloop], placeAP[1, int(contact[UEloop])]]
        #              , color=color[int(contact[UEloop])])
        #     plt.text(placeUE[0, UEloop], placeUE[1, UEloop], str(UEloop), color='k')

        plt.legend([pue, pap], ['UE', 'AP'], loc='upper right')
        plt.show()
        plt.figure(3)
        plt.plot(range(0, 1000), -Loss)
        plt.show()


if __name__ == '__main__':
    Num_AP = 1
    Num_UE = 50
    channel = [1, 2, 1, 2, 2, 3, 2, 1, 1]
    loss_cal = Scenario(Num_AP=Num_AP, Num_UE=Num_UE, freq=2, avr_ap=1)
    contact, placeAP, placeUE, Loss = loss_cal.sendout()
    loss_cal.showplot(placeAP, placeUE, contact[1, :], channel)









