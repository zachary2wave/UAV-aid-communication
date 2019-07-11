
import numpy as np
import matplotlib.pyplot as plt

'''
该程序参考

2000 年 JSCA 论文
Performance analysis of IEEE 802.11 distributed coordination function
总体环境
# avr_ap = 1 为均匀分布  其他为 随机分布
所有的 AP 和UE都利用同样的信道 
这里需要改变载波感知范围、 功率

重点 这里只关心 下行信道 ，也就是UE不去参与竞争

'''


class Scenario:

    def __init__(self, NAP, NUE, freq, avr_ap):
        'the envitonment'
        self.MAZE_H = 100
        self.MAZE_W = 100
        self.NAP = NAP
        self.NUE = NUE
        self.nsf2 = 1.67      # 2.4G 同层的穿透
        self.nsf5 = 1.87       # 5G   同城的穿透
        self.alpha2 = 0.01     # 2.4G 信道衰减指数
        self.alpha5 = 0.4     # 5G   信道衰减指数
        self.FAF2 = 13        # 2.4G 穿透
        self.FAF5 = 24        # 5G   穿透
        self.freq = freq      # 频率选择
        self.normalstd2 = 3   # 2.4G  标准差
        self.normalstd5 = 4   # 5G    标准差
        self.avr_ap = avr_ap  # 1 为均匀分布  其他为 随机分布

        " the speed"
        self.requireSNR = [6, 7.8, 9, 10.8, 17, 18.8, 24, 26]
        rate = [6, 9, 12, 18, 24, 36, 48, 54]


        self.n = 8             # 噪声等级 （dB）
        self.Cue = 0
        self.tao = 0.2
        self.packet_payload = 8184
        MACheader = 272
        PHYheader = 128
        ACK = 112 + PHYheader
        RTS = 160 + PHYheader
        CTS = 112 + PHYheader
        Bitrate = 1e6

        TACK = ACK / Bitrate
        TRTS = RTS / Bitrate
        TCTS = CTS / Bitrate

        PropagationDelay = 1e-6
        SlotTime = 50e-6
        SIFS = 28e-6
        DIFS = 128e-6


        self.Tsucc_p = TRTS + SIFS + TCTS + SIFS + (MACheader+PHYheader)/Bitrate + SIFS + TACK + DIFS
        self.Tidle = SlotTime
        self.Tcoll = RTS/Bitrate+DIFS
    '''
    辅助函数
    '''
    def dB(self, a):
        return 10*np.log10(a)
    def idB(self,a):
        return 10**(a/10)

    '''
    环境配置部分
    '''
    def Enviroment_AP(self):
        if self.avr_ap == 1:
            APavenum = int(np.sqrt(self.NAP))
            avrlengH = self.MAZE_H / (APavenum + 1)
            avrlengW = self.MAZE_W / (APavenum + 1)
            APlocX = np.arange(0, self.MAZE_H, avrlengH)
            APlocY = np.arange(0, self.MAZE_W, avrlengW)
            APX=APlocX[1:]
            APY=APlocY[1:]
            outAPX = np.repeat(APX, APavenum)
            outAPY = np.zeros(self.NAP)
            # temp = np.repeat(APY, APavenum)
            # int()
            for loop1 in range(0, APavenum):
                temp = APY[np.arange(0-loop1, APavenum-loop1)]
                part = np.arange(0 + loop1 * APavenum, APavenum * (1 + loop1))
                for loop2 in range(0, APavenum):
                    outAPY[part[loop2]] = temp[loop2]
        else:
            outAPX = np.random.randint(1, self.MAZE_H, self.NAP)
            outAPY = np.random.randint(1, self.MAZE_W, self.NAP)

        return outAPX, outAPY

    def Enviroment_UE(self):
        UEX = np.random.randint(1, self.MAZE_H, self.NUE)
        UEY = np.random.randint(1, self.MAZE_W, self.NUE)
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

    def loss_metrix(self):
        APX, APY = self.Enviroment_AP()
        UEX, UEY = self.Enviroment_UE()
        # AP 2 UE
        LossAP2UE=np.zeros([self.NAP,self.NUE])
        for loop1 in range(self.NAP):
            for loop2 in range(self.NUE):
                LossAP2UE[loop1, loop2] = self.loss(UEX[loop2], UEY[loop2], APX[loop1], APY[loop2])
        # UE 2 UE
        LossUE2UE = np.zeros([self.NUE, self.NUE])
        for loop1 in range(self.NUE):
            for loop2 in range(self.NUE):
                LossUE2UE[loop1, loop2] = self.loss(UEX[loop2], UEY[loop2], UEX[loop1], UEY[loop2])
        # AP 2 AP
        LossAP2AP=np.zeros([self.NAP,self.NUE])
        for loop1 in range(self.NAP):
            for loop2 in range(self.NAP):
                LossAP2AP[loop1, loop2] = self.loss(APX[loop2], APY[loop2], APX[loop1], APY[loop2])
        return LossAP2UE, LossAP2AP, LossUE2UE
    '''
    根据 输出的P 计算UE所连接的AP
    '''

    def connetion(self, P):
        LossAP2UE, LossAP2AP, LossUE2UE = self.loss_metrix()
        connetion = np.zeros([self.NUE])
        SNR = np.zeros([self.NUE])
        rate = np.zeros([self.NUE])
        for ue in range(0, self.NUE):
            record = np.array([])
            for fap in range(0, self.NAP):
                power = P[fap] - LossAP2UE[fap, ue] - self.n
                if power > self.Cue:
                    record = np.append(record, power)
                else:
                    record = np.append(record, -1e6)
            connetion[ue] = np.max(record)
            SNR[ue] = np.max(record)
            findnear = np.argmin(abs(SNR - self.requireSNR))
            if SNR[ue] >= self.requireSNR[findnear]:
                rate[ue] = self.rate[findnear]
            elif SNR[ue] < np.min(self.requireSNR):
                rate[ue] = 0
            elif SNR[ue] < self.requireSNR[findnear]:
                rate[ue] = self.rate[findnear - 1]
        return connetion, SNR, rate

    '''
    根据 P和C 计算 吞吐量
    '''
    def calculation_NP(self, P, C):
        '''
        只考虑下行信道，所以不考虑UE对AP的影响 但是这个地方要计算AP对UE的影响
        '''
        LossAP2UE, LossAP2AP, LossUE2UE = self.loss_metrix()
        # the first ord
        # calculation for AP
        totalAP = np.zeros([self.NAP, self.NAP])
        for ap in range(0, self.NAP):
            for fap in range(0, self.NAP):
                power = self.idB(P[fap]-LossAP2AP[fap, ap])+self.idB(self.n)
                if self.dB(power) > C[ap]:
                    totalAP[ap, fap] = 1
            # 不考虑ue 对于 AP的影响
            # for fue in range(self.NUE, self.NAP + self.NUE):
            #     power = self.Pue - LossAP2UE[ap, fue] - self.n
            #     if power > C[ap]:
            #         totalAP[ap, fap] = 1

        # calculation for UE
        totalUE = np.zeros([self.NUE, self.NAP])
        for ue in range(0, self.NUE):
            for fap in range(0, self.NAP):  # type: int
                power = self.idB(P[fap] - LossAP2UE[fap, ue]) + self.idB(self.n)
                if self.dB(power) > self.Cue:
                    totalUE[ue, fap] = 1
            # 不考虑UE影响
            # for fue in range(self.NUE, self.NAP + self.NUE):
            #     power = self.idB(self.Pue - LossUE2UE[ue, fue]) + self.idB(self.n)
            #     if self.dB(power) > self.Cue:
            #         totalUE[ap, fue] = 1
        # non interference set
        noAP = []
        oneAP = np.zeros([self.NAP])
        for ap in range(0, self.NAP):
            num = np.where(totalAP[ap, :] != 1)
            noAP.append(num)
            oneAP[ap] = self.NAP - num.shape
        noUE = []
        oneUE = np.zeros([self.NAP])
        for ue in range(0, self.NUE):
            num = np.where(totalUE[ue, :] != 1)
            noUE.append(num)
            oneUE[ap] = self.NAP - num.shape


        # the second order
        '''
        node1 node2 都不是AP的一阶节点 且 node1 和 node2 互相都不是
        '''
        twoAP = np.zeros([self.NAP])
        secordAP = []
        for ap in range(0, self.NAP):
            tempAP = []
            set = set(noAP[ap])
            '选择node1'
            for node1 in set:
                for node2 in set:
                    set1 = set(noAP[node1])
                    set2 = set(noAP[node2])
                    if node1 in set2:
                        break
                    if node2 in set1:
                        break
                    if node1 == node2:
                        break
                    power = self.idB(P[node1] - LossAP2UE[node1, ap]) \
                            + self.idB(P[node2] - LossAP2UE[node1, ap]) \
                            + self.idB(self.n)
                    if self.dB(power) > self.C[ap]:
                        tempAP.append([node1, node2])
            secordAP.append(tempAP)
            twoAP[ap] = len(tempAP)

        twoUE = np.zeros([self.NUE])
        secordUE = []
        for ue in range(0, self.NUE):
            tempUE = []
            set = set(noUE[ue])
            '选择node1'
            for node1 in set:
                '选择node2'
                for node2 in set:
                    set1 = set(noAP[node1])
                    set2 = set(noAP[node2])
                    if node1 in set2:
                        break
                    if node2 in set1:
                        break
                    if node1 == node2:
                        break
                    power = self.idB(P[node1] - LossAP2UE[node1, ue]) \
                            + self.idB(P[node2] - LossAP2UE[node2, ue]) \
                            + self.idB(self.n)
                    if self.dB(power) > self.C[ap]:
                        tempUE.append([node1, node2])
            secordUE.append(tempUE)
            twoUE[ue] = len(tempUE)

        NumAP = twoAP + oneAP
        NumUE = twoUE + oneUE
        return NumAP, NumUE



    def through_out(self, P, C):
        connetion, SNR, rate = self.connetion(P)
        NumAP, NumUE = self.calculation_NP(P, C)
        thought_out = 0
        for i in range(self.NUE):
            con = connetion[i]
            nt = NumAP[con]
            nr = NumUE[i]
            n = nt + nr
            Pt = 1-(1-self.tao)**n
            Ps = n*self.tao*(1+self.tao)**(n-1)/Pt
            Pidle = 1-Pt
            Psucc = Pt*Ps
            Pcoll = Pt*(1-Ps)

            Tsucc = self.packet_payload/rate/1e6
            thought_out += Psucc*self.packet_payload/\
                          (Psucc*Tsucc+Pidle*self.Tidle+Pcoll*self.Tcoll)
        return thought_out

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
        for loop in range(0, self.NAP):
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
        # for UEloop in range(0, self.NUE):
        #     plt.plot([placeUE[0, UEloop], placeAP[0, int(contact[UEloop])]],
        #              [placeUE[1, UEloop], placeAP[1, int(contact[UEloop])]]
        #              , color=color[int(contact[UEloop])])
        #     plt.text(placeUE[0, UEloop], placeUE[1, UEloop], str(UEloop), color='k')

        plt.legend([pue, pap], ['UE', 'AP'], loc='upper right')
        plt.show()
        plt.figure(3)
        plt.plot(range(0, 1000), -Loss)
        plt.show()