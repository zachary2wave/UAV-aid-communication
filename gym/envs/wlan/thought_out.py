"""
该程序参考
2006年 TWC 论文
Joint Access Point Placement and Channel Assignment for 802.11 Wireless LANs
该论文参考
2000 年 JSCA 论文
Performance analysis of IEEE 802.11 distributed coordination function

在本程序中固定 AP位置 和 UE位置
UE链接最强的RSSI的AP

当 UE 小于最小链接强度的时候  不传输数据  同时不记录 其链接 AP的n
因为 n会使得传输的概率变小

"""

import numpy as np
from gym.envs.wlan import env_simulated as env
import matplotlib.pyplot as plt
import csv


class ThoughtOutCal:
    def __init__(self, channel, power, Num_AP, Num_UE):
        self.APchannel = channel                                    # 信道为0、1、2
        self.APpower = power                                        # dB 为单位
        self.Num_AP = Num_AP
        self.Num_UE = Num_UE
        require = np.array([-5, -10, -13, -16, -19, -22, -25, -27, -30, -32])
        self.requireSNR = -require                                  # 求解 所需要的信噪比的底线
        #############################################################################
        # 传输速率####
        self.BB = 20  # Mhz
        Rate = [1/2, 1/2, 3/4, 1/2, 3/4, 2/3, 3/4, 5/6, 3/4, 5/6]   # 码率
        Nsub = 48  # 固定带宽为20MHz 也就是具有48个子载波
        NBPSC = np.array([1, 2, 2, 4, 4, 6, 6, 6, 8, 8])  # 每个符号所承载的子载波数目
        NSS = 8  # 空间流
        Tsym = 3.6                                                    # 码元持续时间
        self.speed = Nsub*NBPSC*Rate*NSS/Tsym
        self.noise = 8
        self.t = 0.5                  # 终端退回为0并发送的概率
        # 产生数据的长度
        # 数据帧最大长度为7955bit 所以假设 此处数据长度为6000bit
        self.bits = 6000
        #############################################################################
        # 数据时间
        t = 9
        sifs = 16           # 短帧间时隙
        difs = sifs + 2*t   # 长帧间时隙
        rts = 44                      # rts时间
        cts = 37                      # cts时间
        ack = 37                      # ACK时间长度
        self.Tcoll = rts+difs         # 碰撞时间
        self.Tslot = t                # 空闲时隙
        self.Ti = rts + cts + difs + ack + 3*sifs

    def SNR(self, loss, num):
        # loss 是该UE对所有AP的损失
        reciAP = np.array(self.APpower)-np.array(loss)
        recist = np.power(10, reciAP/10)                                 # dB 转换成正常值
        contact = np.argmax(recist)
        send = recist[contact]
        jam = np.delete(recist, contact)
        channel = np.array(self.APchannel)
        channel = np.delete(channel, contact)
        nowcon = self.APchannel[contact]
        indexsame = np.argwhere(channel == nowcon)

        # same = indexsame.size
        total_jam = 0
        noise = 1
        # noise = abs(np.random.normal(loc=0, scale=self.noise))
        for APloop in indexsame:
            total_jam += jam[APloop]
        SINR = send/(total_jam+noise)
        SINR = 10 * np.log10(SINR)
        # with open('SNR结果.csv', 'a', newline='') as f:
        #     uw = csv.writer(f, dialect='excel')
        #     uw.writerow(['UE标号', num,'连接的标号', contact, '干扰的标号', indexsame, '噪声', noise])
        #     uw.writerow(['接收到AP的强度', 'UE标号', num, 'SNR', SINR, '总干扰', total_jam])
        #     uw.writerow(recist)

        return SINR, contact
    # ########################################
    # 下面函数主要实现的香浓定理的 求解 理论的速率
    # ########################################

    # def th_speed(self, connetion, Loss):
    #     SINR = []
    #     sitth = []
    #     for UEloop in range(0, self.Num_UE):
    #         tempco=connetion[0, UEloop]
    #         SINR[UEloop] = self.SNR(Loss[UEloop, :], tempco)
    #         sitth[UEloop] = self.BB*np.log10(1+SINR[UEloop])        # 在此场景下的该UE理论最高吞吐量
    #     return sitth
    # ########################################
    # 在该时间段内的吞吐量
    # ########################################

    def subspeed(self, Loss):
        # 这里输入的loss 是单个UE对所有AP的loss
        Speed = np.zeros(self.Num_UE)
        connection = np.zeros(self.Num_UE)
        SNR = np.zeros(self.Num_UE)
        for UEloop in range(0, self.Num_UE):
            RSSI, contact = self.SNR(Loss[UEloop, :], UEloop)
            SNR[UEloop] = RSSI
            findnear = np.argmin(abs(RSSI-self.requireSNR))
            if RSSI >= self.requireSNR[findnear]:
                Speed[UEloop] = self.speed[findnear]
                connection[UEloop] = contact
            elif RSSI < np.min(self.requireSNR):
                Speed[UEloop] = 0
                connection[UEloop] = 1000
            elif RSSI < self.requireSNR[findnear]:
                Speed[UEloop] = self.speed[findnear - 1]
                connection[UEloop] = contact
        return SNR, Speed, connection

    def thomain(self, Speed, connetion):
        nconnetion = np.zeros([self.Num_AP])
        Psucc = np.zeros([self.Num_AP])
        Pidle = np.zeros([self.Num_AP])
        Pcoll = np.zeros([self.Num_AP])
        Ti = np.zeros([self.Num_UE])
        thought_out = np.zeros([self.Num_UE])
        APxconnetion=[]
        for APloop in range(0, self.Num_AP):
            nconnetion = np.sum(connetion == APloop)
            if nconnetion == 0:
                Ptr = 1
                Ps = 0
            else:
                Ptr = 1 - (1 - self.t) ** nconnetion
                Ps = nconnetion * self.t * ((1 - self.t) ** (nconnetion - 1)) / Ptr
            Psucc[APloop] = Ptr * Ps
            Pidle[APloop] = 1 - Ptr
            Pcoll[APloop] = Ptr * (1 - Ps)
            temploc = np.argwhere(connetion == APloop)
            APxconnetion.append(temploc)
        for UEloop in range(0, self.Num_UE):
            if Speed[UEloop] != 0:
                Ti[UEloop] = self.Ti + self.bits/Speed[UEloop]
        for UEloop in range(0, self.Num_UE):
            if Speed[UEloop] != 0:
                conAP = int(connetion[UEloop])
                upP = Psucc[conAP]/np.sum(connetion == conAP)
                totalT = 0
                for tin in APxconnetion[conAP]:
                    totalT += Ti[int(tin)]
                downP = upP*(totalT) + Pidle[conAP]*self.Tslot + Pcoll[conAP]*self.Tcoll
                thought_out[UEloop] = upP/downP*self.bits
            else:
                thought_out[UEloop] = 0
        P = np.zeros([3,self.Num_AP])
        P[0, :] = Psucc
        P[1, :] = Pidle
        P[2, :] = Pcoll
        return thought_out, P


def thAP(thought_out_ue, connettion, Num_AP):
    thought_out_AP = np.zeros([Num_AP])
    for kki in range(0, Num_AP):
        tempN = np.argwhere(connettion == kki)
        for kkj in tempN:
            thought_out_AP[kki] += thought_out_ue[kkj]
    return thought_out_AP


def showplot(placeAP, placeUE, power, channel, connection):
    Loss = np.zeros(1000)
    for distance in range(1, 1000):
        Loss[distance] = 10 * 1.67 * np.log10(distance/2)+0.01 * distance
    APpower = np.transpose([power])-Loss
    plt.figure(1)
    plt.scatter(placeUE[0, :], placeUE[1, :], marker=',')
    plt.scatter(placeAP[0, :], placeAP[1, :], marker='v')
    color = ['r', 'k', 'c', 'm', 'g', 'y', 'b', '#FF99FF', '#9999FF']
    for loop in range(0, len(channel)):
        # ###########  作用范围  ##################
        r1 = np.argmin(abs(APpower[loop, :] - 5))
        r2 = np.argmin(abs(APpower[loop, :] - 30))
        theta = np.arange(0, 2 * np.pi, 0.01)
        x1 = placeAP[0, loop] + r1 * np.cos(theta)
        y1 = placeAP[1, loop] + r1 * np.sin(theta)
        plt.plot(x1, y1, color=color[loop])
        x2 = placeAP[0, loop] + r2 * np.cos(theta)
        y2 = placeAP[1, loop] + r2 * np.sin(theta)
        plt.plot(x2, y2, color=color[loop])
        # ###########  标号      ##################
        plt.text(placeAP[0, loop], placeAP[1, loop], str(loop), color='r')
        # plt.text(placeAP[0, loop]+5, placeAP[1, loop]+5, str(channel[loop]), color='k')
        # ###########  关联  ######################
    for UEloop in range(0,len(connection)):
        plt.text(placeUE[0, UEloop], placeUE[1, UEloop], str(UEloop), color='k')
    unused = np.argwhere(connection == 1000)
    unused = np.squeeze(unused, axis=1)
    placeUE=np.delete(placeUE, unused, axis=1)
    connection=np.delete(connection, unused)
    for UEloop in range(0, len(connection)):
        plt.plot([placeUE[0, UEloop], placeAP[0, int(connection[UEloop])]],
                    [placeUE[1, UEloop], placeAP[1, int(connection[UEloop])]]
                    , color=color[int(connection[UEloop])])



if __name__ == '__main__':
    Num_AP = 9
    Num_UE = 100
    channel = [1, 2, 1, 2, 2, 3, 2, 1, 1]
    # power = 9*[40]
    # env part
    env = env.Scenario(Num_AP=Num_AP, Num_UE=Num_UE, freq=2, avr_ap=1)
    contact, placeAP, placeUE, Loss = env.sendout()

    with open('thought结果.csv', 'w') as f:
        csvwrite = csv.writer(f, dialect='excel')
        # csvwrite.writerow(['time=2019-1-3'])
    # with open('SNR结果.csv', 'w') as f:
    #     uw = csv.writer(f, dialect='excel')
    #     uw.writerow(['time=2019-1-3'])

    # power = np.random.randint(0, 50, Num_AP)
    # tho = ThoughtOutCal(channel, power, Num_AP, Num_UE)
    # RSSI, Speed, connection = tho.subspeed(Loss)
    # thought_out_simulate, P = tho.thomain(Speed, connection)
    # ending = np.sum(thought_out_simulate)
    # thought_out_AP = thAP(thought_out_simulate, connection, Num_AP)
    #
    # with open('thought结果.csv', 'a', newline='') as f:
    #     uw = csv.writer(f, dialect='excel')
    #     uw.writerow(['thought=', str(ending)])
    #     uw.writerow(['power='])
    #     uw.writerow(power)
    #     uw.writerow(np.arange(0, 99, 1))
    #     uw.writerow(['RSSI'])
    #     uw.writerow(RSSI)
    #     uw.writerow(['Speed'])
    #     uw.writerow(Speed)
    #     uw.writerow(['Connection'])
    #     uw.writerow(connection)
    #     print('success writer')
    # showplot(placeAP, placeUE, power, channel, connection)

    record=np.zeros([50])
    for time in range(0, 10):
        ending = 0
        for ti in range(0,100):
            power = Num_AP * [10+2*time]
            tho = ThoughtOutCal(channel, power, Num_AP, Num_UE)
            RSSI, Speed, connection = tho.subspeed(Loss)
            thought_out_simulate, P = tho.thomain(Speed, connection)
            ending += np.sum(thought_out_simulate)
            thought_out_AP = thAP(thought_out_simulate, connection, Num_AP)
        showplot(placeAP, placeUE, power, channel, connection)
        record[time] = ending/100
        print(time)
        with open('thought结果.csv', 'a', newline='') as f:
            uw = csv.writer(f, dialect='excel')
            uw.writerow(thought_out_AP)
    plt.show()
    print('ending')











    #
    #
    # ending = np.zeros([30])
    # SNRrecord = np.zeros([30, Num_UE])
    #
    #
    # for kk in range(0, 30):
    #     # thought_out
    #     print(kk)
    #     for time in range(0,1000):
    #         power = 9*[kk+20]
    #         tho = ThoughtOutCal(channel, power, Num_AP, Num_UE)
    #         RSSI, Speed, connection = tho.subspeed(Loss)
    #         thought_out_simulate = tho.thomain(Speed, connection)
    #         ending[kk] += np.sum(thought_out_simulate)
    #         SNRrecord[kk,:] += RSSI
    # plt.figure(20)
    # plt.plot(np.arange(20, 50), SNRrecord/1000)
    # plt.figure(10)
    # plt.plot(np.arange(20, 50), ending/1000)
    # plt.show()
