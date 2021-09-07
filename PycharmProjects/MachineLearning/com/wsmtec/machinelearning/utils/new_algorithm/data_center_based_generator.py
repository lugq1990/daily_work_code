# -*- coding:utf-8 -*-
import numpy as np

class my_over_sampling:
    def __init__(self,samples,num=10,pos_samples=None):
        self.samples = samples
        self.num = num
        self.n_samples,self.n_cols = samples.shape
        self.pos_samples = pos_samples

    def over_sampling(self):
        self.index = 0
        self.result = np.zeros((self.num,self.n_cols))
        self.centers = np.mean(self.samples,axis=0)
        #print("center ",self.centers)
        dis = self.__computeDis()[0]
        for i in range(self.num):
            rotio = np.random.random()
            obj_choose = np.random.randint(self.n_samples)
            #add the guassian mixer
            mu, sigma = 0,0.1
            noise_rotio = np.random.normal(mu,sigma,size=(1,self.n_cols))[0]
            #print("!!",noise_rotio[0])
            self.result[i] = self.samples[obj_choose] + rotio*dis + noise_rotio
        return self.result

    #compute the center for the data
    def __computeDis(self):
        num_sam = self.samples.shape[0]
        #add the pos data center and distance
        if(self.pos_samples is not None):
            print("now in the pos data !")
            num_sam = self.pos_samples.shape[0]
            self.n_cols = self.pos_samples.shape[1]
            self.pos_center = np.mean(self.pos_samples,axis=0)
            dis_pos = np.zeros((1,self.n_cols))
            for j in range(num_sam):
                dif_pos = self.pos_center - self.pos_samples[j]
                dis_pos += dif_pos
            avg_pos_dis = dis_pos/num_sam
            return avg_pos_dis
        else:
        #use the original data center and distance
            dis = np.zeros((1,self.n_cols))
            for i in range(num_sam):
                dif = self.centers - self.samples[i]
                dis += dif
            avg_dis = dis/num_sam
            #print(avg_dis[0])
            return avg_dis

from sklearn.datasets import load_iris
iris = load_iris()
x,y = iris.data,iris.target
tmp = my_over_sampling(x[:50,:],num=3,pos_samples=x[50:101,:])
re = tmp.over_sampling()
print(re)
print(x[:10])