# -*- coding:utf-8 -*-
import numpy as np
from sklearn.neighbors import NearestNeighbors

class SMOTE:
    #samples final colums is class only 1
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs = samples.shape
        self.N = N
        self.k = k
        self.samples = samples

    #make the oversmapling
    def over_sampling(self):
        if self.N < 100:
            old_n_samples = self.n_samples
            print("old_n_samples",old_n_samples)
            self.n_samples = int(float(self.N)/100*old_n_samples)
            print("n_samples",self.n_samples)
            keep = np.random.permutation(old_n_samples)[:self.n_samples]
            print("keep",keep)
            new_samples=self.samples[keep]
            print("new_samples",new_samples)
            self.samples = new_samples
            print("self samples",self.samples)
            self.N = 100
        N = int(self.N/100)
        self.synthetic = np.zeros((self.n_samples*N,self.n_attrs))
        self.new_index = 0

        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print("neighbors",neighbors)
        # print("the samples is ",self.samples)
        for i in range(len(self.samples)):
            # print(" samples is ",self.samples[i].reshape(1,-1))
            nnarray = neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #save the k nerghbor's index
            print('nnarray is ',nnarray)
            self.__populate(N,nnarray)
        return self.synthetic

    def __populate(self,N,nnarray):
        for i in range(N):
            nn = np.random.randint(0,self.k)
            dif = self.samples[nnarray[nn]] - self.samples[i]
            gap = np.random.rand(1,self.n_attrs)
            print("gap",gap)
            #gap = np.random.random()
            self.synthetic[self.new_index] = self.samples[i] + gap.flatten()*dif
            self.new_index +=1

from sklearn.datasets import load_iris
iris = load_iris()
x,y = iris.data,iris.target
smote = SMOTE(x)
new_data = smote.over_sampling()
print(smote.over_sampling())
print(new_data.shape)