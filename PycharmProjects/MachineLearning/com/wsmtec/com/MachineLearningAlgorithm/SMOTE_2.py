# -*- coding:utf-8 -*-
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

class SMOTE:
    def __init__(self,samples,N=10,k=5):
        self.samples = samples
        self.N = N
        self.k = k
        self.n_samples,self.n_attrs = samples.shape
        #self.num = num

    def over_sample(self):
        if self.N < 100:
            old_num = self.n_samples
            self.n_samples = int(float(self.N)/100*old_num)
            keep = np.random.permutation(old_num)[:self.n_samples]
            new_samples = self.samples[keep]
            self.samples = new_samples
            self.N = 100
        N = int(self.N/100)
        #print(self.samples.shape)
        self.new_index = 0
        self.result = np.zeros((self.n_samples*N,self.n_attrs))
        #print(self.result.shape)

        # according to the nearest neighbors to find the top K samples
        nb = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        # for every negative data,get the closest point
        for i in range(len(self.samples)):
            nnarray = nb.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print("nnarray",nnarray)
            self.__compute(N,nnarray)
        return self.result
    def __compute(self,N,nnarray):
        for i in range(N):
            nn = np.random.randint(1,self.k)
            # random get a negative data, compute the distance current point and selected point
            diff = self.samples[nnarray[nn]] - self.samples[i]
            gap = np.random.rand(1,self.n_attrs)
            # multiply the distance and the random value, add the result to original data to get new data point
            self.result[self.new_index] = self.samples[i] + gap.flatten()*diff
            #print(self.result[i])
            self.new_index +=1

from sklearn.datasets import load_iris
iris = load_iris()
x,y = iris.data,iris.target
smote = SMOTE(x,N=500)
re = smote.over_sample()
print(re.shape)
