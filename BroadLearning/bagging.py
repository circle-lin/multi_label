"""
本次代码为探讨bagging_BLS的可行性,目前实现了bagging，只要传入模型数目和模型本身，就可以进行bagging，仍需进一步探讨加权的可行性
"""
import numpy as np
from sklearn import preprocessing
from numpy import random
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score,hamming_loss,f1_score
import bls3
import copy
from random import seed
from random import randrange,random
from csv import reader

# Create a random subsample from the dataset with replacement
def subsample(dataset, labelset,ratio=0.4,OOB=False):
    if (type(labelset) is np.ndarray):
        {}
    else:
        labelset = labelset.toarray()
    if (type(dataset) is np.ndarray):
        {}
    else:
        dataset = dataset.toarray()
    n_sample = round(dataset.shape[0] * ratio)
    if OOB:
        used_data=[0 for i in range(dataset.shape[0])]
        sample = np.ndarray((n_sample,dataset.shape[1]))
        label=np.ndarray((n_sample,labelset.shape[1]))
        for i in range(n_sample):
            index = randrange(dataset.shape[0])
            sample[i]=dataset[index]
            label[i]=labelset[index]
            used_data[index]=1
        return sample,label,used_data
    else:
        sample = np.ndarray((n_sample, dataset.shape[1]))
        label = np.ndarray((n_sample, labelset.shape[1]))
        for i in range(n_sample):
            index = randrange(dataset.shape[0])
            sample[i] = dataset[index]
            label[i] = labelset[index]
        return sample, label, []



#design the bagging net
class bagging_net:
    def __init__(self,
                 n_estimator=10,
                 base_estimator=bls3.broadnet(),
                 use_OOB=False
                 ):
        self.n_estimator=n_estimator
        self.base_estimator=base_estimator
        self.pred=[]
        self.bls_net=[]
        self.use_OOB = use_OOB
        self.use_data_sum=[]
        self.sample_OOB=[]
        self.label_OOB=[]


    def fit(self,X_train,y_train):
        if (type(X_train)is np.ndarray):
            {}
        else:
            X_train=X_train.toarray()
        if (type(y_train)is np.ndarray):
            {}
        else:
            y_train=y_train.toarray()
        self.bls_net=[]
        self.use_data_sum = []
        if self.use_OOB:
            used_data_sum=[0 for i in range(X_train.shape[0])]
            for i in range(self.n_estimator):
                sample, label,used_data = subsample(X_train, y_train,ratio=0.4,OOB=self.use_OOB)
                bls_base=copy.deepcopy(self.base_estimator)
                self.bls_net.append(bls_base)
                self.bls_net[i].fit(sample, label)
                used_data_sum=np.array(used_data)+np.array(used_data_sum)
            self.use_data_sum = used_data_sum
        else:
            for i in range(self.n_estimator):
                sample, label,used_data = subsample(X_train, y_train,ratio=0.4,OOB=self.use_OOB)
                bls_base = copy.deepcopy(self.base_estimator)
                self.bls_net.append(bls_base)
                self.bls_net[i].fit(sample, label)

        # print(self.use_data_sum,self.use_data_sum.shape)
        if self.use_OOB:
            num=np.where(used_data_sum==0)
            num=num[0]
            sample_OOB = np.ndarray((len(num), X_train.shape[1]))
            label_OOB = np.ndarray((len(num), y_train.shape[1]))

            for i in range(num.shape[0]):
                sample_OOB[i] = X_train[num[i]]
                label_OOB[i] = y_train[num[i]]
            self.sample_OOB=sample_OOB
            self.label_OOB=label_OOB


    def predict(self,X_test):
        self.pred=[]
        for i in range(self.n_estimator):
            self.pred.append(self.bls_net[i].predict(X_test))
        pred_sum=self.pred[0]
        for i in range(1,len(self.pred)):
            pred_sum+=self.pred[i]
        pred_sum[pred_sum<self.n_estimator/2]=0
        pred_sum[pred_sum >= self.n_estimator / 2] = 1
        return pred_sum




