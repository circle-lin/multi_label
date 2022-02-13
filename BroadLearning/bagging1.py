"""
本代码实现了构建n个分类器进行多标签分类，其中每次的数据集为随机选取0.6的训练集构成，在输入模型的训练集中会分出10%的样本用于生成分类的阈值
其中阈值由最优化(X_alpha,y_alpha)获得,会针对每个标签生成多个阈值
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

#该函数用于对输入数据集进行随机采样获得新的子数据集，其中ratio为采样比例，而OOB为是否记录未被采样到的数据索引，可用于利用OOB数据
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
#该函数用于数据规范化，例如把0.3、0.6、0.9的数据变换为0.5,1,1.5
def scale(list_f1):
    f1_mean=np.mean(list_f1)
    for i in range(len(list_f1)):
        list_f1[i]=list_f1[i]/f1_mean
    return list_f1

#该bagging_net初始化所用到的几个参数分别是，子模型个数，子模型类别，是否采用模型加权，默认为否
#design the bagging net
class bagging_net:
    def __init__(self,
                 n_estimator=10,
                 base_estimator=bls3.broadnet(),
                 use_OOB=False
                 ):
        self.n_estimator=n_estimator
        self.base_estimator=base_estimator
        self.use_OOB = use_OOB
        #为每个基模型预测结果
        self.pred=[]
        #为每个基模型构成的集合
        self.bls_net=[]
        #统计每个样本被抽中的次数
        self.use_data_sum=[]
        #bagging选取算法没有用到的样本
        self.sample_OOB=[]
        #bagging选取算法没有用到的标签
        self.label_OOB=[]
        #用于存储每个模型所对应的权重
        self.W=[]

    #该函数用于模型的训练，只要输入数据即可实现n个子模型的训练，若use_oob为true，则会同时计算每个模型的f1 score以确定每个模型的权重
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
            #采用深复制以避免每个模型是同个地址，无法实现bagging的效果
            for i in range(self.n_estimator):
                sample, label,used_data = subsample(X_train, y_train,ratio=0.4,OOB=self.use_OOB)
                bls_base = copy.deepcopy(self.base_estimator)
                self.bls_net.append(bls_base)
                self.bls_net[i].fit(sample, label)

        # print(self.use_data_sum,self.use_data_sum.shape)
        if self.use_OOB:
            num=np.where(used_data_sum==0)
            num=num[0]
            #采用先创建一个空的数据集，再采用for循环挑选出OOB样本替代掉空样本
            sample_OOB = np.ndarray((len(num), X_train.shape[1]))
            label_OOB = np.ndarray((len(num), y_train.shape[1]))

            for i in range(num.shape[0]):
                sample_OOB[i] = X_train[num[i]]
                label_OOB[i] = y_train[num[i]]
            self.sample_OOB=sample_OOB
            self.label_OOB=label_OOB

            _,X_val,_,y_val=train_test_split(X_train,y_train,test_size=0.1)
            X_val=np.r_[X_val,self.sample_OOB]
            y_val=np.r_[y_val,self.label_OOB]
            self.W= []
            #此处是为了生成分类器权重矩阵，将F1分数减去一个常数，因为F1分数大都大于0.55
            for i in range(self.n_estimator):
                self.W.append(f1_score(y_pred=self.bls_net[i].predict(X_val),y_true=y_val,average='micro'))
            W_min = np.min(self.W)
            for i in range(self.n_estimator):
               self.W[i]=self.W[i]-W_min+0.1

            print(self.W)
            self.W=scale(self.W)
            print(self.W)


    def predict(self,X_test):
        self.pred=[]
        for i in range(self.n_estimator):
            self.pred.append(self.bls_net[i].predict(X_test))
        if self.use_OOB:
            pred_sum = self.pred[0] * self.W[0]
            for i in range(1, len(self.pred)):
                pred_sum += self.pred[i] * self.W[i]
            # print(pred_sum)
        else:
            pred_sum=self.pred[0]
            for i in range(1,len(self.pred)):
                pred_sum+=self.pred[i]
            # print(pred_sum)
        pred_sum[pred_sum<self.n_estimator/2.0]=0
        pred_sum[pred_sum >= self.n_estimator / 2.0] = 1
        return pred_sum




