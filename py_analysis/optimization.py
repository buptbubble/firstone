import pandas as pd
from scipy.optimize import fmin
import numpy as np
from matplotlib import pyplot as plt
import math
import analysis_main



def get_color(index):
    colorlist = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    return colorlist[index % len(colorlist)]

def gene_timeslice_feature(value,step=6,length=144):
    count_sum = math.floor(length/step)+1
    pos = math.floor(value/step)
    f = [0]*count_sum
    f[pos] = 1
    return f



class optimization:

    coff = 0
    isTrained = 0
    feature_len = 0
    sigma = 10

    def rbf_kernel(self,x,z,sigma):
        return np.exp(  (-1) * (np.square(np.linalg.norm(x-z))/(2* sigma * sigma))  )

    def opt_func(self,coff_, x_train_, y_train_):
        coff = np.array(coff_)
        ErrRateSum = 0
        for i, label in enumerate(y_train_):
            if label == 0:
                continue
            x = np.array(x_train_[i])
            ErrRate = abs((label - np.dot(x, coff)) / label)
            #ErrRate = abs((label - self.rbf_kernel(x, coff,self.sigma)) / label)
            #print(self.rbf_kernel(x, coff,self.sigma))
            ErrRateSum += ErrRate
        ErrAdj = ErrRateSum + 1 * np.dot(coff, coff)
        return ErrAdj

    def fit(self,x_train,y_train):

        feature_count = len(x_train[0])
        x_init = [1]*feature_count
        self.coff = fmin(self.opt_func,x_init,args=(x_train,y_train),disp=0)
        self.coff = np.array(self.coff)
        self.isTrained = 1
        self.feature_len = feature_count

    def predict(self,feature):
        if self.isTrained == 0:
            print("[Error] Model has not been trained!")
            exit(1)

        f_len = len(feature[0])
        if f_len!=self.feature_len:
            print("[Error] Input feature length is not matching for model!")
            print("Input len is "+str(f_len)+". But model len is "+str(self.feature_len))
            exit(1)
        y_list = []
        for f in feature:
            f = np.array(f)
            predict_y = int(np.dot(f,self.coff))
            if predict_y <0:
                predict_y=0
            y_list.append(predict_y)
        return y_list




