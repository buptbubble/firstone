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
        ErrAdj = ErrRateSum + 0.1 * np.dot(coff, coff)
        return ErrAdj

    def fit(self,x_train,y_train):
        # fw=open('coff.txt','a')
        feature_count = len(x_train[0])
        x_init = [1]*feature_count
        self.coff = fmin(self.opt_func,x_init,args=(x_train,y_train),disp=0)
        #print(self.coff)
        # str_w = ''
        # for c in self.coff:
        #     str_w+= str(c)+','
        # str_w +='\n'
        self.coff = np.array(self.coff)
        self.isTrained = 1
        self.feature_len = feature_count
        # fw.write(str_w)
        # fw.close()

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
            predict_y = np.dot(f,self.coff)
            #predict_y = self.rbf_kernel(f,self.coff,self.sigma)
            if predict_y <0:
                predict_y=0
            y_list.append(predict_y)
        return y_list



if __name__ == '__main__':

    ana = analysis_main.analysis()

    #fr = open("feature.txt",'r')
    names = ['label','timeslice','gap1','gap2','gap3','t1','t2','t3']
    data = pd.read_table("feature.txt",names = names,header= None,sep=',',index_col=None)

    featureset = ['gap1','gap2','gap3','t1','t2','t3']
    featureset = ['gap1', 'gap2', 'gap3']
    f_set = data.loc[:,featureset]
    #f_set['delta1'] = data['gap1']-data['gap2']
    #f_set['delta2'] = data['gap2']-data['gap3']
    # print(f_set)
    # exit(0)
    featureList = []
    for i in range(f_set.shape[0]):
        f = f_set.iloc[i].values.tolist()
        # timeslice = data.iloc[i].values.tolist()[1]
        # print(timeslice)
        # ts_feature = gene_timeslice_feature(timeslice)
        # f.extend(ts_feature)
        f.append(1)
        featureList.append(f)
    label = data['label'].tolist()
    #print(featureList)
    #print(label)

    opt = optimization()
    opt.fit(featureList,label)
    daytest = [4, 5, 6]
    mape = ana.calculate_mape_by_DayDistinct(opt,daytest,8)
    print("Mape=",mape)


    #data['timeslice'] = timeslice


    timeslicelist = []
    predict_y_list = []
    real_y_list = []
    curve_count = 0

    timeslice = data['timeslice']
    labellist = data['label']
    f_length = f_set.shape[0]


    for i in range(f_length):
        ts = timeslice.iloc[i]
        label = labellist.iloc[i]
        feature = f_set.iloc[i].values.tolist()
        feature.append(1)
        # timeslice_cur = data.iloc[i].values.tolist()[1]
        # ts_feature = gene_timeslice_feature(timeslice_cur)
        # feature.extend(ts_feature)

        predict_y = opt.predict([feature])
        print(predict_y)
        predict_y_list.append(predict_y)
        real_y_list.append(label)
        timeslicelist.append(ts)
        if ts == 144:
            color = get_color(curve_count)
            plt.plot(timeslicelist, real_y_list, ls='-', color=color)
            plt.plot(timeslicelist,predict_y_list,ls='--',lw=2,color=color)
            timeslicelist = []
            predict_y_list = []
            real_y_list = []

            curve_count += 1
    plt.show()







