from pybrain.structure import *
from pybrain.datasets.supervised import *
from pybrain.supervised.trainers import BackpropTrainer
from data_io import DataIO
from tools import *
from feature import  cFeature
import numpy as np
import pickle
import os



class neural_network:
    fnn = FeedForwardNetwork()
    inputlen = 0
    outputlen = 7
    dataio = DataIO()
    feature = cFeature()
    dataset = {}

    def network_init(self):
        #输入feature len
        tempfeature,gap = self.feature.generate('2016-01-03-100',1)
        self.inputlen = len(tempfeature)


        # 设立三层，一层输入层（3个神经元，别名为inLayer），一层隐藏层，一层输出层
        inLayer = LinearLayer(self.inputlen, name='inLayer')
        hiddenLayer1 = SigmoidLayer(7, name='hiddenLayer1')
        hiddenLayer2 = SigmoidLayer(7, name='hiddenLayer2')
        outLayer = LinearLayer(7, name='outLayer')

        # 将三层都加入神经网络（即加入神经元）
        self.fnn.addInputModule(inLayer)
        self.fnn.addModule(hiddenLayer1)
        self.fnn.addModule(hiddenLayer2)
        self.fnn.addOutputModule(outLayer)

        # 建立三层之间的连接
        in_to_hidden1 = FullConnection(inLayer, hiddenLayer1)
        hidden1_to_hidden2 = FullConnection(hiddenLayer1, hiddenLayer2)
        hidden_to_out = FullConnection(hiddenLayer2, outLayer)

        # 将连接加入神经网络
        self.fnn.addConnection(in_to_hidden1)
        self.fnn.addConnection(hidden1_to_hidden2)
        self.fnn.addConnection(hidden_to_out)


        # 让神经网络可用
        self.fnn.sortModules()

    def gene_training_sample(self):

        self.DS = SupervisedDataSet(self.inputlen,self.outputlen)


        if os.path.exists('nn_dataset.pkl'):
            with open('nn_dataset.pkl', 'rb') as f:
                self.dataset = pickle.load(f)
                for i in range(len(self.dataset['feature'])):
                    #print(self.dataset['feature'][i])
                    self.DS.addSample(self.dataset['feature'][i], self.dataset['label'][i])

        else:
            backdaylen = 3
            prefix = '2016-01-'
            loop=0
            featurelist = []
            targetlist = []
            for day in range(2,22,1):
                date = prefix+"{:02}".format(day)
                for distinct in range(1,67):
                    for slice in range(1,145):
                        if slice < backdaylen:
                            continue
                        ts_cur = date+'-'+str(slice)
                        gap_cur = self.dataio.select_gap(ts_cur, distinct)
                        if gap_cur>10:
                            continue

                        f_cur,gap = self.feature.generate(ts_cur,distinct)
                        if f_cur == None:
                            continue
                        output = self.gene_output(gap_cur)

                        featurelist.append(f_cur)
                        targetlist.append(output)


                        loop+=1
                        if loop%1000 == 0:
                            print(loop)


            self.dataset['feature'] = featurelist
            self.dataset['label'] = targetlist
            for i in range(len(featurelist)):
                self.DS.addSample(featurelist[i],targetlist[i])
            print("Building training set is finished. Total amount is {}".format(loop))
            with open('nn_dataset.pkl', 'wb') as f:
                pickle.dump(self.dataset, f)



    def training_nerual_network(self):
        dataTrain,dataTest = self.DS.splitWithProportion(0.7)
        xTrain, yTrain = dataTrain['input'], dataTrain['target']

        xTest, yTest = dataTest['input'], dataTest['target']

        trainer = BackpropTrainer(self.fnn, dataTrain,verbose = True, learningrate=0.03, momentum=0.1)
        trainer.trainUntilConvergence(maxEpochs=20)

        output = self.fnn.activateOnDataset(dataTest)
        count = 0
        countRight = 0
        error = 0
        for i in range(len(output)):
            posReal = yTest[i].argmax()
            posPredict = output[i].argmax()
            #print('o',output[i],posPredict)
            #print('r',yTest[i],posReal)
            error += abs(posReal-posPredict)

            if posReal == posPredict:
                countRight+=1
            count +=1
        error/=count
        print('Correct rate:{:.2f}   Average error:{:.2f}'.format(countRight/count,error))




    def gene_output(self,val):
        output = np.zeros(self.outputlen)
        if val == 0 or val == 1:
            output[0] = 1
        if val == 2:
            output[1] = 1
        if val == 3:
            output[2] = 1
        if val == 4 or val == 5:
            output[3] = 1
        if val == 6 or val == 7:
            output[4] = 1
        if val == 8 or val == 9:
            output[5] = 1
        if val>9:
            output[6] = 1
        return output



if __name__ == '__main__':
    nn = neural_network()
    nn.network_init()
    nn.gene_training_sample()
    nn.training_nerual_network()
