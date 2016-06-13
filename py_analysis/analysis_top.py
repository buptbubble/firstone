from analysis_main import analysis
from matplotlib import cm
from matplotlib import pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing
import math
import numpy as np
from tools import *
import pickle
from observe import observe


def check_pos_in_list(pos,datalist):
    for item in datalist:
        if (abs(pos[0]-item[0])+abs(pos[1] - item[1]))<0.000000000001:
            return item
    return -1

class analysis_top:
    ana_m = analysis()

    weekend_train = [2, 3, 9, 17]
    weekday_train = [4, 5, 6, 12, 13, 14, 15, 18]
    obs = observe()


    def do_test_diff_method(self):
        distinctlist = list(range(66))
        dis_sep = chunks(distinctlist, 4)
        manager = multiprocessing.Manager()
        diffcurvelist = manager.list(range(66))
        count = manager.list([0])

        pool = multiprocessing.Pool(processes=4)
        for part in dis_sep:
            pool.apply_async(self.ana_m.gene_filter_gap_list, (part, diffcurvelist, count))

        pool.close()
        pool.join()
        print("Calculating MAPE...")
        mape = self.ana_m.verifying_in_training_set_bydiff(diffcurvelist)

        str_w = "MAPE=" + str(mape) + '\n'
        print(str_w)
        return mape



    def do_test_all(self,model = 'OPT',gamma=1,alpha=0.0001):
        distinctlist = list(range(66))
        dis_sep = chunks(distinctlist, 4)
        manager = multiprocessing.Manager()
        clflist = manager.list(range(66))
        count = manager.list([0])

        # part = range(66)
        # self.ana_m.train_OPT_clf_bydaylist(part, clflist, count)

        pool = multiprocessing.Pool(processes=4)
        for part in dis_sep:
            if model == 'KRR':
                pool.apply_async(self.ana_m.gene_KRR_clf_bydaylist, (part, clflist, count, gamma, alpha))
            elif model == 'OPT':
                pool.apply_async(self.ana_m.train_OPT_clf_bydaylist, (part, clflist, count))
            else:
                print("[Error] \""+model+"\" is what?")
                exit(1)

        pool.close()
        pool.join()


        clflist = list(clflist)
        with open('clflist.pkl', 'wb') as f:
            pickle.dump(clflist, f)

        print("Calculating MAPE...")
        mape = self.ana_m.verifying_in_training_set(clflist)
        str_w = "MAPE=" + str(mape) + '\n'
        print(str_w)
        return mape


    def do_test_in_small_sample(self,train_day,test_day,distinct,isdrawing=0,gamma=0.1,alpha=0.001):

        clf = self.ana_m.train_kernel_ridge_regression_clf(train_day,distinct,gamma,alpha)
        #clf = self.ana_m.train_optimzation_model(train_day,distinct)
        mape = self.ana_m.calculate_mape_by_DayDistinct(clf,test_day,distinct)
        norm2err = self.ana_m.calculate_norm2_error(clf,test_day,distinct)

        if isdrawing:
            print('mape:', mape)
            testdaylist = select_test_day(test_day)
            for day in testdaylist:
                self.obs.drawResult(day,distinct,clf)
        return mape,norm2err

    def train_KRR_returnNorm2Error(self, train_day, test_day, distinct, isdrawing=0, gamma=0.1, alpha=0.001):

        clf = self.ana_m.train_kernel_ridge_regression_clf(train_day, distinct, gamma, alpha)
        # clf = self.ana_m.train_optimzation_model(train_day,distinct)
        mape = self.ana_m.calculate_mape_by_DayDistinct(clf, test_day, distinct)

        return mape

    def search_best_model_paras(self, center,train_day, test_day, distinct):
        point_searched = []


        norm2 = 10000000


        step = [center[0]/2,center[1]/2]
        model_paras = {}
        model_paras['test_day']=test_day
        model_paras['train_day'] = train_day
        model_paras['distinct'] = distinct
        count=0
        while(1):
            count+=1
            print("---------------Round "+str(count)+"-----------------")
            print("Center Before:")
            print("Gamma:",center[0],"Alpha:",center[1])
            centerlast = center[:]
            norm2last = norm2
            norm2,mape = self.four_point_searching(center,step,point_searched,model_paras)
            samepointflag = 0
            if center[0] == centerlast[0] and center[1] == centerlast[1]:
                step[0]/=2
                step[1]/=2
                samepointflag=1


            print("Center After:")
            print("Gamma:", center[0], "Alpha:", center[1])
            print("Gamma step=",step[0],"Alpha step=",step[1])
            print("Cur Norm2 ={:.04}\tCur mape = {:.04f}\tDelta={:.06}".format(norm2,mape,norm2last-norm2))


            if (norm2last - norm2 <0.0005) and samepointflag ==0 :
                break

        data_draw = np.array(point_searched)
        # print(point_searched)
        X = data_draw[:, 0]
        Y = data_draw[:, 1]
        Z = data_draw[:, 2]

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_trisurf(X, Y, Z, cmap=cm.jet)
        plt.show()





    def four_point_searching(self,center,step,point_searched,model_paras):# center,step中 0:gamma  1:alpha
        bestCenter = [0, 0]
        bestMape = 10000000
        bestNorm2 = 10000000
        bestDir = -1
        for direction in range(5):
            if direction == 0:
                cur_gamma = center[0]
                cur_alpha = center[1]
            if direction == 1:
                cur_gamma = center[0] + step[0]
                cur_alpha = center[1]
            if direction == 2:
                cur_gamma = center[0]
                cur_alpha = center[1] + step[1]
            if direction == 3:
                while center[0] - step[0] <= 0:
                    step[0] /= 2
                cur_gamma = center[0] - step[0]
                cur_alpha = center[1]
            if direction == 4:
                cur_gamma = center[0]
                while center[1] - step[1] <= 0:
                    step[1] /= 2
                cur_alpha = center[1] - step[1]

            isInData = check_pos_in_list([cur_gamma, cur_alpha], point_searched)
            if isInData != -1:
                #print("hit!")
                #print("---->",isInData,"cur--->",[cur_gamma, cur_alpha])
                if isInData[2] < bestNorm2:
                    #print(isInData[2],bestMape)
                    bestNorm2 = isInData[2]
                    bestMape = isInData[3]
                    bestCenter[0] = cur_gamma
                    bestCenter[1] = cur_alpha
                    bestDir = direction
            else:
                mape,norm2err = ana_top.do_test_in_small_sample(model_paras['train_day'], model_paras['test_day'],\
                                                       model_paras['distinct'],0 ,cur_gamma, cur_alpha)
                point_searched.append([cur_gamma, cur_alpha, norm2err,mape])
                #print("hat!")
                #print([cur_gamma, cur_alpha, mape])
                #print(point_searched)
                print('norm2:',norm2err,'\tmape:',mape)
                if norm2err < bestNorm2:
                    bestNorm2 = norm2err
                    bestMape = mape
                    bestCenter[0] = cur_gamma
                    bestCenter[1] = cur_alpha
                    bestDir = direction
        center[0] = bestCenter[0]
        center[1] = bestCenter[1]

        print ("Best direction:"+str(bestDir))
        return bestNorm2,bestMape



if __name__ == '__main__':
    ana_top = analysis_top()
    trainday = [4,5,8,11,13,14,15,18,19]
    #trainday = [4,5]
    testday = [4,6,7,12]
    distinct = 9

    gamma = 2.5e-5
    alpha = 0.01

    bestgamma = 2.5e-5
    bestalpha = 0.001

    # ana_top.do_test_all('KRR',bestgamma,bestalpha)
    # exit(0)

    mape,norm2 = ana_top.do_test_in_small_sample(trainday,testday,distinct,1,gamma,alpha)
    exit(0)

    # center = [gamma,alpha]
    # step = [gamma/2,alpha/2]
    # pointsearched = []
    # ana_top.search_best_model_paras(center,trainday,testday,9)


