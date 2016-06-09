from analysis_main import analysis
from matplotlib import cm
from matplotlib import pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing
import math
import numpy as np

def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

def check_pos_in_list(pos,datalist):
    for item in datalist:
        if (abs(pos[0]-item[0])+abs(pos[1] - item[1]))<0.000000000001:
            return item
    return -1

class analysis_top:
    ana_m = analysis()

    weekend_train = [2, 3, 9, 17]
    weekday_train = [4, 5, 6, 12, 13, 14, 15, 18]



    def do_test_diff_method(self):
        distinctlist = list(range(66))
        dis_sep = chunks(distinctlist, 4)
        manager = multiprocessing.Manager()
        diffcurvelist = manager.list(range(66))
        count = manager.list([0])

        pool = multiprocessing.Pool(processes=4)
        for part in dis_sep:
            pool.apply_async(self.ana_m.train_gap_diff_by_distinctlist,(part,diffcurvelist,count))

        pool.close()
        pool.join()
        print("Calculating MAPE...")
        mape = self.ana_m.verifying_in_training_set_bydiff(diffcurvelist)

        str_w = "MAPE=" + str(mape) + '\n'
        print(str_w)
        return mape



    def do_test_all(self,model = 'OPT',alpha=1,gamma=0.0001):
        distinctlist = list(range(66))
        dis_sep = chunks(distinctlist, 4)
        manager = multiprocessing.Manager()
        clflist = manager.list(range(66))
        count = manager.list([0])



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


        print("Calculating MAPE...")
        mape = self.ana_m.verifying_in_training_set(clflist)
        str_w = "MAPE=" + str(mape) + '\n'
        print(str_w)
        return mape


    def do_test_in_small_sample(self,train_day,test_day,distinct,gamma,alpha,isdrawing=0):

        clf = self.ana_m.train_kernel_ridge_regression_clf(train_day,distinct,gamma,alpha)
        mape = self.ana_m.calculate_mape_by_DayDistinct(clf,test_day,distinct)
        if isdrawing:
            self.ana_m.drawing_perform_by_distinct_daylist(clf,test_day,distinct)
        return mape

    def search_best_model_paras(self, train_day, test_day, distinct):
        point_searched = []
        gamma_start = 0.00002
        gamma_step = 0.000004
        alpha_start = 0.1
        alpha_step = 0.02
        mape = 10000000

        center = [gamma_start,alpha_start]
        step = [gamma_step,alpha_step]
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
            mapelast = mape
            mape = self.four_point_searching(center,step,point_searched,model_paras)
            samepointflag = 0
            if center[0] == centerlast[0] and center[1] == centerlast[1]:
                step[0]/=2
                step[1]/=2
                samepointflag=1


            print("Center After:")
            print("Gamma:", center[0], "Alpha:", center[1])
            print("Gamma step=",step[0],"Alpha step=",step[1])
            print("Current Mape={:.04}\tDelta={:.06}".format(mape,mapelast-mape))


            if (mapelast - mape <0.0005) and samepointflag ==0 :
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




        #
        #
        # for i in range(interval):
        #     gamma = i*gamma_d+gammarange[0]
        #     for j in range(interval):
        #         alpha = j*alpha_d+alpharange[0]
        #         mape = ana_top.do_test_in_small_sample(test_day,test_day,8,gamma,alpha)
        #         data.append([gamma,alpha,mape])
        #         count+=1
        #         proc = int(count/(interval*interval)*100)
        #         print("Processing "+str(proc)+"%")
        #
        #         data_draw = np.array(data)
        #         X = data_draw[:, 0]
        #         Y = data_draw[:, 1]
        #         Z = data_draw[:, 2]
        #
        #         if X.size<6:
        #             continue
        #         fig = plt.figure()
        #         ax = Axes3D(fig)
        #         ax.plot_trisurf(X, Y, Z, cmap=cm.jet)
        #         plt.show(block=False)

    def four_point_searching(self,center,step,point_searched,model_paras):# center,step中 0:gamma  1:alpha
        bestCenter = [0, 0]
        bestMape = 10000000
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
                if isInData[2] < bestMape:
                    #print(isInData[2],bestMape)
                    bestMape = isInData[2]
                    bestCenter[0] = cur_gamma
                    bestCenter[1] = cur_alpha
                    bestDir = direction
            else:
                mape = ana_top.do_test_in_small_sample(model_paras['train_day'], model_paras['test_day'],\
                                                       model_paras['distinct'], cur_gamma, cur_alpha,1)
                point_searched.append([cur_gamma, cur_alpha, mape])
                #print("hat!")
                #print([cur_gamma, cur_alpha, mape])
                #print(point_searched)
                if mape < bestMape:

                    bestMape = mape
                    bestCenter[0] = cur_gamma
                    bestCenter[1] = cur_alpha
                    bestDir = direction
        center[0] = bestCenter[0]
        center[1] = bestCenter[1]

        print ("Best direction:"+str(bestDir))
        return bestMape



if __name__ == '__main__':
    ana_top = analysis_top()
    trainday = [4,5]
    testday = [6,7]
    gamma = 0.0001
    alpha = 5.5

    gamma_range = [0.00001,0.00002]
    alpha_range = [0.1,0.2]
    #ana_top.search_best_model_p
    # aras(trainday, testday, 8)

    #ana_top.do_test_all(model = 'OPT')
    ana_top.do_test_diff_method()
