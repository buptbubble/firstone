import math
import multiprocessing
from tools import *
from Wavelet_Ana import *
from matplotlib import pyplot as plt
from data_io import *
import os
import pickle

from feature import cFeature


class observe:
    clflist = []
    fileio = DataIO()
    feature = cFeature()

    def drawResult(self,date,distinct,clf):

        gaplist = []
        gap_predict_list = []
        slicelist = []
        gap_filter_list = []
        errratelist = []

        count = 0
        errate_sum = 0
        for slice in range(144):
            dateslice = date + "-" + str(slice + 1)
            feature, gap = self.feature.generate(dateslice, distinct)
            if feature == None:
                continue
            gap_predict = clf.predict([feature])
            datetype = isWeekendsText(date)
            filterGap = self.fileio.select_filter_gap(dateslice, distinct, datetype)
            gap_filter_list.append(filterGap)
            gaplist.append(gap)
            gap_predict_list.append(gap_predict)
            if gap != 0:
                errrate = abs((gap - gap_predict) / gap)[0]
                errate_sum += errrate
                count += 1
            else:
                errrate = 0
            errratelist.append(errrate)
            slicelist.append(slice + 1)
        errate_sum /= count

        ax1 = plt.gca()
        ax1.plot(slicelist, gaplist, 'ro-', label='Gap')
        ax1.plot(slicelist, gap_predict_list, 'bo-', label='Predict')
        ax1.plot(slicelist, gap_filter_list, 'yo-', label='Filtered')
        ax2 = ax1.twinx()
        #print(errate_sum)
        legendText = "Error rate:{:.2f}".format(errate_sum)
        ax2.bar(slicelist, errratelist, color='g', alpha=0.2, align='center', label=legendText)
        plt.grid()
        ax1.legend(loc=2)
        ax2.legend(loc=1)

        titleText = date + " " + datetype + " Distinct:" + str(distinct)
        plt.title(titleText)
        plt.show()


    def prediction_observe(self,distinct):
        if os.path.exists('clflist.pkl'):
            with open('clflist.pkl', 'rb') as f:
                clflist = pickle.load(f)
        else:
            print("Clf doesn't exist.")
            exit(0)

        prefix = '2016-01-'
        for day in range(3, 21, 1):
            day = "{:02}".format(day + 1)
            date = prefix + day
            print("----------------" + date + "-------------------")

            gaplist = []
            gap_predict_list = []
            slicelist = []
            gap_filter_list = []
            errratelist = []

            count = 0
            errate_sum = 0
            for slice in range(144):
                dateslice = date + "-" + str(slice + 1)


                isWeekend = isWeekends(date)

                feature, gap = self.feature.generate(dateslice, distinct)
                if feature == None:
                    continue
                if isWeekend == 2:
                    isWeekend=1
                gap_predict = clflist[distinct-1][isWeekend].predict([feature])

                datetype = isWeekendsText(date)
                filterGap = self.fileio.select_filter_gap(dateslice,distinct,datetype)
                gap_filter_list.append(filterGap)
                gaplist.append(gap)
                gap_predict_list.append(gap_predict)
                if gap != 0:
                    errrate = abs((gap-gap_predict)/gap)[0]
                    errate_sum+=errrate
                    count+=1
                else:
                    errrate = 0
                errratelist.append(errrate)

                slicelist.append(slice+1)
            errate_sum/= count

            ax1 = plt.gca()
            ax1.plot(slicelist,gaplist,'ro-',label = 'Gap')
            ax1.plot(slicelist,gap_predict_list,'bo-',label = 'Predict')
            ax1.plot(slicelist,gap_filter_list,'yo-',label = 'Filtered')
            ax2 = ax1.twinx()
            print(errate_sum)
            legendText = "Error rate:{:.2f}".format(errate_sum)
            ax2.bar(slicelist,errratelist,color = 'g',alpha = 0.2,align='center',label = legendText)
            plt.grid()
            ax1.legend(loc = 2)
            ax2.legend(loc = 1 )

            titleText  = date+" "+datetype+" Distinct:"+str(distinct)
            plt.title(titleText)
            plt.show()




if __name__ == '__main__':
    ob = observe()
    ob.prediction_observe(8)
    exit(0)


    clflist = []
    if os.path.exists('clflist.pkl'):
        with open('clflist.pkl', 'rb') as f:
            clflist = pickle.load(f)
    else:
        print("Clf doesn't exist.")
        exit(0)

    distinct = 8
    ana_m = analysis_main.analysis()
    distinctlist = list(range(66))
    testday = range(distinct-2,distinct+2,1)
    dis_sep = chunks(testday, 4)
    manager = multiprocessing.Manager()
    diffcurvelist = manager.list(range(66))
    count = manager.list([0])
    fileio = DataIO()
    wa = wavelet_ana()
    pool = multiprocessing.Pool(processes=4)
    for part in dis_sep:
        pool.apply_async(ana_m.gene_filter_gap_list, (part, diffcurvelist, count))

    pool.close()
    pool.join()
    print("Finish...")

    prefix = '2016-01-'

    for day in range(3,21,1):
        day = "{:02}".format(day + 1)
        date = prefix + day
        print("----------------" + date + "-------------------")
        slicelist = []
        difflist = []
        gaplist = []
        for slice in range(144):
            dateslice = date + "-" + str(slice + 1)

            diff = fileio.select_orderDiff_by_ds_distinct(dateslice, distinct)

            gap = fileio.select_gap(dateslice, distinct)
            if diff != None:
                slicelist.append(slice)
                difflist.append(diff)
                gaplist.append(gap)

        tag = isWeekends(date)
        title = 'Distinct '+str(distinct)+" " +date+"  "
        if tag == 0:
            curve = diffcurvelist[distinct-1]['weekday']
            title +='weekday'
        if tag == 1:
            curve = diffcurvelist[distinct-1]['sat']
            title += 'sat'
        if tag == 2:
            curve = diffcurvelist[distinct-1]['sun']
            title += 'sun'

        diff_pred = [0]
        curve = curve[0:143]
        for i,val in enumerate(curve):
            if i == 142:
                break
            diff_temp = 2*val - difflist[i] +curve[i+1]
            diff_pred.append(diff_temp)



        plt.plot(diff_pred,'gs-',label = 'Pred_Diff')
        plt.plot(curve, color='r', label='Reconstruction')
        plt.plot(difflist, 'bo-', label='Diff_Ori')
        #plt.plot(gaplist, color='g', label='Gap_Ori')
        plt.title(title)
        plt.legend()
        plt.show()
