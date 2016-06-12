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
            gap_predict = clf.predict([feature])[0]

            gap_predict = int(math.pow(10, gap_predict))
            if gap_predict<0:
                gap_predict=0
            isWeekend = isWeekendsText(date)
            gap_filtered = self.fileio.select_filter_gap(dateslice, distinct, isWeekend)
            if gap_predict > 2 * gap_filtered:
                gap_predict = 2 * gap_filtered

            datetype = isWeekendsText(date)
            filterGap = self.fileio.select_filter_gap(dateslice, distinct, datetype)
            gap_filter_list.append(filterGap)
            gaplist.append(gap)
            gap_predict_list.append(gap_predict)
            if gap != 0:
                errrate = abs((gap - gap_predict) / gap)
                errate_sum += errrate
                count += 1
            else:
                errrate = 0
            errratelist.append(errrate)
            slicelist.append(slice + 1)
        errate_sum /= count
        plt.subplot(211)
        ax1 = plt.gca()
        ax1.plot(slicelist, gaplist, 'ro-', label='Gap')
        ax1.plot(slicelist, gap_predict_list, 'bo-', label='Predict')
        ax1.plot(slicelist, gap_filter_list, 'yo-', label='Filtered')
        ax2 = ax1.twinx()

        legendText = "Error rate:{:.2f}".format(errate_sum)
        ax2.bar(slicelist, errratelist, color='g', alpha=0.2, align='center', label=legendText)
        plt.grid()
        ax1.legend(loc=2)
        ax2.legend(loc=1)

        titleText = date + " " + datetype + " Distinct:" + str(distinct)
        plt.title(titleText)
        plt.subplot(212)

        plt.plot(slicelist, gaplist, 'ro-', label='Gap')
        plt.plot(slicelist, gap_predict_list, 'bo-', label='Predict')
        plt.plot(slicelist, gap_filter_list, 'yo-', label='Filtered')
        plt.grid()
        plt.legend(loc = 2)
        plt.ylim(0,10)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.bar(slicelist, errratelist, color='g', alpha=0.2, align='center', label=legendText)
        ax2.set_ylim(0,1)
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
            #print("----------------" + date + "-------------------")

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
                gap_predict = clflist[distinct-1][isWeekend].predict([feature])[0]

                gap_predict = int(math.pow(10,gap_predict))

                datetype = isWeekendsText(date)
                filterGap = self.fileio.select_filter_gap(dateslice,distinct,datetype)
                gap_filter_list.append(filterGap)
                gaplist.append(gap)
                gap_predict_list.append(gap_predict)
                if gap != 0:
                    errrate = abs((gap-gap_predict)/gap)
                    errate_sum+=errrate
                    count+=1
                else:
                    errrate = 0
                errratelist.append(errrate)

                slicelist.append(slice+1)
            errate_sum/= count


            plt.subplot(211)
            ax1 = plt.gca()
            ax1.plot(slicelist,gaplist,'ro-',label = 'Gap')
            ax1.plot(slicelist,gap_predict_list,'bo-',label = 'Predict')
            ax1.plot(slicelist,gap_filter_list,'yo-',label = 'Filtered')
            ax2 = ax1.twinx()

            legendText = "Error rate:{:.2f}".format(errate_sum)
            ax2.bar(slicelist,errratelist,color = 'g',alpha = 0.2,align='center',label = legendText)
            plt.grid()
            ax1.legend(loc = 2)
            ax2.legend(loc = 1 )

            titleText  = date+" "+datetype+" Distinct:"+str(distinct)
            plt.title(titleText)

            plt.subplot(212)
            plt.plot(slicelist, gaplist, 'ro-', label='Gap')
            plt.plot(slicelist, gap_predict_list, 'bo-', label='Predict')
            plt.plot(slicelist, gap_filter_list, 'yo-', label='Filtered')
            plt.grid()
            plt.legend(loc=2)
            plt.ylim(0, 10)
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.bar(slicelist, errratelist, color='g', alpha=0.2, align='center', label=legendText)
            ax2.set_ylim(0, 1)
            plt.show()




if __name__ == '__main__':
    ob = observe()
    ob.prediction_observe(9)
    exit(0)

