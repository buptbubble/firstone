import math
import multiprocessing
import analysis_main
from Wavelet_Ana import *
from matplotlib import pyplot as plt
from data_io import *


def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

def check_pos_in_list(pos,datalist):
    for item in datalist:
        if (abs(pos[0]-item[0])+abs(pos[1] - item[1]))<0.000000000001:
            return item
    return -1


def isWeekends(date):
    day = int(date.split('-')[-1])
    if day == 1:
        return 1
    else:
        if (day - 1) % 7 == 1:
            return 1
        if (day - 1) % 7 == 2:
            return 2
        else:
            return 0




if __name__ == '__main__':
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
        pool.apply_async(ana_m.train_gap_diff_by_distinctlist, (part, diffcurvelist, count))

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
