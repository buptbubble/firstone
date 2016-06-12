from data_io import DataIO
from tools import *
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import datetime
import math
import os
from sklearn.kernel_ridge import KernelRidge
import multiprocessing
import time
import random
from optimization import *
from Wavelet_Ana import *
from feature import cFeature


#需要测试的 ： 7,8,10,16,19

#weekend = [ 2, 3, 9, 10, 16, 17]
#weekday = [4, 7, 8, 12, 13, 14, 15, 18, 19]

weekend = [ 2, 3, 9, 17]
weekday = [4,5,6,12, 13, 14, 15, 18]

def gene_timeslice_feature(value,step=6,length=144):
    count_sum = math.floor(length/step)+1
    pos = math.floor(value/step)
    f = [0]*count_sum
    f[pos] = 1
    return f

def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

def get_color(index):
    colorlist = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    return colorlist[index % len(colorlist)]

class analysis:
    dataio = DataIO()
    feature = cFeature()
    wa = wavelet_ana()
    verify_file_path = './predict_data_in_training.txt'

    weekend = [2, 3, 9, 17]
    weekday = [4, 5, 6, 12, 13, 14, 15, 18]
    Sat = [2,9]
    Sun = [3,17]



    def time2slice(self, i_time):
        t_array = datetime.datetime.strptime(i_time, "%Y-%m-%d %H:%M:%S")
        slice = t_array.hour * 6 + math.floor(t_array.minute / 10) + 1

        return slice
    def slice2time(self,slice):
        slice = int(slice)
        hour = math.floor((slice-1)/6)
        min = (slice-1-hour*6)*10
        timenow = "{:02}:{:02}".format(hour,min)
        return timenow



    def select_test_day(self,daylist):
        daytest = []
        for i in daylist:
            day = '{:02d}'.format(i)
            prefix = '2016-01-'
            date = prefix + day
            daytest.append(date)

        return daytest

    def weather_main_trend(self,date,hour_interval=1):
        #print(self.dataio.select_weatherdata_by_dateslice(date))
        weatherlist = []
        for i in range(1,144,6*hour_interval):
            dateslice = date+'-'+str(i)
            weather = self.dataio.select_weatherdata_by_dateslice(dateslice)
            if date == '2016-01-16':
                print(weather)

            if type(weather)!= type(None) :
                weatherlist.append(weather)
        if len(weatherlist)==0:
            print("len(weatherlist)==0")
            exit(1)

        weatherPD = pd.DataFrame(weatherlist)
        if date == '2016-01-16':
            print(weatherPD)

        #del weatherPD['temp']
        #del weatherPD['pm2.5']

        timelist = []
        for idx in weatherPD.index:
            slice = idx.split('-')[-1]

            timetext = self.slice2time(slice)
            timelist.append(timetext)
        weatherPD.index = timelist
        return weatherPD



    def write_weather_info(self):
        for day in range(21):

            prefix = '2016-01-'
            date = prefix+'{:02d}'.format(day+1)
            print(date)
            pd_weather = self.weather_main_trend(date)
            filepath = './weather_info'
            filename = date+".txt"
            fw = open(os.path.join(filepath,filename),'w')
            pd_weather.to_csv(fw)
            fw.close()

    def do_analysis_drawGapTrend(self):
        weekend = [1, 2, 3, 9, 10, 16, 17]
        weekday1 = [4, 5, 6, 7, 8]
        weekday2 = [11, 12, 13, 14, 15]

        for type in range(3):

            if type ==0:
                daytest = self.select_test_day(weekend)
                ax = plt.subplot(311)
                ax.set_title("weekend")
            if type == 1:
                daytest = self.select_test_day(weekday1)
                ax = plt.subplot(312)
                ax.set_title("weekday1")
            if type == 2:
                daytest = self.select_test_day(weekday2)
                ax = plt.subplot(313)
                ax.set_title("weekday2")

            for day in daytest:
                data = self.dataio.select_orderdata_by_district(day, 8)
                gap = (data['demand'] - data['supply'])
                gaplen = gap.shape[0]
                idx = np.array(range(gaplen)) + 1
                x_label = []
                for i in range(144):
                    x_label.append(ana.slice2time(i + 1))
                gap.index = x_label
                gap.plot(label=day)


            ax.legend(loc=2)
        plt.show()

    def train_kernel_ridge_regression_clf(self,train_daylist,distinct,gamma=1,alpha=1):
        daytest = self.select_test_day(train_daylist)
        y_train = []
        X_train = []

        for day in daytest:
            for slice in range(144):
                dateslice = day+'-'+str(slice+1)
                #feature,gap = self.generateFeatureLabel(dateslice,distinct)
                feature, gap = self.feature.generate(dateslice, distinct)


                if feature != None:
                    if gap != 0:
                        gap = math.log10(float(gap))
                    else:
                        gap = -0.1
                    X_train.append(feature)
                    y_train.append(gap)
        clf = KernelRidge(kernel='polynomial',  gamma=gamma,alpha=alpha)
        #clf = KernelRidge(kernel='polynomial', degree=3,alpha=0.01)
        clf.fit(X_train, y_train)

        return clf

    def train_optimzation_model(self,train_daylist,distinct):
        daytest = self.select_test_day(train_daylist)
        y_train = []
        X_train = []

        for day in daytest:
            for slice in range(144):
                dateslice = day + '-' + str(slice + 1)
                #feature, label = self.generateFeatureLabel(dateslice, distinct)
                feature, label = self.feature.generate(dateslice, distinct)
                #print(feature,label)
                #print(feature1,label1)
                #print("-----------")

                if feature != None:
                    X_train.append(feature)
                    y_train.append(label)

        opt = optimization()
        opt.fit(X_train,y_train)
        return opt

    def train_gap_diff_curve(self,day,distinct):
        if len(day.split('-')) != 3:
            print("The input of train_gap_diff_curve_by_distinct_day should be a xx-xx-xx")
            exit(1)

        difflist = []
        for slice in range(144):
            dateslice = day + '-' + str(slice + 1)
            diffval = self.dataio.select_orderDiff_by_ds_distinct(dateslice,distinct)
            if diffval != None:
                difflist.append(diffval)
        coeffs = self.wa.get_wavelet_coeffs(difflist)
        #coeffs = self.wa.coeffs_process(coeffs)
        curve = self.wa.reconstruction_from_coeffs(coeffs)
        return np.array(curve)

    def train_gap_diff_by_distinctlist(self,distinct_list,diffcurveList,count):
        for distinct in distinct_list:
            count[0]+=1
            print("Training model in " + "{:.1f}".format(count[0] / 66 * 100) + "% completed...")
            curve_dict = {}

            weekday = self.select_test_day(self.weekday)
            curve_sum = np.zeros(144)

            for day in weekday:
                curve = self.train_gap_diff_curve(day,distinct+1)
                curve_sum+=curve
            curve_dict['weekday'] = curve_sum/len(weekday)


            sat = self.select_test_day(self.Sat)
            curve_sum = np.zeros(144)
            for day in sat:
                curve = self.train_gap_diff_curve(day, distinct + 1)
                curve_sum += curve
            curve_dict['sat'] = curve_sum/len(sat)


            sun = self.select_test_day(self.Sun)
            curve_sum = np.zeros(144)
            for day in sun:
                curve = self.train_gap_diff_curve(day, distinct + 1)
                curve_sum += curve
            curve_dict['sun'] = curve_sum / len(sun)

            diffcurveList[distinct] = curve_dict


    def drawing_perform_by_distinct_daylist(self,clf,daylist,distinct):
        daytest = self.select_test_day(daylist)
        for i,day in enumerate(daytest):
            gap_real = []
            gap_predict = []
            slice_x = []
            for slice in range(144):
                dateslice = day+'-'+str(slice+1)
                #feature,gap = self.generateFeatureLabel(dateslice,distinct)
                feature, gap = self.feature.generate(dateslice, distinct)
                if feature == None:
                    continue
                label_predicted = clf.predict([feature])
                gap_real.append(gap)
                gap_predict.append(label_predicted)
                slice_x.append(slice)

            plt.plot(slice_x,gap_real,color = get_color(i),label =day )
            plt.plot(slice_x,gap_predict,color = get_color(i),ls='--',lw=2)
        plt.legend(loc=2)
        plt.grid()
        plt.show()

    def verifying_in_training_set(self,clf):
        fr = open(self.verify_file_path,'r')
        timeslicelist = []
        for line in fr:
            timeslice = line.split(' ')[0]
            timeslicelist.append(timeslice)
        fr.close()
        #------clf------distinct(0,65)-------type(0:weekday, 1:weekend)-----
        count = 0
        err_rate_sum = 0
        for timeslice in timeslicelist:
            for dis_ind in range(66):
                #clf[distinct][]
                distinct = dis_ind+1

                date = timeslice[0:10]

                isWeekend = isWeekends(date)
                #feature,gap = self.generateFeatureLabel(timeslice,distinct)
                feature,gap = self.feature.generate(timeslice,distinct)

                if feature == None or gap == 0:
                    continue
                gap_predicted = clf[dis_ind][isWeekend].predict([feature])[0]

                gap_predicted = int(math.pow(10,gap_predicted))
                if gap_predicted<0:
                    gap_predicted =0
                err_rate = abs((gap-gap_predicted)/gap)

                err_rate_sum+=err_rate
                count+=1

        err_rate_sum/=count
        return err_rate_sum

    def verifying_in_training_set_bydiff(self,diffcurve):
        fr = open(self.verify_file_path, 'r')
        timeslicelist = []
        for line in fr:
            timeslice = line.split(' ')[0]
            timeslicelist.append(timeslice)
        fr.close()
        # ------clf------distinct(0,65)-------type(0:weekday, 1:weekend)-----
        count = 0
        err_rate_sum = 0
        for timeslice in timeslicelist:
            for dis_ind in range(66):

                distinct = dis_ind + 1
                slice = int(timeslice.split('-')[-1])
                date = timeslice[0:10]

                gap = self.dataio.select_gap(timeslice, distinct)
                if gap == 0:
                    continue

                ts_before1 = date+'-'+str(slice-1)
                ts_before2 = date+'-'+str(slice-2)
                ts_before3 = date+'-'+str(slice-3)
                gap1 = self.dataio.select_gap(ts_before1,distinct)
                gap2 = self.dataio.select_gap(ts_before2,distinct)
                gap3 = self.dataio.select_gap(ts_before3,distinct)
                diff1 = gap1-gap2
                diff2 = gap2-gap3




                daytype = isWeekends(date)
                diffval1 = 0
                diffval0 = 0
                if daytype == 0:
                    curve = diffcurve[dis_ind]['weekday']
                    diffval0 = curve[slice - 1]
                    diffval1 = curve[slice - 2]
                if daytype == 1:
                    curve = diffcurve[dis_ind]['sat']
                    diffval0 = curve[slice - 1]
                    diffval1 = curve[slice - 2]
                if daytype == 2:
                    curve = diffcurve[dis_ind]['sun']
                    diffval0 = curve[slice - 1]
                    diffval1 = curve[slice - 2]

                gapdiff_predict = 2*diffval1-diff1+diffval0
                gap_predicted =gap1+gapdiff_predict
                if gap_predicted<0:
                    gap_predicted=0
                err_rate = abs((gap - gap_predicted) / gap)

                err_rate_sum += err_rate
                count += 1

        err_rate_sum /= count
        return err_rate_sum

    def calculate_norm2_error(self, clf, daylist, distinct):
        err_val = 0
        count = 0
        daylist = self.select_test_day(daylist)
        for date in daylist:
            for slice in range(144):
                timeslice = date + '-' + str(slice + 1)

                feature, gap = self.feature.generate(timeslice, distinct)

                if feature == None or gap == 0:
                    continue
                if gap !=0:
                    gap_log = math.log10(gap)
                else:
                    gap_log = 0

                gap_predicted = clf.predict([feature])[0]

                err_val+=(gap_log-gap_predicted)**2
                count+=1
        err_val /= count
        return err_val

    def calculate_mape_by_DayDistinct(self,clf,daylist,distinct):
        err_rate_sum=0
        count=0
        daylist = self.select_test_day(daylist)
        for date in daylist:
            for slice in range(144):
                timeslice = date+'-'+str(slice+1)
                #feature, gap = self.generateFeatureLabel(timeslice, distinct)
                feature, gap = self.feature.generate(timeslice, distinct)

                if feature == None or gap == 0:
                    continue
                gap_predicted = clf.predict([feature])[0]
                #print('Before log:',gap_predicted)
                gap_predicted = int(math.pow(10, gap_predicted))

                if gap_predicted<0:
                    gap_predicted = 0
                isWeekend = isWeekendsText(date)
                gap_filtered = self.dataio.select_filter_gap(timeslice,distinct,isWeekend)
                if gap_predicted>2*gap_filtered:
                    gap_predicted = 2*gap_filtered
               # print('After log:', gap_predicted,gap)
                err_rate = abs((gap - gap_predicted) / gap)
                #print(timeslice+"\t{:.2f}\t{}\t{:.0f}".format(err_rate,gap,gap_predicted))
                err_rate_sum += err_rate
                count += 1
        err_rate_sum /= count
        return err_rate_sum

    #
    # def generateFeatureLabel(self,dateslice,distinct):
    #     date = dateslice[0:10]
    #     weather = self.dataio.select_weatherdata_by_dateslice(dateslice)
    #     if type(weather) == type(None):
    #         #print("Weather info. does not exist in "+dateslice)
    #         return None,None
    #
    #
    #
    #     weather_feature = [0] * 4
    #     cur_weather = int(weather['weather'])
    #     if cur_weather == 2 or cur_weather == 3 or cur_weather == 4:
    #         weather_feature[0] = 1
    #     elif cur_weather == 8:
    #         weather_feature[1] = 1
    #     elif cur_weather == 9:
    #         weather_feature[2] = 1
    #     else:
    #         weather_feature[3] = 1
    #     #print(weather_feature)
    #     #weather_feature[int(weather['weather']) - 1] = 1
    #
    #     orderdata = self.dataio.select_orderdata_by_district(dateslice,distinct)
    #     gap_real = (orderdata['demand']-orderdata['supply']).values
    #     gap_real = gap_real[0]
    #     timeslice = int(dateslice.split('-')[-1])
    #     if timeslice <4:
    #         return None,None
    #     traffic_info = self.dataio.select_trafficdata_by_district(dateslice,distinct)
    #     if traffic_info.empty and distinct !=54:
    #         return None,None
    #
    #     ts_feature = gene_timeslice_feature(timeslice,4)
    #
    #
    #     result = isWeekends(date)
    #     if result == 0:
    #         daytype = 'weekday'
    #     if result == 1:
    #         daytype = 'sat'
    #     if result == 2:
    #         daytype = 'sun'
    #
    #     gap_filtered = self.dataio.select_filter_gap(dateslice,distinct,daytype)
    #     gap_filtered_last = self.dataio.select_filter_gap(get_last_ts(dateslice),distinct,daytype)
    #     traffic_level =[1,1,1,1]
    #     if not traffic_info.empty:
    #         level1 = (traffic_info['level1'].values)[0]
    #         level2 = (traffic_info['level2'].values)[0]
    #         level3 = (traffic_info['level3'].values)[0]
    #         level4 = (traffic_info['level4'].values)[0]
    #         traffic_level[0] = level1
    #         traffic_level[1] = level2
    #         traffic_level[2] = level3
    #         traffic_level[3] = level4
    #
    #     #print(traffic_level)
    #
    #     trafficBeList = []
    #     GapBeList = []
    #     for delta in range(3):
    #         datesliceBe = dateslice[0:11]+str(timeslice-delta-1)
    #         orderdataBe = self.dataio.select_orderdata_by_district(datesliceBe, distinct)
    #         gap_real_Be = (orderdataBe['demand'] - orderdataBe['supply']).values
    #         gap_real_Be = gap_real_Be[0]
    #         GapBeList.append(gap_real_Be)
    #
    #         traffic_info = self.dataio.select_trafficdata_by_district(datesliceBe,distinct)
    #         if not traffic_info.empty:
    #             level1 = (traffic_info['level1'].values)[0]
    #             level2 = (traffic_info['level2'].values)[0]
    #             level3 = (traffic_info['level3'].values)[0]
    #             level4 = (traffic_info['level4'].values)[0]
    #             traffic_temp = level1 + level2 * 2 + level3 * 3 + level4 * 4
    #         else:
    #             traffic_temp = 1
    #         trafficBeList.append(traffic_temp)
    #
    #
    #     #GapBeListExp2 = [x*x for x in GapBeList]
    #     GapBeListExp2 = math.pow(GapBeList[0],2)
    #     #GapBeListExp2 = math.exp(GapBeList[0])
    #     feature = []
    #
    #     #feature.extend(weather_feature)
    #     feature.extend(GapBeList)
    #     #feature.extend(ts_feature)
    #     feature.append(gap_filtered)
    #     feature.append(gap_filtered_last)
    #
    #     # diff = abs(gap_filtered - GapBeList[0])
    #     # diff_exp05 = math.pow(diff,0.5)
    #     # if gap_filtered - GapBeList[0]>0:
    #     #     pass
    #     # else:
    #     #     diff_exp05 *= -1
    #
    #     #feature.append(math.log(gap_filtered-GapBeList[0]))
    #
    #     #feature.append(math.pow((GapBeList[0] - GapBeList[1]),2))
    #     #feature.extend(GapBeListExp2)
    #     #feature.append(diff_exp05)
    #
    #
    #     #feature.extend(traffic_level)
    #     #feature.extend(trafficBeList)
    #     feature.append(1)
    #
    #     return feature,gap_real

    def gene_KRR_clf_bydaylist(self,distinct_list,clflist,count,gamma=1,alpha=1):

        for distinct in distinct_list:
            rand = random.random()
            time.sleep(rand/10)
            count[0]+=1
            print("Training model in " + "{:.1f}".format(count[0]/66*100) + "% completed...")
            clf_weekday = self.train_kernel_ridge_regression_clf(weekday, distinct + 1,gamma,alpha)
            clf_weekend = self.train_kernel_ridge_regression_clf(weekend, distinct + 1,gamma,alpha)
            clflist[distinct] = [clf_weekday,clf_weekend]
    def train_OPT_clf_bydaylist(self,distinct_list,clflist,count):
        for distinct in distinct_list:
            rand = random.random()
            time.sleep(rand / 10)
            count[0] += 1
            print("Training model in " + "{:.1f}".format(count[0] / 66 * 100) + "% completed...")
            clf_weekday = self.train_optimzation_model(self.weekday,distinct + 1)
            clf_weekend = self.train_optimzation_model(self.weekend,distinct + 1)
            clflist[distinct] = [clf_weekday, clf_weekend]
