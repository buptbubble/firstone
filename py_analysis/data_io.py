
import os
import numpy as np
import pandas as pd
import pickle
import codecs
import time
import math
from matplotlib import pyplot as plt

import pywt



class DataIO(object):
    root = os.path.abspath(os.path.dirname(__file__))

    def __init__(self):
        data_path = '..\data_clean'
        self.order_data_path = os.path.join(data_path, 'orderdata')
        self.weather_data_path = os.path.join(data_path, 'weatherdata')
        self.traffic_data_path = os.path.join(data_path, 'trafficdata')
        self.order_diff_data_path = os.path.join(data_path, 'gapdiffdata_train')
        self.do_init()

    def do_init(self):
        self.init_order_data()
        self.init_weather_data()
        self.init_traffic_data()
        self.init_order_diff_data()


    #--------------diff data-----------------#
    def init_order_diff_data(self):
        self.order_diff_data_by_time = None
        if os.path.exists('order_diff.pkl'):
            with codecs.open('order_diff.pkl', 'rb') as f:
                self.order_diff_data_by_time = pickle.load(f)

        else:
            self.order_diff_data_by_time = dict()
            names = ['district_id', 'diff']
            for file in os.listdir(self.order_diff_data_path):
                table = pd.read_csv(os.path.join(
                    self.order_diff_data_path, file),
                    header=None, names=names,
                    index_col=None)
                self.order_diff_data_by_time[file] = table

            with open('order_diff.pkl', 'wb') as f:
                pickle.dump(self.order_diff_data_by_time, f)

    def select_orderDiff_by_ds_distinct(self,dateslice,distinct):
        if len(dateslice.split('-')) == 4:
            if dateslice not in self.order_diff_data_by_time.keys():
                return  None

            df = self.order_diff_data_by_time[dateslice]
            diff = df[df['district_id']==distinct]['diff'].values[0]
            return diff




    #-------------weather data----------------#

    def select_weatherdata_by_dateslice(self,dateslice):
        if(len(dateslice.split('-')))==3:

            weather_day = []
            for i in range(144):
                key = dateslice +'-'+ str(i+1)
                # if key in self.weather_data.index:
                #     weather_day.append(self.weather_data.ix[key])
                # else:
                #     keybefore = dateslice +'-'+ str(i)
                #     keynext = dateslice +'-'+ str(i+2)
                #
                #
                #     if keynext in self.weather_data.index and keybefore in self.weather_data.index:
                #         weather_before = self.weather_data.ix[keybefore]
                #         weather_before.name = key
                #         weather_day.append(weather_before)
                weather = self.select_weatherdata_by_dateslice(key)
                weather_day.append(weather)
            dataAll = pd.DataFrame(weather_day)

            return dataAll


        if (len(dateslice.split('-')))==4:
            if dateslice in self.weather_data.index:
                return self.weather_data.ix[dateslice]
            else:
                year = dateslice.split('-')[0]
                month = dateslice.split('-')[1]
                day = dateslice.split('-')[2]
                slice = int(dateslice.split('-')[3])
                date = year+'-'+month+'-'+day
                for deltaslice in range(6):
                    ds = deltaslice+1
                    daybefore = date+'-'+ str(slice-ds)
                    dayafter = date+'-'+ str(slice+ds)
                    if daybefore in self.weather_data.index :
                        weather = self.weather_data.ix[daybefore]
                        weather.name = dateslice
                        return weather
                    if dayafter in self.weather_data.index:
                        weather = self.weather_data.ix[dayafter]
                        weather.name = dateslice
                        return weather
                return None


    def init_weather_data(self):
        self.weather_data = None
        if os.path.exists('weather.pkl'):
            with codecs.open('weather.pkl', 'rb') as f:
                self.weather_data = pickle.load(f)
            #print(self.weather_data)
        else:
            weathers = pd.DataFrame()
            weather_data = dict()
            for file in os.listdir(self.weather_data_path):

                if file[0] is not '.' and not file.endswith('.py'):
                    names = ['datetime', 'weather', 'temp', 'pm2.5']

                    table = pd.read_table(os.path.join(self.weather_data_path,file), header=None, names = names, index_col=None)
                    today = file.split('_')[-1]
                    last_time_slice, cur_time_slice = 0, 0
                    for index, row in table.iterrows():
                        dt = row['datetime']
                        tm = time.strptime(dt, '%Y-%m-%d %H:%M:%S')
                        cur_time_slice = tm.tm_hour * 6 + math.floor(tm.tm_min / 10) + 1
                        ts = today +'-'+ str(cur_time_slice)
                        # if cur_time_slice == last_time_slice:
                        #     weather_data[ts] = (np.add(weather_data[ts],
                        #                                (row['weather'], row['temp'], row['pm2.5'])) / 2).astype(np.int32)
                        # else:
                        weather_data[ts] = np.array([row['weather'], row['temp'], row['pm2.5']])
                        last_time_slice = cur_time_slice

            df = pd.DataFrame.from_dict(weather_data).T
            df.columns = ['weather', 'temp', 'pm2.5']


            self.weather_data = df
            with open(('weather.pkl'), 'wb') as f:
                pickle.dump(df, f)


    #----------------order data-----------------#
    def init_order_data(self):
        self.order_data_by_time = None
        if os.path.exists('table.pkl'):
            with codecs.open('table.pkl', 'rb') as f:
                self.order_data_by_time = pickle.load(f)

        else:
            self.order_data_by_time = dict()
            names = ['district_id', 'demand', 'supply']
            for file in os.listdir(self.order_data_path):
                table = pd.read_csv(os.path.join(
                                    self.order_data_path, file),
                                    header=None, names=names,
                                    index_col=None)
                self.order_data_by_time[file] = table

            with open('table.pkl', 'wb') as f:
                pickle.dump(self.order_data_by_time, f)


    def select_orderdata_by_timeslice(self, date):
        assert isinstance(date, str)
        return self.order_data_by_time[date]

    #第一个参数可以是xxxx-xx-xx也可以是xxxx-xx-xx-xxx
    def select_orderdata_by_district(self, date, district):
        if len(date.split('-')) == 3:
            df_all = pd.DataFrame()
            for i in range(144):

                timeslice = str(i+1)
                dateslice = date+"-"+timeslice
                df_oneslice = self.select_orderdata_by_timeslice(dateslice)
                df_result = df_oneslice[df_oneslice.district_id == district]
                #df_result['timeslice'] = i+1
                df_result.insert(loc=0, column='timeslice', value=i+1)

                if df_all.size==0:
                    df_all = df_result
                else:
                    df_all=pd.concat([df_all,df_result])

            return df_all
        if len(date.split('-')) == 4:
            df_oneslice = self.select_orderdata_by_timeslice(date)
            df_result = df_oneslice[df_oneslice.district_id == district]
            return df_result


    #--------traffic data---------#
    def init_traffic_data(self):
        self.traffic_data = None
        if os.path.exists('traffic.pkl'):
            with codecs.open('traffic.pkl', 'rb') as f:
                self.traffic_data = pickle.load(f)

        else:
            self.traffic_data = {}
            names = ['timeslice', 'level1', 'level2', 'level3', 'level4']
            for file in os.listdir(self.traffic_data_path):
                table = pd.read_csv(os.path.join(
                    self.traffic_data_path, file),
                    header=None, names=names,
                    index_col=None)
                self.traffic_data[file] = table
            with open('traffic.pkl', 'wb') as f:
                pickle.dump(self.traffic_data, f)

    def select_trafficdata_by_district(self,dateslice,district):
        if len(dateslice.split('-'))==4:
            date = dateslice[0:10]
            slice = int(dateslice.split('-')[3])

            key = date+'-'+str(district)
            traffic_df = self.traffic_data[key]

            traffic_result = traffic_df[traffic_df.timeslice == slice]
            if traffic_result.empty:
                for delta in range(3):
                    traffic_result_af = traffic_df[traffic_df.timeslice == slice + 1 + delta]
                    if not traffic_result_af.empty:
                        traffic_result_af.iloc[0]['timeslice']=slice
                        return traffic_result_af
                    traffic_result_be = traffic_df[traffic_df.timeslice == slice - 1 - delta]
                    if not traffic_result_be.empty:
                        traffic_result_be.iloc[0]['timeslice'] = slice
                        return traffic_result_be
                return traffic_result
            else:
                return traffic_result
        if len(dateslice.split('-'))==3:
            key = dateslice + '-' + str(district)
            traffic_df = self.traffic_data[key]
            return traffic_df

if __name__ == '__main__':
    fileio = DataIO()
#     #print(fileio.select_orderdata_by_timeslice('2016-01-01-100'))
#     print(fileio.select_orderdata_by_district('2016-01-01',1))
    count=0
    prefix = '2016-01-'
    distinct = 10
    for day in range(10):
        day = "{:02}".format(day+1)
        date = prefix+day
        print("----------------"+date+"-------------------")
        slicelist = []
        difflist = []
        gaplist = []
        for slice in range(144):
            dateslice = date+"-"+str(slice+1)


            diff = fileio.select_orderDiff_by_ds_distinct(dateslice,distinct)
            order = fileio.select_orderdata_by_district(dateslice,distinct)
            gap = (order['demand']-order['supply']).values[0]

            if diff != None:
                slicelist.append(slice)
                difflist.append(diff)
                gaplist.append(gap)

        levelsum = 4



        coeffs = pywt.wavedec(difflist,'db3',level = levelsum)

        for level in range(len(coeffs)):
            if level<2:

                if level == 0:
                    continue
                if level == 1:

                    coeffs[level] = pywt.threshold(coeffs[level], 50, 'hard')
                if level == 2:
                    coeffs[level] = pywt.threshold(coeffs[level], 50, 'hard')

                print(coeffs[level])
            else:
                coeffs[level] = pywt.threshold(coeffs[level],0,'less')
                coeffs[level] = pywt.threshold(coeffs[level], 0, 'greater')
                #print(coeffs_thd)
                #coeffs[level] = coeffs_thd

        curve = pywt.waverec(coeffs,'db3')


        # plt.plot(curve, color='r', label='Reconstruction')
        # plt.plot(difflist, color='b', label='Diff_Ori')
        # plt.plot(gaplist, color='g', label='Gap_Ori')
        # plt.legend()
        # plt.show()
        plt.plot(coeffs[0])
    plt.show()

        #exit(0)
        # plt.plot(slicelist,difflist,color = 'r',label = 'Diff')
        # plt.plot(slicelist,gaplist,color = 'b',label = 'Gap')
        #
        #
        # plt.title(date+"   "+str(distinct))
        # plt.legend()
        #
        # plt.show()



