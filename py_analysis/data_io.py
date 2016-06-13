
import os
import numpy as np
import pandas as pd
import pickle
import codecs
import time
import math
from matplotlib import pyplot as plt
#from Wavelet_Ana import *
import multiprocessing
from tools import *

def singleton(cls):
    instances = {}
    def _wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)

        return instances[cls]
    return _wrapper

@singleton
class DataIO:
    root = os.path.abspath(os.path.dirname(__file__))

    weekend = [2, 3, 9, 17]
    weekday = [4, 5, 6, 12, 13, 14, 15, 18]
    Sat = [2, 9]
    Sun = [3, 17]
    #wa = wavelet_ana()

    def __init__(self):
        data_path = '..\data_clean'
        self.order_data_path = os.path.join(data_path, 'orderdata')
        self.weather_data_path = os.path.join(data_path, 'weatherdata')
        self.traffic_data_path = os.path.join(data_path, 'trafficdata')
        self.order_diff_data_path = os.path.join(data_path, 'gapdiffdata_train')
        self.do_init()

    def do_init(self):
        self.init_order_data()
        self.init_order_diff_data()
        self.init_weather_data()
        self.init_traffic_data()

        self.init_filter_gap()

    #---------------filtered GAP----------------#
    def init_filter_gap(self):
        print("Init filtered gap data...",__name__)
        self.filter_gap_data = None
        if os.path.exists('gap_filtered.pkl'):
            with codecs.open('gap_filtered.pkl', 'rb') as f:
                self.filter_gap_data = pickle.load(f)
        else:
            distinctlist = list(range(66))
            dis_sep = chunks(distinctlist, 4)
            manager = multiprocessing.Manager()
            filterd_gap_list = manager.list(list(range(66)))

            count = manager.list([0])



            # part = list(range(66))
            # self.gene_filter_gap_list(part, filterd_gap_list, count)

            pool = multiprocessing.Pool(processes=4)
            for part in dis_sep:
                print(part)
                pool.apply_async(self.gene_filter_gap_list, (part, filterd_gap_list, count))

            pool.close()
            pool.join()

            self.filter_gap_data = list(filterd_gap_list)
            with open('gap_filtered.pkl', 'wb') as f:
                pickle.dump(self.filter_gap_data, f)

    def select_filter_gap(self,ts,distinct,type):

        if len(ts.split('-'))!=4:
            return None
        else:
            slice = int(ts.split('-')[-1])
            if slice >=144:
                return None
            else:
                return self.filter_gap_data[distinct-1][type][slice-1]


    def gene_filter_gap_list(self, distinct_list, filter_gap_list, count):
        for distinct in distinct_list:
            count[0] += 1
            print("Training model in " + "{:.1f}".format(count[0] / 66 * 100) + "% completed...")
            curve_dict = {}

            weekday = select_test_day(self.weekday)
            curve_sum = np.zeros(144)

            for day in weekday:
                curve = self.gene_gap_day(day, distinct + 1)
                curve_sum += curve
            curve_dict['weekday'] = curve_sum / len(weekday)

            sat = select_test_day(self.Sat)
            curve_sum = np.zeros(144)
            for day in sat:
                curve = self.gene_gap_day(day, distinct + 1)
                curve_sum += curve
            curve_dict['sat'] = curve_sum / len(sat)

            sun = select_test_day(self.Sun)
            curve_sum = np.zeros(144)
            for day in sun:
                curve = self.gene_gap_day(day, distinct + 1)
                curve_sum += curve
            curve_dict['sun'] = curve_sum / len(sun)

            filter_gap_list[distinct] = curve_dict

    def gene_gap_day(self, day, distinct):
        if len(day.split('-')) != 3:
            print("The input of train_gap_diff_curve_by_distinct_day should be a xx-xx-xx")
            exit(1)

        gaplist = []
        for slice in range(144):
            dateslice = day + '-' + str(slice + 1)
            gapval = self.select_gap(dateslice, distinct)
            if gapval != None:
                gaplist.append(gapval)
        coeffs = self.wa.get_wavelet_coeffs(gaplist)
        #coeffs = self.wa.coeffs_process(coeffs)
        curve = self.wa.reconstruction_from_coeffs(coeffs)
        return np.array(curve)


    #---------------gap data-------------#
    def select_gap(self,ts,distinct):
        if len(ts.split('-')) != 4:
            print("Error in format of timeslice")
        data = self.select_orderdata_by_district(ts,distinct)
        gap = (data['demand']-data['supply']).values[0]
        return gap



    #--------------diff data-----------------#
    def init_order_diff_data(self):
        print("Init order diff data...")
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
        print("Init weather data...")
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
        print("Init order data...")
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
        print("Init traffic data...")
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
    a = DataIO()