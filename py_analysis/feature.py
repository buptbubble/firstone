from data_io import DataIO
from tools import *
import math

class cFeature:
    dataio = DataIO()
    datelice = ''
    date = ''
    distinct = 0
    daytype = ''
    back_len = 3
    def generate(self,ds,distinct):
        self.date = ds[0:10]
        self.datelice = ds
        self.distinct = distinct
        self.daytype = isWeekendsText(self.date)
        slice = int(ds.split('-')[-1])
        if slice <=self.back_len:

            return None,None

        #--------------------feature generate----------------------#
        f = []

        # wea_feature = self.weather_feature()
        # if wea_feature != None:
        #     f.extend(wea_feature)
        # else:
        #     return None, None

        gap_feature = self.gap_feature()
        if gap_feature == None:
            return None,None
        f.extend(gap_feature)
        f.append(1)
        gap = self.dataio.select_gap(self.datelice,self.distinct)
        return f,gap

    def weather_feature(self):
        weather = self.dataio.select_weatherdata_by_dateslice(self.datelice)
        if type(weather) == type(None):
            return None
        wea_feature= [0] * 4
        cur_weather = int(weather['weather'])
        if cur_weather == 2 or cur_weather == 3 or cur_weather == 4:
            wea_feature[0] = 1
        elif cur_weather == 8:
            wea_feature[1] = 1
        elif cur_weather == 9:
            wea_feature[2] = 1
        else:
            wea_feature[3] = 1
        return wea_feature

    def gap_feature(self):
        gapfeature = []

        ls = get_last_ts(self.datelice)
        gap_b1 = self.dataio.select_gap(ls,self.distinct)
        ls = get_last_ts(ls)
        gap_b2 = self.dataio.select_gap(ls,self.distinct)
        ls = get_last_ts(ls)
        gap_b3 = self.dataio.select_gap(ls,self.distinct)

        gap_diff_b1 = gap_b1 - gap_b2
        gap_diff_b2 = gap_b2 - gap_b3

        if gap_b2 != 0:
            gapfeature.append(gap_diff_b1/gap_b2)
        else:
            gapfeature.append(5)

        gapfeature.append(gap_b1)
        gapfeature.append(gap_diff_b1)
        gapfeature.append(gap_diff_b2)


        #ls = self.datelice
        # for i in range(self.back_len):
        #     gap_filtered = self.dataio.select_filter_gap(ls,self.distinct,self.daytype)
        #     #print(ls,self.daytype)
        #     gapfeature.append(gap_filtered)
        #     ls = get_last_ts(ls)
        gap_filtered_b2 = self.dataio.select_filter_gap(get_last_ts(get_last_ts(self.datelice)),self.distinct,self.daytype)
        gap_filtered_b1 = self.dataio.select_filter_gap(get_last_ts(self.datelice),self.distinct,self.daytype)
        gap_filtered_cur = self.dataio.select_filter_gap(self.datelice,self.distinct,self.daytype)
        gap_filtered_a1 = self.dataio.select_filter_gap(get_next_ts(self.datelice),self.distinct,self.daytype)
        if gap_filtered_a1 == None or gap_filtered_b1 == None or gap_filtered_b2 == None:
            return None

        gap_filter_diff_b2 = gap_filtered_b1 - gap_filtered_b2
        gap_filter_diff_b1 = gap_filtered_cur-gap_filtered_b1
        gap_filter_diff_a1 = gap_filtered_a1-gap_filtered_cur
        #gapfeature.append(gap_filter_diff_b2)
        gapfeature.append(gap_filter_diff_b1)


        #gapfeature.append(math.pow(gap_filter_diff_b1,3))
        #gapfeature.append(math.pow(gap_filter_diff_b1,2))
        #gapfeature.append(gap_filter_diff_a1)

        return gapfeature

