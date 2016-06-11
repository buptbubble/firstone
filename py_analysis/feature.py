from data_io import DataIO
from tools import *

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

        wea_feature = self.weather_feature()
        if wea_feature != None:
            f.extend(wea_feature)
        else:
            return None, None

        gap_feature = self.gap_feature()
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
        for i in range(self.back_len):
            gap_temp = self.dataio.select_gap(ls,self.distinct)
            gapfeature.append(gap_temp)
            ls = get_last_ts(ls)

        # ls = self.datelice
        # for i in range(self.back_len):
        #     gap_filtered = self.dataio.select_filter_gap(ls,self.distinct,self.daytype)
        #     #print(ls,self.daytype)
        #     gapfeature.append(gap_filtered)
        #     ls = get_last_ts(ls)

        return gapfeature

