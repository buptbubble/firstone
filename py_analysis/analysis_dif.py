from data_io import DataIO
from matplotlib import pyplot as plt
import statsmodels.api as sm

class analysis_dif:
    dataio = DataIO()
    def diff_analysis(self,date,distinct):
        difflist = []
        for slice in range(143):
            ts = date+'-'+str(slice+1)
            diffdata = self.dataio.select_orderDiff_by_ds_distinct(ts,distinct)
            difflist.append(diffdata)
        #plt.plot(difflist,'ro-')
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        fig = sm.graphics.tsa.plot_acf(difflist, lags=20, ax=ax1)
        ax2 = fig.add_subplot(312)
        fig = sm.graphics.tsa.plot_pacf(difflist, lags=20, ax=ax2)
        ax3 = fig.add_subplot(313)
        ax3.plot(difflist,'ro-')
        title = date+" Distinct:"+str(distinct)
        plt.title(title)
        plt.show()


if __name__ == '__main__':
    dif_ana = analysis_dif()
    prefix = '2016-01-'
    distinct = 10
    for day in range(2,21,1):
        date = prefix+"{:02}".format(day)
        dif_ana.diff_analysis(date,distinct)