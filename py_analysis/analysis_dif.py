from data_io import DataIO
from matplotlib import pyplot as plt
import statsmodels.api as sm

class analysis_dif:
    dataio = DataIO()
    def diff_analysis(self,date,distinct):
        difflist = []
        gaplist = []
        for slice in range(143):
            ts = date+'-'+str(slice+1)
            diffdata = self.dataio.select_orderDiff_by_ds_distinct(ts,distinct)
            gap = self.dataio.select_gap(ts,distinct)
            gaplist.append(float(gap))
            difflist.append(float(diffdata))
            #print(type(diffdata))
        #plt.plot(difflist,'ro-')
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        fig = sm.graphics.tsa.plot_acf(gaplist, lags=20, ax=ax1)
        ax2 = fig.add_subplot(312)
        fig = sm.graphics.tsa.plot_pacf(gaplist, lags=20, ax=ax2)
        ax3 = fig.add_subplot(313)
        ax3.plot(difflist,'ro-')
        title = date+" Distinct:"+str(distinct)
        plt.title(title)


        #arma_11 = sm.tsa.ARMA(difflist,(1,1)).fit()
        #arma_02 = sm.tsa.ARMA(difflist,(0,2)).fit()
        #arma_01 = sm.tsa.ARMA(gaplist,(1,0)).fit()

        arima = sm.tsa.ARIMA(gaplist,(1,1,0)).fit()

        fig1 = plt.figure(1)
        fig1 = arima.plot_predict()

        plt.show()


if __name__ == '__main__':
    dif_ana = analysis_dif()
    prefix = '2016-01-'
    distinct = 8
    for day in range(2,21,1):
        date = prefix+"{:02}".format(day)
        dif_ana.diff_analysis(date,distinct)