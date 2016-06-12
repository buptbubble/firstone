import math

def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def select_test_day(daylist):
    daytest = []
    for i in daylist:
        day = '{:02d}'.format(i)
        prefix = '2016-01-'
        date = prefix + day
        daytest.append(date)
    # print("Day under testing...")
    # for day in daytest:
    #     print("\t"+day)
    return daytest


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

def isWeekendsText(date):
    day = int(date.split('-')[-1])
    if day == 1:
        return 1
    else:
        if (day - 1) % 7 == 1:
            return 'sat'
        if (day - 1) % 7 == 2:
            return 'sun'
        else:
            return 'weekday'



def get_yesterday(date):
    if len(date.split('-'))!=3:
        return None
    else:
        prefix = date[0:8]
        yesterday = int(date.split('-')[-1])-1
        yesterday = prefix+"{:02}".format(yesterday)
        return yesterday

def get_last_ts(ts):
    if len(ts.split('-')) != 4:
        return None
    else:
        prefix = ts[0:11]
        slice = int(ts.split('-')[-1])-1
        ts_last = prefix+str(slice)
        return  ts_last

def get_next_ts(ts):
    if len(ts.split('-')) != 4:
        return None
    else:
        prefix = ts[0:11]
        slice = int(ts.split('-')[-1]) + 1
        ts_last = prefix + str(slice)
        return ts_last


def gene_timeslice_feature(value,step=6,length=144):
    count_sum = math.floor(length/step)+1
    pos = math.floor(value/step)
    f = [0]*count_sum
    f[pos] = 1
    return f

if __name__ == '__main__':

    ts = '2016-01-01-100'
    print(get_last_ts(ts))




