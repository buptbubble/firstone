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