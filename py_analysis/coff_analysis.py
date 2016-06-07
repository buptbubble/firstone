import numpy as np

fr = open("coff_.txt",'r')
w_list = []
for line in fr:
    line = line.strip('\n').split(',')
    w = []
    #print(line)
    for index in range(9):
        #print(line[index+3])
        w.append(float(line[index+3]))
    w_list.append(w)

w_list = np.array(w_list)

result = []
for i in range(9):
    j=i+1
    while j<9:
        diff = w_list[:, i]-w_list[:, j]
        mean = abs(diff.mean())
        var = diff.var()
        print("{} to {} mean:{} var:{}".format(i+1,j+1,mean,var))
        relation = "{} to {}".format(i+1,j+1)
        result.append([relation,mean,var])
        j+=1

sorted_item = sorted(result, key=lambda x: x[2])
for item in sorted_item:
    print(item)
