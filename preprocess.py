# Codes to pre-process a log data file into usable data input
# Some parts (names, etc.) are excluded for confidentiality

import csv
import os
import numpy as np

# ##### Labels ##### #                           
labels = np.empty([19, ], dtype='U32')
# labels[x] = ''
# labels[x] = ''
# labels[x] = ''
# ##### ###### ##### #

# Min/Max                                           Outlier
min = [1.7, 1.7, 1.7, 1.7, 2.1, 2.6, 12.0, 27.0]
max = [3.3, 3.3, 3.3, 3.3, 3.9, 4.4, 16.0, 33.0]


# Delete lines with non-numerical data (message log) / insufficient columns
def del_nondata(array, start_row=0):
    del_lines = []                                  
    for x in range(len(array)-start_row, len(array)):
        if len(array[x]) != 31:
            del_lines.append(x)
    for x in reversed(range(len(del_lines))):
        array.pop(del_lines[x])
    return array


# Check for min/max boundaries
def check_minmax(array, start_ind=2):
    for x in range(8):
        ind = np.where((array[:, x+start_ind] < min[x]).astype(int) + (array[:, x+start_ind] > max[x]).astype(int) > 0)
        array[:, x+start_ind][ind] = 0
    return array


# Finding zero coordinates in data array
def findzero(array, start_ind=2):
    zeros = []
    for j in range(start_ind, start_ind+8):
        for i in range(len(array)):
            if array[i][j] == 0:
                zeros.append([i, j])

    # convert zero coordinates to express in consecutive zeroes
    zlist = []
    i = 0
    while i < len(zeros):
        j = 1
        while True:
            if (i + j >= len(zeros)):
                break
            elif (zeros[i+j][1] == zeros[i][1]) and (zeros[i+j][0] == zeros[i][0]+j):
                j += 1
            else:
                break
        zlist.append([zeros[i][0], zeros[i][1], j])
        i += j
    return zlist


# Data interpolation                                
def interpolate(zlist, array):
    for x in range(len(zlist)):
        if zlist[x][0]+zlist[x][2]+1 > len(array):  # Copy last valid data if there are zeroes at the end
            for y in range(zlist[x][2]):
                array[zlist[x][0]+y][zlist[x][1]] = array[zlist[x][0] - 1][zlist[x][1]]
        elif zlist[x][0] - 1 < 0:                   # Copy first valid data if there are zeroes at the beginning
            for y in range(zlist[x][2]):
                array[zlist[x][0]+y][zlist[x][1]] = array[zlist[x][0]+zlist[x][2]][zlist[x][1]]
        else:                                       # Interpolate (linear) using previous and next values
            for y in range(zlist[x][2]):
                array[zlist[x][0]+y][zlist[x][1]] = array[zlist[x][0]-1][zlist[x][1]] + (y+1) * (array[zlist[x][0]+zlist[x][2]][zlist[x][1]] - array[zlist[x][0] - 1][zlist[x][1]]) / (zlist[x][2]+1)
    return array


# Splits sessions                                  
def sess_split(array):
    split_ind = np.argwhere(array[:, 0] > 10000)
    split_ind = np.asarray(split_ind).reshape(len(split_ind),)
    return np.split(array, split_ind)


# Preprocessing function
def writedata(filename, foldpath):
    # Copy data, check for invalid values
    with open('data_v4/'+foldpath + '/' + filename, 'r') as csv_file:
        temp = list(csv.reader(csv_file))
    csv_file.close()

    temp.pop(0)                                     # Delete labels
    temp = del_nondata(temp)                        # Delete nonvalid data lines

    temp = np.asarray(temp)                         # Convert to numpy
    data = temp[:, 1:].astype(np.float32)

    procdata = np.empty([data.shape[0], 19], dtype=np.float32)

    # ##### copy data ##### #                       # Copy only the required data
    # dimensions
    procdata[:, 2:8] = data[:, 1:7]
    procdata[:, 8:10] = data[:, 9:11]
    # control data
    procdata[:, 10:12] = data[:, 18:20]
    procdata[:, 12:16] = data[:, 21:25]
    procdata[:, 16:] = data[:, 26:29]
    # ##### ######### ##### #

    # calc time differences
    times = np.empty([len(temp), ])
    times[0] = 0
    for x in range(1, len(times)):
        cur = 60 * float(temp[x][0].split(' ')[1].split(':')[1]) + \
                         float(temp[x][0].split(' ')[1].split(':')[2])
        prev = 60 * float(temp[x-1][0].split(' ')[1].split(':')[1]) + \
                         float(temp[x-1][0].split(' ')[1].split(':')[2])
        if (cur-prev) < 0:
            times[x] = "%.0f" % (1000 * (cur-prev+(60*60)))
        else:
            times[x] = "%.0f" % (1000*(cur-prev))
    # write into processed data
    procdata[:, 0] = times

    # Check for min/max boundaries
    procdata = check_minmax(procdata)  

    # Check redundant time data
    del_ind = []
    for x in reversed(range(1, len(procdata))):
        if procdata[x][0] == 0:
            for y in range(2, 10):
                if (procdata[x][y] != 0) and (procdata[x-1][y] != 0):
                    procdata[x-1][y] = (procdata[x][y] + procdata[x-1][y]) / 2
                elif procdata[x-1][y] == 0:
                    procdata[x-1][y] = procdata[x][y]
            del_ind.append(x)
    procdata = np.delete(procdata, del_ind, 0)

    # ##### ######### ##### #                     
    # ##### ######### ##### #
    zeros = findzero(procdata)

    for x in reversed(range(len(zeros))):
        if zeros[x][2] <= 5:
            zeros.pop(x)

    del_ind = []
    for x in range(len(zeros)):
        for y in range(zeros[x][2]):
            d = zeros[x][0] + y
            try:
                del_ind.index(d)
            except:
                del_ind.append(d)

    del_ind.sort()

    i = 0
    while i < len(del_ind):
        j = 1
        add = 0
        while True:
            if (i + j >= len(del_ind)):
                break
            elif (del_ind[i+j] == del_ind[i]+j):
                add += procdata[del_ind[i + j]][0]
                j += 1
            else:
                break
        if (i+j >= len(del_ind)):
            break
        procdata[del_ind[i+j]][0] += add
        i += j

    procdata = np.delete(procdata, del_ind, 0)
    # ##### ######### ##### #
    # ##### ######### ##### #

    # Discard short data sessions                  
    if len(procdata) < 4000:
        return "Data below 4000 entries"

    sess_list = sess_split(procdata)             

    # Interpolate and Write Data              
    for ind, sess in enumerate(sess_list):
        if len(sess) < 4000:
            continue
        zlist = findzero(sess)
        sess = interpolate(zlist, sess)
        sess[0][0] = 0
        # Assign new time data with 250ms
        sess[:, 1] = 250 * np.arange(len(sess))
        data = sess.astype(str)
        data = np.insert(data, 0, labels, 0)
        pr_file = 'processed/' + foldpath + '/' + filename.split('.')[0] + '-' + str(ind) + '.csv'
        with open(pr_file, 'w', newline='') as newfile:
            writer = csv.writer(newfile)
            writer.writerows(data)
        newfile.close()
        print(pr_file, 'file write success')

    return "File processed"
