from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from keras import initializers
from keras import optimizers
from random import shuffle
import csv
import os
import numpy as np

#
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
steps = 3


# Min, Max array for normalization
dmin = np.array([1.7, 1.7, 1.7, 1.7, 2.1, 2.6, 12.0, 25.0], dtype=float)
dmax = np.array([3.3, 3.3, 3.3, 3.3, 3.9, 4.4, 16.0, 34.0], dtype=float)


def data_preprocess(M=5):
    X = np.empty([0, steps, 3])
    Y = np.empty([0])
    it = 0
    flist = os.listdir('smooth_20step_1/')
    shuffle(flist)
    for ind, fname in enumerate(flist):
        if X.shape[0] > 30000:
            break
        with open('smooth_20step_1/'+fname, 'r') as df:
            data = np.asarray(list(csv.reader(df)))
        df.close()
        #if np.abs(data[:,12]).sum() == 0 and np.abs(data[:,14]).sum() == 0:
        #    continue
        len = data.shape[0]
        tmp = np.empty([steps, 3])
        for i in range(steps, len):
            tmp[:, [0]] = data[i-steps:i,[0]]
            tmp[:, 1] = data[i-steps:i, 12]
            tmp[:, 2] = data[i - steps:i, 14]
            X = np.insert(X, X.shape[0], tmp, axis=0)
            Y = np.insert(Y, Y.shape[0], data[[i], [0]], axis=0)
        print(fname, it, X.shape)
        it += 1
    print('preprocessing done')
    # np.save(open('train_data_20_'+str(steps)+'.dat', 'wb'), X)
    # np.save(open('test_data_20_'+str(steps)+'.dat', 'wb'), Y)
    flist.pop(ind)
    return X, Y, flist

def getdata():
    a, b, c = data_preprocess()
    # a = np.load(open('train_data_'+str(steps)+'dat', 'rb'))
    # b = np.load(open('test_data_'+str(steps)+'.dat', 'rb'))
    a[:,:,[0]] = (a[:,:,[0]]-dmin[0])/(dmax[0]-dmin[0])
    a[:,:,1:] = np.around(a[:,:,1:])
    b = (b-dmin[0])/(dmax[0]-dmin[0])
    return a, b, c

def model(M=5,N=200, fn=''):
    x_data, y_data, test = getdata()
    x_train, x_val = np.split(x_data, [int(x_data.shape[0]*8/9)])
    y_train, y_val = np.split(y_data, [int(y_data.shape[0]*8/9)])

    ind = [i for i in range(x_train.shape[0])]
    shuffle(ind)
    x_train = x_train[ind]
    y_train = y_train[ind]


    model = Sequential()
    model.add(LSTM(N, input_shape=(steps, 3), activation="sigmoid", kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05)))   #memory cell
    model.add(Dense(1, activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05)))


    early_stopping = EarlyStopping()  # 조기종료 콜백함수 정의


    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)
    model.fit(x_train, y_train, validation_data=(x_val, y_val),  epochs=20, batch_size=64, callbacks=[early_stopping])

    output = []


    for fname in test:
        with open('smooth_20step_1/'+fname, 'r') as df:
            data = np.asarray(list(csv.reader(df)))
        df.close()

        X = np.empty([0, steps, 3])
        Y = np.empty([0])

        len = data.shape[0]
        tmp = np.empty([steps, 3])
        for i in range(steps, len):
            tmp[:, [0]] = data[i-steps:i,[0]]
            tmp[:, 1] = data[i-steps:i, 12]
            tmp[:, 2] = data[i - steps:i, 14]
            X = np.insert(X, X.shape[0], tmp, axis=0)
        X[:, :, [0]] = (X[:, :, [0]] - dmin[0]) / (dmax[0] - dmin[0])
        X[:, :, 1:] = np.around(X[:, :, 1:])
        Y_pred = model.predict(X)

        for i in range(data.shape[0]):
            if i < 3:
                output.append(data[i].tolist())
            if i >=3:
                a = data[i].tolist()
                a.append(Y_pred[i-3][0]*(dmax[0]-dmin[0])+dmin[0])
                output.append(a)
        output.append('')
    print(output)

    with open('LSTM_20steps_shuff_1_'+fn+'.csv', 'w', newline='') as predfile:
        writer = csv.writer(predfile)
        writer.writerows(output)
        predfile.close()
        print('sss ')

for x in range(1, 10):
    model(fn=str(x))
