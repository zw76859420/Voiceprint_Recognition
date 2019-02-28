# -*- coding:utf-8 -*-
# author: Huang Zilong
# 将wav文件随机打乱，并打标签
import numpy as np
from DCASE2018_1 import read_file

def file_path_shuffle(feature, label):
    train_f, train_l = np.array(feature), np.array(label)
    np.random.seed(1)
    shuffle_indices = np.random.permutation(np.arange(len(train_f)))
    train_f = train_f[shuffle_indices]
    train_l = train_l[shuffle_indices]
    train_file, train_label = list(train_f), list(train_l)
    for i in range(len(train_label)):
        if train_label[i] == 'airport\n':
            train_label[i] = 0
        if train_label[i] == 'shopping_mall\n':
            train_label[i] = 1
        if train_label[i] == 'metro_station\n':
            train_label[i] = 2
        if train_label[i] == 'street_pedestrian\n':
            train_label[i] = 3
        if train_label[i] == 'public_square\n':
            train_label[i] = 4
        if train_label[i] == 'street_traffic\n':
            train_label[i] = 5
        if train_label[i] == 'tram\n':
            train_label[i] = 6
        if train_label[i] == 'bus\n':
            train_label[i] = 7
        if train_label[i] == 'metro\n':
            train_label[i] = 8
        if train_label[i] == 'park\n':
            train_label[i] = 9
    return train_file, train_label

# if __name__ == '__main__':
#     a= 'D:\\007DataSet\\TUT\\DCASE2018\\fold1_train.txt'
#     b= 'D:\\007DataSet\\TUT\\DCASE2018\\fold1_evaluate.txt'
#     train_x, train_y = read_file.read_file(a)
#     train_x, train_y = file_path_shuffle(train_x,train_y)
#     val_x, val_y = read_file.read_file(b)
#     val_x, val_y = file_path_shuffle(val_x,val_y)
#     print(len(train_x))
#     print(len(train_y))
#     print(train_x[0:5])
#     print(train_y[0:5])
#     print(len(val_x))
#     print(len(val_y))
#     print(val_x[0:5])
#     print(val_y[0:5])