# -*- coding:utf-8 -*-
# author: Huang Zilong
# 导入npy文件，并进行每个样本同样帧数处理
import numpy as np
def same_step():
    features1 = np.load('D:\\TUT\\DCASE2018_1\\npy\\mfcc=174_2\\train_features_10.npy')
    labels1 = np.load('D:\\TUT\\DCASE2018_1\\npy\\mfcc=174_2\\train_labels_10.npy')
    features2 = np.load('D:\\TUT\\DCASE2018_1\\npy\\mfcc=174_2\\val_features_10.npy')
    labels2 = np.load('D:\\TUT\\DCASE2018_1\\npy\\mfcc=174_2\\val_labels_10.npy')
    print(features1.shape)
    print(features2.shape)
    # 训练集和测试集的特征不足max_len的填充0,使所有的输入形式为[216,174]
    wav_max_len = max([len(feature) for feature in features2])
    print("max_len:", wav_max_len)
    # wav_max_len=432
    train_features = []
    for mfccs in features1:
        while len(mfccs) < 216:  # 只要小于wav_max_len就补n_inputs个0
            mfccs.append([0] * 174)
        train_features.append(mfccs)
    train_features = np.array(train_features)
    test_features = []
    for mfccs in features2:
        while len(mfccs) < 216:  # 只要小于wav_max_len就补n_inputs个0
            mfccs.append([0] * 174)
        test_features.append(mfccs)
    test_features = np.array(test_features)
    # 训练集、验证集、测试集
    train_x = train_features
    train_y = labels1
    test_x = test_features
    test_y = labels2
    print(train_x.shape)
    print(test_x.shape)
    print('DataSet successfully import! ')

    # # 训练集、验证集、测试集的类别分布情况
    # print('the train_set samples is:', len(train_y))
    # print('the val_set samples is:', len(val_y))
    # print('the test_set samples is:', len(test_y))
    # print("the train_set classes is:\n", counter_y.counter_y(train_y))
    # print("the val_set classes is:\n", counter_y.counter_y(val_y))
    # print("the test_set classes is:\n", counter_y.counter_y(test_y))

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
     same_step()