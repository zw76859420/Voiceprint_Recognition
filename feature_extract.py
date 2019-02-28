# -*- coding:utf-8 -*-
# author: Huang Zilong
# 提取mfcc特征，保存为npy格式
import numpy as  np
import librosa
from tqdm import tqdm
from DCASE2018_1 import read_file
from DCASE2018_1 import file_shuffle

a ,b = read_file.read_file('D:\\007DataSet\\TUT\\DCASE2018\\fold1_train.txt')
train_feature, train_label = file_shuffle.file_path_shuffle(a,b)
c, d = read_file.read_file('D:\\007DataSet\\TUT\\DCASE2018\\fold1_evaluate.txt')
val_feature, val_label = file_shuffle.file_path_shuffle(c,d)
# print(train_feature[0:10])
# print(train_label[0:10])
# print(val_feature[0:10])
# print(val_label[0:10])

# 使用librosa提取mfcc特征
def feature_extract(wav_files):
    features = []
    # 提取mfcc特征
    for wav_file in tqdm(wav_files):
        y, fs = librosa.load(wav_file)
        mfccs = np.transpose(librosa.feature.mfcc(y = y, sr = 22050 , n_mels = 128 , n_mfcc= 128, hop_length=512 ),[1,0])
        mfccs1 = np.transpose(librosa.feature.mfcc(y=y, sr=fs, n_mels=128, n_mfcc=23), [1, 0])
        mfcc_delta = librosa.feature.delta(mfccs1, order=1)
        mfcc_delta2 = librosa.feature.delta(mfccs1, order=2)
        mfccs = np.hstack((mfccs, mfcc_delta, mfcc_delta2))
        a = []
        # 每2帧取取一次mfcc
        for i in range(len(mfccs)):
              if i % 2 == 0:
                  a.append(mfccs[i].tolist())
        b = np.array(a, dtype=np.int)
        features.append(b.tolist())
        # features.append(mfccs[0:174].tolist())
    return features

if __name__ == '__main__':
    train_features =  feature_extract(train_feature)
    val_features = feature_extract(val_feature)
    np.save('train_features_10.npy',train_features)
    np.save('train_labels_10.npy', train_label)
    np.save('val_features_10.npy', val_features)
    np.save('val_labels_10.npy',val_label)