# -*- coding:utf-8 -*-
# author: Huang Zilong
# 读取txt文件

def read_file(filename):
    data = []
    label = []
    file =[]
    with open(filename, 'r')as fr:
        lines = fr.readlines()
        for line in lines:
            res = line.split('\t')
            label.append(res[1])
            data.append(res[:-1])
    for i in range(len(data)):
        a = str('D:/007DataSet/TUT/DCASE2018/') + data[i][0]
        file.append(a)
    return file, label

# if __name__ == '__main__':
#     filename = 'D:\\007DataSet\\TUT\\DCASE2018\\fold1_train.txt'
#     a,b = read_file(filename)
#     print(a)
#     print(len(a))
#     print(b)
