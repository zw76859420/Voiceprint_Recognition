程序步骤：  
1、read_file.py                读取txt文件，获取文件名
2、file_shuffle                  将训练集中文件随机打乱顺序
3、feature_extract.py      提取mfcc特征保存为npy文件
4、same_step.py                  导入npy文件，统一样本帧数
5、DenseNet.py                   训练模型并保存为h5文件
训练好的模型中验证集准确率有：63.90%、64.22%