# -*- coding:utf-8 -*-
# author: Huang Zilong
# 训练模型

import time
start = time.time()
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense , Dropout , Input  , Flatten
from keras.layers import Conv2D , MaxPooling2D  , Activation , regularizers , concatenate,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD , Adadelta , Adam
from keras.utils import to_categorical
from DCASE2018_1 import same_step
from DCASE2016 import confusion_matrix

# 导入数据集，划分训练集、验证集、测试集
train_x, train_y,test_x, test_y = same_step.same_step()
# 将特征转换成三维数据,将标签转为需要的格式（one-hot 编码）
train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)
test_x = test_x.reshape(test_x.shape[0],test_x.shape[1],test_x.shape[2],1)
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

def dense_block(input_tensor, channels):
    conv1 = Conv2D(filters=3 * channels, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(input_tensor)
    bn1 = BatchNormalization()(conv1)
    relu = Activation(activation='relu')(bn1)
    conv2 = Conv2D(filters=1 * channels, kernel_size=[3, 3], padding='same', kernel_initializer='he_normal')(relu)
    bn2 = BatchNormalization()(conv2)
    relu2 = Activation(activation='relu')(bn2)
    return relu2

def transition_layer(input_tensor , channels):
    conv = Conv2D(filters=channels , kernel_size=[1, 1])(input_tensor)
    pool = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(conv)
    return pool


def creat_model():
        k = 24
        input_data = Input(shape=[216, 174, 1] , name='Input')
        conv1 = Conv2D(filters=k * 1 , kernel_size=[5 , 5] )(input_data)
        # pool1 = MaxPooling2D(pool_size=[2 , 2] , strides=[2 , 1])(conv1)
        # pool1 =Dropout(0.1)(pool1)
        # conv2 = Conv2D(filters=k * 2, kernel_size=[4, 1])(pool1)
        # pool2 = MaxPooling2D(pool_size=[2, 2], strides=[2, 1])(conv2)
        # pool2 = Dropout(0.1)(pool2)
        x = MaxPooling2D(pool_size=[3, 3], strides=[3, 3])(conv1)

        b1_1 = dense_block(x, k)
        b1_1_conc = concatenate([x, b1_1], axis=-1)
        b1_1_conc = Dropout(0.1)(b1_1_conc)
        b1_2 = dense_block(b1_1_conc, k,)
        b1_2_conc = concatenate([x, b1_1, b1_2], axis=-1)
        # b1_2_conc = Dropout(0.1)(b1_2_conc)
        # b1_3 = dense_block(b1_2_conc, k)
        # b1_3_conc = concatenate([x, b1_1, b1_2, b1_3], axis=-1)
        # b1_4 = dense_block(b1_3_conc, k)
        # b1_4_conc = concatenate([x, b1_1, b1_2, b1_3, b1_4], axis=-1)
        # b1_5 = dense_block(b1_4_conc, k)
        # b1_5_conc = concatenate([b1_1, b1_2, b1_3, b1_4, b1_5], axis=-1)
        transion_1 = transition_layer(b1_2_conc, k)

        b2_1 = dense_block(transion_1, k)
        b2_1_conc = concatenate([transion_1, b2_1], axis=-1)
        b2_1_conc = Dropout(0.1)(b2_1_conc)
        b2_2 = dense_block(b2_1_conc, k)
        b2_2_conc = concatenate([transion_1, b2_1, b2_2], axis=-1)
        # b2_3 = dense_block(b2_2_conc, k)
        # b2_3_conc = concatenate([transion_1, b2_1, b2_2, b2_3], axis=-1)
        # b2_4 = dense_block(b2_3_conc, k)
        # b2_4_conc = concatenate([transion_1, b2_1, b2_2, b2_3, b2_4], axis=-1)
        # b2_5 = dense_block(b2_4_conc, k)
        # b2_5_conc = concatenate([b2_1, b2_2, b2_3, b2_4, b2_5], axis=-1)
        transion_2 = transition_layer(b2_2_conc, k)

        b3_1 = dense_block(transion_2, k)
        b3_1_conc = concatenate([transion_2, b3_1], axis=-1)
        b3_1_conc = Dropout(0.1)(b3_1_conc)
        b3_2 = dense_block(b3_1_conc, k)
        b3_2_conc = concatenate([transion_2, b3_1, b3_2], axis=-1)
        # b3_3 = dense_block(b3_2_conc, k)
        # b3_3_conc = concatenate([transion_2, b3_1, b3_2, b3_3], axis=-1)
        # # # b3_4 = dense_block(b3_3_conc, k)
        # # # b3_4_conc = concatenate([transion_2, b2_1, b2_2, b2_3, b2_4], axis=-1)
        # # # b3_5 = dense_block(b3_4_conc, k)
        # # # b3_5_conc = concatenate([b3_1, b3_2, b3_3, b3_4, b3_5], axis=-1)
        transion_3 = transition_layer(b3_2_conc, k)


        dense = Flatten()(transion_3)
        dense2 = Dense(units=512 , use_bias=True , kernel_initializer='he_normal' ,
                        activation='relu' , kernel_regularizer=regularizers.l2(1e-7))(dense)
        dense3 = Dropout(0.6)(dense2)
        dense4 = Dense(units=10)(dense3)
        prediction = Activation(activation='softmax')(dense4)


        model = Model(inputs=input_data , outputs=prediction)
        model.summary(line_length=150)
        # plot_model(model , 'DenseNet frame.png' , show_shapes=True)
        from keras.callbacks import ModelCheckpoint
        filepath = "D:\\TUT\\DCASE2018_1\\Model\\weights-improvement-{epoch:02d}-{val_acc:.4f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x=train_x, y=train_y, batch_size=32, epochs=50, validation_data=(test_x, test_y), callbacks=[checkpoint])
        # loss, evaluate = model.evaluate(x=test_x, y=test_y)
        # print('Test Accuracy : ', evaluate)
        # model.save('DenseNet_model1.h5')
        # loss_acc.show_history(history)
        # y = model.predict(x=test_x)
        # confusion_matrix.confusion_matrix_1(y, test_y)

if __name__ == '__main__':
    creat_model()
    # 计算程序运行时间
    end = time.time()
    print("Running time(min)：", (end - start) // 60)

