


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import Conv3D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.layers.convolutional import MaxPooling3D
from tensorflow.python.keras.layers.convolutional import ZeroPadding3D
# from tensorflow.python.keras.layers.convolutional import GlobalAveragePooling3D
from keras.layers import GlobalAveragePooling3D
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.models import load_model
import os
import random
import pickle
import cv2
from tensorflow.python.keras.layers import GlobalAveragePooling3D
import numpy as np


def process_batch(lines,img_path,train=True):
    IMG_WIDTH = 171
    IMG_HEIGHT = 128

    num = len(lines)
    batch = np.zeros((num,16,IMG_HEIGHT,IMG_WIDTH,3),dtype='float16')
    labels = np.zeros(num,dtype='int')



    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        symbol = lines[i].split(' ')[1]
        label = label.strip('\n')
        label = int(label)
        symbol = int(symbol)-1
        imgs = os.listdir(img_path+path)
        imgs.sort(key=str.lower)
        if i%1000 == 0:
            print(i)
        if train:
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            # is_flip = random.randint(0, 1)
            for j in range(16):
                img = imgs[symbol + j]
                image = cv2.imread(img_path + path + '/' + img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # openCV stores data color as BGR
                # TODO image resize
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                # if is_flip == 1:
                #     image = cv2.flip(image, 1)
                # 16 frame을 다 넣었다는 증거!!
                batch[i][j][:][:][:] = image
            labels[i] = label
        else:
            for j in range(16):
                img = imgs[symbol + j]
                image = cv2.imread(img_path + path + '/' + img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                batch[i][j][:][:][:] = image
            labels[i] = label
    return batch, labels

train_file = '/home/pirl/Documents/c3dTest/newTrainlist.txt'
# test_file = 'newTestlist.txt'
f1 = open(train_file, 'r')
# f2 = open(test_file, 'r')
lines = f1.readlines()
f1.close()


batch, labels = process_batch(lines,'/home/pirl/PycharmProjects/cnnTest/FrameImg/',train=True)

batch = np.transpose(batch, (0,2,3,1,4))

one_label=[]
for i in range(len(labels)):
    if i == 0:
        one_label.append([1,0])
    else:
        one_label.append([0,1])
one_label=np.array(one_label)


# image size = 160 x 320
def posla_net():
    # model setting
    H = 128
    W = 171
    D = 16
    CH = 3

    inputShape = (H, W, D, CH)
    input_shape = (128, 171, 16, 3)
    activation = 'relu'
    keep_prob_conv = 0.25
    keep_prob_dense = 0.5

    # init = 'glorot_normal'
    # init = 'he_normal'
    init = 'he_uniform'
    chanDim = -1
    classes = 1

    # First
    model = Sequential()
    model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(strides=2, pool_size=(2, 2, 1), padding='same'))

    # 2nd
    model.add(Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(strides=2, pool_size=(2, 2, 2), padding='same'))

    # 3rd
    model.add(Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same'))
    model.add(Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(strides=2, pool_size=(2, 2, 2), padding='same'))

    # 4th
    model.add(Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'))
    model.add(Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(strides=2, pool_size=(2, 2, 2), padding='same'))

    # 5th
    model.add(Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'))
    model.add(Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(strides=2, pool_size=(2, 2, 2), padding='same'))

    # add zero padding
    model.add(ZeroPadding3D(padding=1))
    # additional conv. layer
    model.add(Conv3D(1024, (3, 3, 3), strides=(1, 1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(GlobalAveragePooling3D())

    model.add(Dense(2, activation='softmax'))

    return model

model = posla_net()
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001)

EPOCHS = 5
INIT_LR = 1e-4
BS = 16
split_ratio = 0.2

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])


hist = model.fit(batch, one_label,
                 epochs=EPOCHS, batch_size=BS,
                 validation_split=split_ratio,
                 verbose = 1
                 ,callbacks=[reduce_lr]
                )

model.save('./model_data/video_model_3.h5')