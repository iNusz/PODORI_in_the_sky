# -*- coding:utf-8 -*-
import tensorflow as tf
from keras.models import Model
from models import c3d_model
from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation, GlobalAveragePooling3D, ZeroPadding3D
import numpy as np
import cv2
import h5py
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.optimizers import SGD,Adam
from keras.models import model_from_json



'''


#   이미 훈련된 모델 weight를 불러와서 드론에서 받아온 영상으로 테스트

1. input = (128,171,16,3) numpy array
2. output = model test result

'''
#
# def video2batch(filePath):
#     videoIn = cv2.VideoCapture(filePath)
#
#     IMG_WIDTH = 171
#     IMG_HEIGHT = 128
#     TOTAL_FRAME = int(videoIn.get(cv2.CAP_PROP_FRAME_COUNT))  # number of frames in given video
#
#     batches = []
#     frames = []
#     while videoIn.isOpened():
#         if len(frames) < 16:
#             ret, frame = videoIn.read()
#             try:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
#                 frames.append(frame)
#             except:
#                 pass
#         else:
#             batches.append(frames)
#             frames == []
#
#         if len(batches) == TOTAL_FRAME // 16:
#             break
#
#     batches = np.array(batches).astype(np.float32)
#     # batches[..., 0] -= 99.9
#     # batches[..., 1] -= 92.1
#     # batches[..., 2] -= 82.6
#     # batches[..., 0] /= 65.8
#     # batches[..., 1] /= 62.3
#     # batches[..., 2] /= 60.3
#     batches = np.transpose(batches, (0, 2, 3, 1, 4))
#     # print(batches.shape)
#     return batches

def videoSeg(filePath):
    videoIn = cv2.VideoCapture(filePath)

    IMG_WIDTH = 171
    IMG_HEIGHT = 128
    TOTAL_FRAME = int(videoIn.get(cv2.CAP_PROP_FRAME_COUNT))  # number of frames in given video

    # batch = np.zeros((TOTAL_FRAME, IMG_HEIGHT, IMG_WIDTH, 3), dtype='float32')
    batch =[]
    idx = 0

    while videoIn.isOpened():
        ret, frame = videoIn.read()
        curFrame = int(videoIn.get(cv2.CAP_PROP_POS_FRAMES))  # current frame number

        if (curFrame == TOTAL_FRAME ):
            break

        if frame is None:
            break

        # resize (128, 171). resize함수는 변수 순서가(넓이, 높이)로 정의되어있음.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    #     print(np.shape(frame))


        # batch[..., 0] -= 99.9
        # batch[..., 1] -= 92.1
        # batch[..., 2] -= 82.6
        # batch[..., 0] /= 65.8
        # batch[..., 1] /= 62.3
        # batch[..., 2] /= 60.3
        # batch /= 255.0
        if idx <= TOTAL_FRAME:
            batch.append(frame)
        idx += 1

    videoIn.release()

    #     print(np.shape(batch))

    batch = np.array(batch)
    res = np.array(batch[0:(TOTAL_FRAME // 16) * 16]).reshape(-1, 16, 128, 171, 3)



    res = np.moveaxis(res, 1, 3)
    #     print(np.shape(res))

    return res



def modelPredict(weightPath, modelPath):
    json_file = open(modelPath, 'r')
    model = json_file.read()
    json_file.close()
    model = model_from_json(model)
    model.summary()

    #   model load
    model.load_weights(weightPath, by_name=True)

    #   compile
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    # normal: walking
    # res = videoSeg('/home/pirl/Downloads/walking(1).mp4')
    # violence: fight

    res = videoSeg('/home/pirl/Downloads/fight(4).mp4')

    # for i in range(1, 10):
    #     res = videoSeg('/home/pirl/PycharmProjects/cnnTest/Real Life Violence Dataset/Violence/V_'+str(i)+'.mp4')
    #     print(model.predict(res, verbose=1, batch_size=16))
    #
    # for i in range(11, 20):
    #     res = videoSeg('/home/pirl/PycharmProjects/cnnTest/Real Life Violence Dataset/NonViolence/NV_'+str(i)+'.mp4')

    pred = model.predict(res, verbose=1, batch_size=16)
    for i in range(len(pred)):
        if pred[i][0] < pred[i][1]:
            print("violence detected!")
        else:
            print("Normal")


modelPredict(weightPath='/home/pirl/PycharmProjects/cnnTest/FinalWeightJson/weights_c3d(lr=0.0001,ADAM,binary,epoch10).h5',
             modelPath='/home/pirl/PycharmProjects/cnnTest/FinalWeightJson/model(lr=0.0001,ADAM,binary,epoch10).json'
             )

