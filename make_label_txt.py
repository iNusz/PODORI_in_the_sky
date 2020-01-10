import os

img_path = '/home/pirl/PycharmProjects/cnnTest/FrameImg/'
f1 = open('./VDatalist/trainlist.txt','r')
f2 = open('./VDatalist/testlist.txt','r')

train_list = f1.readlines()
test_list = f2.readlines()

f3 = open('newTrainlist.txt', 'w')
f4 = open('newTestlist.txt', 'w')

clip_length = 16

for line in train_list:
    name = line.split(' ')[0]
    image_path = img_path+name
    label = line.split(' ')[-1]
    images = os.listdir(image_path)
    nb = len(images) // clip_length
    for i in range(nb):
        f3.write(name+' '+ str(i*clip_length+1)+' '+label)


for line in test_list:
    name = line.split(' ')[0]
    image_path = img_path+name
    label = line.split(' ')[-1]
    images = os.listdir(image_path)
    nb = len(images) // clip_length
    for i in range(nb):
        f4.write(name+' '+ str(i*clip_length+1)+' '+label)

f1.close()
f2.close()
f3.close()
f4.close()
