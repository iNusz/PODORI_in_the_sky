# c3d-keras
train c3d with keras

## requirements

opencv-3.2

keras-2.0.8

tensorflow-1.3

------------------update video test demo------------------------

    python video_demo.py
    
![](https://github.com/TianzhongSong/C3D-keras/blob/master/videos/out.gif)

---------------------------------------------------------------
A simple reproduce of C3D (https://github.com/facebook/C3D).

Origin paper:https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html

## Files:

### video2img.py   

Convert videos to images for UCF-101 dataset (http://crcv.ucf.edu/data/UCF101.php).

### make_label_txt.py  

Generate label texts.

### models.py   

Define the C3D model.

### train_c3d.py   

Train the C3D model, the init learning rate was set to 0.005, and it divided by 10 after 4,8,
12 epoch as same as in the origin perper. Video clips are resized to 128x171x16, then flip the data and apply a center crop
(112x112x16)on clips.

## Results：

### train and val acc curve during training
the final val acc of origin paper is about 44% in Figure 2, where in my implementation is 42.96%, the batch size was set to 16 , I only have one GTX1080.

![image](https://github.com/TianzhongSong/c3d-keras/blob/master/results/model_accuracy.png)

### train and val loss during training.

![image](https://github.com/TianzhongSong/c3d-keras/blob/master/results/model_loss.png)

The trained weight file can be downloaded from (https://pan.baidu.com/s/1gfNEaCr)
