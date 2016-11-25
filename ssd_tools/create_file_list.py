#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

IMG_ROOT_PATH="/home/caffemaker/caffe/dataset/pic_video/clean_img"
ANNO_ROOT_PATH="/home/caffemaker/caffe/dataset/pic_video/anno"

val_list=open("/home/caffemaker/caffe/dataset/pic_video/val.txt","w")
train_list=open("/home/caffemaker/caffe/dataset/pic_video/train.txt","w")

val_num = 300
cnt = 0
for root,dirs,files in os.walk(IMG_ROOT_PATH):
    for file in files:
        img_name = file.split(".")[0]
        img_path = os.path.join(IMG_ROOT_PATH,file)
        anno_path = os.path.join(ANNO_ROOT_PATH,img_name+".txt")
        info_str = img_path + " "+anno_path+"\n"

        if (cnt < val_num):
            info_file = val_list
        else :
            info_file = train_list
        info_file.write(info_str)
        cnt += 1


