#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

IMG_ROOT_PATH="/home/caffemaker/caffe/dataset/pic_video/img"
IMG_LIST="/home/caffemaker/caffe/dataset/pic_video/img_list"

img_list = open(IMG_LIST,"w")
for root, dirs, files in os.walk(IMG_ROOT_PATH):
    for img in files:
        img_path = os.path.join(root,img)
        list_info = img_path + "\n"
        img_list.write(list_info)

