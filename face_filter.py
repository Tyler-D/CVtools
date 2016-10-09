#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

val_list_file = open("/home/caffemaker/caffe/dataset/flickr/complete_face_list.txt", "r")
new_val_list_file = open("/home/caffemaker/caffe/dataset/flickr/1or2_img_list.txt", "w")
# train_list_file = open("/home/caffemaker/caffe/dataset/flickr/pyramid_img_list/train.txt","r")
# new_train_list_file = open("/home/caffemaker/caffe/dataset/flickr/train.txt", "w")
# faces_list=open("/home/caffemaker/caffe/dataset/flickr/faces_list.txt","w")


val_list = val_list_file.readlines()
# train_list = train_list_file.readlines()
img_id = 0
#random pick val list
orig_val_num = len(val_list)
for i in xrange(0, orig_val_num):
    anno_path = val_list[img_id].split()[1]
    anno_file = open(anno_path,"r")
    if (len(anno_file.readlines())>2):
        anno_file.close()
        # faces_list.write(val_list[img_id])
        img_id += 1
        continue
    anno_file.close()
    # mask = random.randint(0,3)
    new_val_list_file.write(val_list[img_id])
    img_id += 1

#random pick train list
# img_id = 0
# orig_train_num = len(train_list)/4
# for i in xrange(0, orig_train_num):
    # anno_path = train_list[img_id].split()[1]
    # anno_file = open(anno_path,"r")
    # if (len(anno_file.readlines())>2):
        # anno_file.close()
        # img_id += 4
        # continue
    # anno_file.close()
    # mask = random.randint(0,3)
    # new_train_list_file.write(train_list[img_id+mask])
    # img_id += 4
