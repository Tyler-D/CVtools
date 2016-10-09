#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

val_list_file = open("/home/caffemaker/caffe/dataset/flickr/pyramid_list/val.txt", "r")
new_val_list_file = open("/home/caffemaker/caffe/dataset/flickr/val.txt", "w")
train_list_file = open("/home/caffemaker/caffe/dataset/flickr/pyramid_list/train.txt","r")
new_train_list_file = open("/home/caffemaker/caffe/dataset/flickr/train.txt", "w")


val_list = val_list_file.readlines()
train_list = train_list_file.readlines()
img_id = 0
#random pick val list
orig_val_num = len(val_list)/4
for i in xrange(0, orig_val_num):
    mask = random.randint(1,3)
    new_val_list_file.write(val_list[img_id+mask])
    img_id += 4

#random pick train list
img_id = 0
orig_train_num = len(train_list)/4
for i in xrange(0, orig_train_num):
    mask = random.randint(1,3)
    new_train_list_file.write(train_list[img_id+mask])
    img_id += 4
