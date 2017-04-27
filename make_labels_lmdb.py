#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import lmdb
import random

caffe_root = '/home/caffemaker/detection/caffe-ssd/'
sys.path.insert(0, caffe_root + 'python')
import caffe

val_cnt = 3000
labels_file = open("/mnt/ssd-data-2/cele_attribute/clean_labels", "r")
train_file = open("/home/caffemaker/ZTE/train_img_list.txt", "w")
val_file = open("/home/caffemaker/ZTE/val_img_list.txt", "w")
img_root_path = "/mnt/ssd-data-2/cele_attribute/img_align_celeba/"
# 根据多标签的位置选择从数据库、文件等中读取每幅图片的多标签，将其构造成一维的np.array类型，并追加入all_labels列表
train_labels = []
val_labels = []
# # Add your code of reading labels here ！
line_list = labels_file.readlines()
random.shuffle(line_list)
cnt = 0
for line in line_list:
    img_name = line.split()[0]
    img_path = os.path.join(img_root_path, img_name)
    wearing_hat = line.split()[1]
    eye_glass = line.split()[2]
    male = line.split()[3]

    labels = np.array([float(wearing_hat), float(eye_glass), float(male)])

    if (cnt < val_cnt):
        val_labels.append(labels)
        val_file.write(img_path + " 0" + "\n")
        cnt +=1
    else:
        train_labels.append(labels)
        train_file.write(img_path + " 0" + "\n")
# # 创建标签LMDB
key = 0
lmdb_path = "/home/caffemaker/ZTE/lmdb/train_label_lmdb"
env = lmdb.open(lmdb_path)
with env.begin(write=True) as txn:
    for labels in train_labels:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = labels.shape[0]
        datum.height = 1
        datum.width = 1
        for label in labels:
            float_data = datum.float_data.append(label)  # or .tobytes() if numpy < 1.9
        datum.label = 0
        key_str = '{:08}'.format(key)

        txn.put(key_str.encode('ascii'), datum.SerializeToString())
        key += 1

key = 0
lmdb_path = "/home/caffemaker/ZTE/lmdb/val_label_lmdb"
env = lmdb.open(lmdb_path)
with env.begin(write=True) as txn:
    for labels in val_labels:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = labels.shape[0]
        datum.height = 1
        datum.width =1
        for label in labels:
            float_data = datum.float_data.append(label)  # or .tobytes() if numpy < 1.9
        datum.label = 0
        key_str = '{:08}'.format(key)

        txn.put(key_str.encode('ascii'), datum.SerializeToString())
        key += 1
