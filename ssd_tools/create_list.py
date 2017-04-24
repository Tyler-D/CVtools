#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

FILE_LIST_PATH = "/home/caffemaker/ZTE/img_list"
TRAIN_FILE_PATH = "/home/caffemaker/ZTE/train.txt"
VAL_FILE_PATH = "/home/caffemaker/ZTE/val.txt"
train_list = open(TRAIN_FILE_PATH,"w")
val_list = open(VAL_FILE_PATH,"w")
file = open(FILE_LIST_PATH,"r")

file_list = file.readlines()

random.shuffle(file_list)

val_num = int(len(file_list)*0.1)
cnt = 0
for pair in file_list:
    if (cnt < val_num):
        info_list = val_list
    else :
        info_list = train_list
    info_str = pair
    info_list.write(info_str)
    cnt+=1


