#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import random

def classAwareBalance(root_path, file_list_path, max_sample_num=0):
    '''
    create class aware shuffling list.
    Parameters:
    ----------
    root_path: str. path to image root
    file_list_path: str. target file list path

    Return:
    -------
    '''

    cls_dict = {}
    cls_label = {}
    label_file = open(os.path.join(file_list_path, "label.txt"),"w")
    for idx, cls in enumerate(os.listdir(root_path)):
        cls_dict[cls] = len(os.listdir(os.path.join(root_path, cls)))
        cls_label[cls] = idx
        label_file.write(cls + " " + str(idx)+"\n")


    print cls_dict
    print cls_label
    if max_sample_num == 0:
        max_sample_num = max(cls_dict.values())
    print max_sample_num
    max_sample_num = 300

    cls_sample_list = {}
    for cls in cls_dict:
        cls_sample_list[cls] = random.sample(range(max_sample_num), max_sample_num)

    final_list = []
    for cls in cls_dict:
        ord_file_list = os.listdir(os.path.join(root_path, cls))
        single_cls_list = []
        for idx in cls_sample_list[cls]:
            single_cls_list.append(os.path.join(root_path, cls, ord_file_list[idx % cls_dict[cls]])+" "+ str(cls_label[cls]))
        final_list += single_cls_list

    random.shuffle(final_list)
    target_file = open(os.path.join(file_list_path, "file_list.txt"), "w")
    for file in final_list:
        target_file.write(file+"\n")


# classAwareBalance("/home/caffemaker/kaggle/data/train_crop/", "./")

