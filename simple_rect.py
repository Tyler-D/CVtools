#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os

PROCESS_LIST_PATH = "/home/caffemaker/caffe/dataset/rect/img.list"
TARGET_IMG_PATH = "/home/caffemaker/caffe/dataset/rect/rect_results"

file_list = open(PROCESS_LIST_PATH, "r")
for line in file_list.readlines():
    line_value = line.split()
    img_path = line_value[0]
    img_name = img_path.split('/')[-1]
    # anno_path = line.split()[1]
    img = cv2.imread(img_path)
    # anno_list = open(anno_path, "r")
    # for anno_str in anno_list.readlines():
        # anno_value = anno_str.split()
    cv2.rectangle(img, (int(float(line_value[3])),int(float(line_value[4]))),
                      (int(float(line_value[5])),int(float(line_value[6]))),
                      (255,0,0), 2)
    cv2.putText(img, str(line_value[2]),(int(float(line_value[3])), int(float(line_value[4]))), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0),1)
    new_img_path = os.path.join(TARGET_IMG_PATH,img_name)
    cv2.imwrite(img_path,img)
    cv2.imwrite(new_img_path,img)

