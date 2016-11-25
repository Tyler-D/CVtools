#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import Image

test_file_list = "/home/caffemaker/caffe/dataset/pic_video/val.txt"
test_name_size = "/home/caffemaker/caffe/dataset/pic_video/test_name_size.txt"
info_file = open(test_name_size,"w")
with open(test_file_list, "r") as f:
    for line in f.readlines():
        img_path = line.split()[0]
        img_name_type = (img_path.split('/')[-1])
        img_name = img_name_type.split('.')[0]

        img = Image.open(img_path)
        img_size = img.size

        info_str = img_name + ' '+ str(img_size[1]) + ' '+ str(img_size[0]) + "\n"
        info_file.write(info_str)
    f.close()
info_file.close()

