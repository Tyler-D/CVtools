#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import caffe
import csv

caffe.set_device(1)
caffe.set_mode_gpu()

# model_def="/home/caffemaker/part-jobs/model/classify/deploy.prototxt"
# model_weights="/home/caffemaker/part-jobs/model/classify/tinyface_iter_250000.caffemodel"
model_def="/home/caffemaker/caffe/models/lcnn/LightenedCNN_B_deploy.prototxt"
model_weights="/home/caffemaker/caffe/models/lcnn/LightenedCNN_B.caffemodel"

net = caffe.Net(model_def, model_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
# transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

file_list=(open("/home/caffemaker/caffe/models/lcnn/lfw_lcnn_gray.list", "r")).readlines()
feature_list_file=open("/home/caffemaker/caffe/models/lcnn/feature_list", "w")

feature_list = []
for img in file_list:
    img = img.strip()
    print "Processing ", img
    img = caffe.io.load_image(img,color=False)
    transformed_img = transformer.preprocess("data", img);
    net.blobs['data'].data[...] = transformed_img
    output = net.forward()

    temp_buffer = ""
    for item in output['eltwise_fc1'][0]:
        temp_buffer += str(item)+","
    temp_buffer = temp_buffer[:-1]
    temp_buffer += "\n"
    feature_list_file.write(temp_buffer)


