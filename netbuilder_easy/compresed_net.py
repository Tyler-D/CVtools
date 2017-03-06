#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from common_module_libs import *
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def compVGG16(net, from_layer):
    '''
     Compressed VGG16 Network Using the method in "DeepRebirth"
     reconstruct the conv-pooling stacks to strided conv
    '''
    norm_kwargs = {
                   'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                   'weight_filler': dict(type='xavier'),
                   'bias_filler': dict(type='constant', value=0)}
    merge_kwargs ={
                   'param': [dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
                   'weight_filler': dict(type='xavier'),
                   'bias_filler': dict(type='constant', value=0)}
    freeze_kwargs = {
                   'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
                   'weight_filler': dict(type='msra'),
                   'bias_filler': dict(type='constant', value=0)}

    net.conv1_1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **norm_kwargs)
    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, stride=2, **merge_kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)
    from_layer = 'relu1_2'

    net.conv2_1 = L.Convolution(net[from_layer], num_output=128, pad=1, kernel_size=3, **norm_kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, stride=2, **merge_kwargs)

# short connection:
    net.conv2_2_shortcut = L.Convolution(net.conv2_2, num_output=256, kernel_size=1, **freeze_kwargs)

    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)
    from_layer = 'relu2_2'

    net.conv3_1 = L.Convolution(net[from_layer], num_output=256, pad=1, kernel_size=3, **norm_kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **norm_kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, stride=2, **merge_kwargs)

    net.eltwise_stage3 = L.Eltwise(net.conv2_2_shortcut, net.conv3_3)

    net.relu3_3 = L.ReLU(net.eltwise_stage3, in_place=True)
    from_layer = 'relu3_3'


    net.conv4_1 = L.Convolution(net[from_layer], num_output=512, pad=1, kernel_size=3, **norm_kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **norm_kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, stride=2, **merge_kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)
    from_layer = 'relu4_3'

    net.conv5_1 = L.Convolution(net[from_layer], num_output=512, pad=1, kernel_size=3, **norm_kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=512, pad=1, kernel_size=3, **norm_kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3 = L.Convolution(net.relu5_2, num_output=512, pad=1, kernel_size=3, stride=2, **merge_kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)
    from_layer = 'relu5_3'

    net.fc6 = L.Convolution(net[from_layer], num_output=1024, pad=6, kernel_size=3, dilation=6, **norm_kwargs)
    net.relu6 = L.ReLU(net.fc6, in_place=True)
    net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **norm_kwargs)
