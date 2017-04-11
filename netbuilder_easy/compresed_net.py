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
    net.conv2_2_new = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, stride=2, **merge_kwargs)

    net.relu2_2 = L.ReLU(net.conv2_2_new, in_place=True)
    from_layer = 'relu2_2'
# short connection:
    net.conv2_2_shortcut = L.Convolution(net.conv2_2_new, num_output=256, kernel_size=1, **freeze_kwargs)

    net.conv3_1 = L.Convolution(net[from_layer], num_output=256, pad=1, kernel_size=3, **norm_kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **norm_kwargs)
# short connection:
    net.eltwise_stage3_new = L.Eltwise(net.conv2_2_shortcut, net.conv3_2)
    net.relu3_2 = L.ReLU(net.eltwise_stage3_new, in_place=True)
    net.conv3_3_new = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, stride=2, **merge_kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3_new, in_place=True)
    from_layer = 'relu3_3'

    net.conv4_1 = L.Convolution(net[from_layer], num_output=512, pad=1, kernel_size=3, **norm_kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **norm_kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3_new = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, stride=2, **merge_kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3_new, in_place=True)
    from_layer = 'relu4_3'

    net.conv5_1 = L.Convolution(net[from_layer], num_output=512, pad=1, kernel_size=3, **norm_kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=512, pad=1, kernel_size=3, **norm_kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3_new = L.Convolution(net.relu5_2, num_output=512, pad=1, kernel_size=3, stride=2, **merge_kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3_new, in_place=True)
    from_layer = 'relu5_3'

    net.fc6 = L.Convolution(net[from_layer], num_output=1024, pad=6, kernel_size=3, dilation=6, **norm_kwargs)
    net.relu6 = L.ReLU(net.fc6, in_place=True)
    net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **norm_kwargs)

def JRes22Block(net, from_layer, stage_id, num_output, freeze=False):
    if freeze:
        kwargs = {
                'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
                'weight_filler': dict(type='msra'),
                'bias_filler': dict(type='constant', value=0)}
    else :
        kwargs = {
                'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                'weight_filler': dict(type='xavier'),
                'bias_filler': dict(type='constant', value=0)}
# block_a
    skip_layer = from_layer
    conv_name = 'conv{}a_1'.format(stage_id)
    relu_name = 'relu{}a_1'.format(stage_id)
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output, pad=1, kernel_size=3, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    from_layer = relu_name
    conv_name = 'conv{}a_2'.format(stage_id)
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output, pad=1, kernel_size=3, **kwargs)
    from_layer = conv_name

    eltwise_name = 'res{}_2'.format(stage_id)
    net[eltwise_name] = L.Eltwise(net[skip_layer], net[from_layer])
    relu_name = 'relu{}_2'.format(stage_id)
    net[relu_name] = L.PReLU(net[eltwise_name], in_place=True)

# block_b
    from_layer = eltwise_name
    skip_layer = from_layer
    conv_name = 'conv{}a_3'.format(stage_id)
    relu_name = 'relu{}a_3'.format(stage_id)
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output, pad=1, kernel_size=3, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    from_layer = relu_name
    conv_name = 'conv{}a_4'.format(stage_id)
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output, pad=1, kernel_size=3, **kwargs)
    from_layer = conv_name

    eltwise_name = 'res{}_4'.format(stage_id)
    net[eltwise_name] = L.Eltwise(net[skip_layer], net[from_layer])
    relu_name = 'relu{}_4'.format(stage_id)
    net[relu_name] = L.PReLU(net[eltwise_name], in_place=True)
    return relu_name

def compJRes22(net, from_layer):
    kwargs = {
              'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
              'weight_filler': dict(type='xavier'),
              'bias_filler': dict(type='constant', value=0)}
    #stage1:
    conv_name = 'conv1a'
    relu_name = 'relu1a'
    net[conv_name] = L.Convolution(net[from_layer], num_output=32, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    from_layer = relu_name
    conv_name = 'conv1'
    relu_name = 'relu1'
    net[conv_name] = L.Convolution(net[from_layer], num_output=48, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    net.pool1 = L.Pooling(net[relu_name], pool=P.Pooling.MAX, kernel_size=2, stride=2)

    #stage2:
    from_layer='pool1'
    stage_id = 2
    output = JRes22Block(net, from_layer, stage_id, num_output=48)
    from_layer=output
    conv_name = 'conv2'
    relu_name = 'relu2'
    net[conv_name] = L.Convolution(net[from_layer], num_output=96, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    net.pool2 = L.Pooling(net[relu_name], pool=P.Pooling.MAX, kernel_size=2, stride=2)

#stage3:
    from_layer='pool2'
    stage_id = 3
    output = JRes22Block(net, from_layer, stage_id, num_output=96)
    from_layer=output
    conv_name = 'conv3'
    relu_name = 'relu3'
    net[conv_name] = L.Convolution(net[from_layer], num_output=192, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    net.pool3 = L.Pooling(net[relu_name], pool=P.Pooling.MAX, kernel_size=2, stride=2)

#stage4:
    from_layer='pool3'
    stage_id = 4
    output = JRes22Block(net, from_layer, stage_id, num_output=192)
    from_layer=output
    conv_name = 'conv4'
    relu_name = 'relu4'
    net[conv_name] = L.Convolution(net[from_layer], num_output=128, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    net.pool4 = L.Pooling(net[relu_name], pool=P.Pooling.MAX, kernel_size=2, stride=2)

#stage5:
    from_layer='pool4'
    stage_id = 5
    output = JRes22Block(net, from_layer, stage_id, num_output=128)
    from_layer=output
    conv_name = 'conv5'
    relu_name = 'relu5'
    net[conv_name] = L.Convolution(net[from_layer], num_output=128, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    net.pool5 = L.Pooling(net[relu_name], pool=P.Pooling.MAX, kernel_size=2, stride=2)

def compJRes24(net, from_layer):
    kwargs = {
              'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
              'weight_filler': dict(type='xavier'),
              'bias_filler': dict(type='constant', value=0)}
    freeze_kwargs = {
                   'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
                   'weight_filler': dict(type='msra'),
                   'bias_filler': dict(type='constant', value=0)}
    #stage1:
    conv_name = 'conv1a'
    relu_name = 'relu1a'
    net[conv_name] = L.Convolution(net[from_layer], num_output=32, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    from_layer = relu_name
    conv_name = 'conv1'
    relu_name = 'relu1'
    net[conv_name] = L.Convolution(net[from_layer], num_output=48, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    net.pool1 = L.Pooling(net[relu_name], pool=P.Pooling.MAX, kernel_size=2, stride=2)

    #stage2:
    from_layer='pool1'
    stage_id = 2
    output = JRes22Block(net, from_layer, stage_id, num_output=48)
    from_layer=output
    conv_name = 'conv2'
    relu_name = 'relu2'
    net[conv_name] = L.Convolution(net[from_layer], num_output=96, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    net.pool2 = L.Pooling(net[relu_name], pool=P.Pooling.MAX, kernel_size=2, stride=2)

#stage3:
    from_layer='pool2'
    stage_id = 3
    output = JRes22Block(net, from_layer, stage_id, num_output=96)
    from_layer=output
    conv_name = 'conv3'
    relu_name = 'relu3'
    net[conv_name] = L.Convolution(net[from_layer], num_output=192, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    net.pool3 = L.Pooling(net[relu_name], pool=P.Pooling.MAX, kernel_size=2, stride=2)

#stage4:
    from_layer='pool3'
    stage_id = 4
    output = JRes22Block(net, from_layer, stage_id, num_output=192)
    from_layer=output
    conv_name = 'conv4'
    relu_name = 'relu4'
    net[conv_name] = L.Convolution(net[from_layer], num_output=128, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    net.pool4 = L.Pooling(net[relu_name], pool=P.Pooling.MAX, kernel_size=2, stride=2)

#stage5:
    from_layer='pool4'
    stage_id = 5
    output = JRes22Block(net, from_layer, stage_id, num_output=128)
    from_layer=output
    conv_name = 'conv5'
    relu_name = 'relu5'
    net[conv_name] = L.Convolution(net[from_layer], num_output=128, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    net.pool5 = L.Pooling(net[relu_name], pool=P.Pooling.MAX, kernel_size=2, stride=2)

#stage6
    from_layer = 'pool5'
    conv_name = 'conv6'
    relu_name = 'relu6'
    net[conv_name] = L.Convolution(net[from_layer], num_output=128, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    from_layer = 'relu6'
    conv_name = 'conv7'
    relu_name = 'relu7'
    net[conv_name] = L.Convolution(net[from_layer], num_output=128, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)

def compJRes12(net, from_layer):
    kwargs = {
              'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
              'weight_filler': dict(type='xavier'),
              'bias_filler': dict(type='constant', value=0)}
    #stage1:
    conv_name = 'conv1a'
    relu_name = 'relu1a'
    net[conv_name] = L.Convolution(net[from_layer], num_output=32, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    from_layer = relu_name
    conv_name = 'conv1'
    relu_name = 'relu1'
    net[conv_name] = L.Convolution(net[from_layer], num_output=48, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    net.pool1 = L.Pooling(net[relu_name], pool=P.Pooling.MAX, kernel_size=2, stride=2)

    #stage2:
    from_layer='pool1'
    stage_id = 2
    output = JRes22Block(net, from_layer, stage_id, num_output=48)
    from_layer=output
    conv_name = 'conv2'
    relu_name = 'relu2'
    net[conv_name] = L.Convolution(net[from_layer], num_output=96, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)
    net.pool2 = L.Pooling(net[relu_name], pool=P.Pooling.MAX, kernel_size=2, stride=2)

#stage3:
    from_layer='pool2'
    stage_id = 3
    output = JRes22Block(net, from_layer, stage_id, num_output=96)
    from_layer=output
    conv_name = 'conv3'
    relu_name = 'relu3'
    net[conv_name] = L.Convolution(net[from_layer], num_output=192, pad=1, kernel_size=3, stride=1, **kwargs)
    net[relu_name] = L.PReLU(net[conv_name], in_place=True)

