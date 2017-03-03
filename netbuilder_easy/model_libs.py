import os

from common_module_libs import *
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def VGGNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, nopool=False, dropout=True, freeze_layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    net.conv1_1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)

    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    if nopool:
        name = 'conv1_3'
        net[name] = L.Convolution(net.relu1_2, num_output=64, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool1'
        net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    if nopool:
        name = 'conv2_3'
        net[name] = L.Convolution(net.relu2_2, num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool2'
        net[name] = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    if nopool:
        name = 'conv3_4'
        net[name] = L.Convolution(net.relu3_3, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool3'
        net[name] = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    if nopool:
        name = 'conv4_4'
        net[name] = L.Convolution(net.relu4_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool4'
        net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv5_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3 = L.Convolution(net.relu5_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=6, kernel_size=3, dilation=6, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=6, kernel_size=7, dilation=2, **kwargs)
            else:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=3, kernel_size=3, dilation=3, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=3, kernel_size=7, **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=4096, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=4096)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=4096)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net

def ResNet152Body(net, from_layer, use_pool5=True):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
        num_output=64, kernel_size=7, pad=3, stride=2,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True)
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)

    ResBody(net, 'res2c', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True)

    from_layer = 'res3a'
    for i in xrange(1, 8):
      block_name = '3b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True)

    from_layer = 'res4a'
    for i in xrange(1, 36):
      block_name = '4b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=2, use_branch1=True)
    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net


def InceptionV3Body(net, from_layer, output_pred=False):
  # scale is fixed to 1, thus we ignore it.
  use_scale = False

  out_layer = 'conv'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=32, kernel_size=3, pad=0, stride=2, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'conv_1'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=32, kernel_size=3, pad=0, stride=1, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'conv_2'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=64, kernel_size=3, pad=1, stride=1, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'pool'
  net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
      kernel_size=3, stride=2, pad=0)
  from_layer = out_layer

  out_layer = 'conv_3'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=80, kernel_size=1, pad=0, stride=1, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'conv_4'
  ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
      num_output=192, kernel_size=3, pad=0, stride=1, use_scale=use_scale)
  from_layer = out_layer

  out_layer = 'pool_1'
  net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
      kernel_size=3, stride=2, pad=0)
  from_layer = out_layer

  # inceptions with 1x1, 3x3, 5x5 convolutions
  for inception_id in xrange(0, 3):
    if inception_id == 0:
      out_layer = 'mixed'
      tower_2_conv_num_output = 32
    else:
      out_layer = 'mixed_{}'.format(inception_id)
      tower_2_conv_num_output = 64
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=48, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=64, kernel_size=5, pad=2, stride=1),
        ])
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
        dict(name='conv_2', num_output=96, kernel_size=3, pad=1, stride=1),
        ])
    towers.append(tower)
    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=tower_2_conv_num_output, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  # inceptions with 1x1, 3x3(in sequence) convolutions
  out_layer = 'mixed_3'
  towers = []
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=384, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  tower_name = '{}/tower'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
      dict(name='conv_2', num_output=96, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  out_layer = '{}/join'.format(out_layer)
  net[out_layer] = L.Concat(*towers, axis=1)
  from_layer = out_layer

  # inceptions with 1x1, 7x1, 1x7 convolutions
  for inception_id in xrange(4, 8):
    if inception_id == 4:
      num_output = 128
    elif inception_id == 5 or inception_id == 6:
      num_output = 160
    elif inception_id == 7:
      num_output = 192
    out_layer = 'mixed_{}'.format(inception_id)
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        ])
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_2', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_3', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_4', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        ])
    towers.append(tower)
    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  # inceptions with 1x1, 3x3, 1x7, 7x1 filters
  out_layer = 'mixed_8'
  towers = []
  tower_name = '{}/tower'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=320, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  tower_name = '{}/tower_1'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
      dict(name='conv_1', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
      dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
      dict(name='conv_3', num_output=192, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  tower_name = '{}'.format(out_layer)
  tower = InceptionTower(net, from_layer, tower_name, [
      dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
      ])
  towers.append(tower)
  out_layer = '{}/join'.format(out_layer)
  net[out_layer] = L.Concat(*towers, axis=1)
  from_layer = out_layer

  for inception_id in xrange(9, 11):
    num_output = 384
    num_output2 = 448
    if inception_id == 9:
      pool = P.Pooling.AVE
    else:
      pool = P.Pooling.MAX
    out_layer = 'mixed_{}'.format(inception_id)
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=320, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)

    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        ])
    subtowers = []
    subtower_name = '{}/mixed'.format(tower_name)
    subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
        dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ])
    subtowers.append(subtower)
    subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
        dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ])
    subtowers.append(subtower)
    net[subtower_name] = L.Concat(*subtowers, axis=1)
    towers.append(net[subtower_name])

    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=num_output2, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=num_output, kernel_size=3, pad=1, stride=1),
        ])
    subtowers = []
    subtower_name = '{}/mixed'.format(tower_name)
    subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
        dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ])
    subtowers.append(subtower)
    subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
        dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ])
    subtowers.append(subtower)
    net[subtower_name] = L.Concat(*subtowers, axis=1)
    towers.append(net[subtower_name])

    tower_name = '{}/tower_2'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=pool, kernel_size=3, pad=1, stride=1),
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

  if output_pred:
    net.pool_3 = L.Pooling(net[from_layer], pool=P.Pooling.AVE, kernel_size=8, pad=0, stride=1)
    net.softmax = L.InnerProduct(net.pool_3, num_output=1008)
    net.softmax_prob = L.Softmax(net.softmax)

  return net

def ZfaceNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, nopool=False, dropout=True, freeze_layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    net.conv1_1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)

    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    if nopool:
        name = 'conv1_3'
        net[name] = L.Convolution(net.relu1_2, num_output=64, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool1'
        net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    if nopool:
        name = 'conv2_3'
        net[name] = L.Convolution(net.relu2_2, num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool2'
        net[name] = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    if nopool:
        name = 'conv3_4'
        net[name] = L.Convolution(net.relu3_3, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool3'
        net[name] = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    if nopool:
        name = 'conv4_4'
        net[name] = L.Convolution(net.relu4_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool4'
        net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv5_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3 = L.Convolution(net.relu5_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=6, kernel_size=3, dilation=6, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=6, kernel_size=7, dilation=2, **kwargs)
            else:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=3, kernel_size=3, dilation=3, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=3, kernel_size=7, **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=4096, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=4096)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=4096)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net

def LightenedfaceBody(net, from_layer, use_batchnorm=False,  freeze_layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    use_relu = True
    assert from_layer in net.keys()
    # stage 1
    out_layer = "conv1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 96, 5, 2, 1)
    # net.conv1 = L.Convolution(net[from_layer], num_output=96, pad=2, kernel_size=5, **kwargs)
    # net.relu1 = L.ReLU(net.conv1, in_place=True)
    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    # stage 2
    from_layer = "pool1"
    out_layer = "conv2a"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 96, 1, 0, 1)
    # net.conv2a = L.Convolution(net.pool1, num_output=96, kernel_size=1, **kwargs)
    # net.relu2a = L.ReLU(net.conv2a, in_place=True)
    from_layer = "conv2a"
    out_layer = "conv2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 192, 3, 1, 1)
    # net.conv2 = L.Convolution(net.relu2a, num_output=192, pad=1, kernel_size=3, **kwargs)
    # net.relu2 = L.ReLU(net.conv2, in_place=True)
    net.pool2 = L.Pooling(net.conv2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    # stage 3
    from_layer = "pool2"
    out_layer = "conv3a"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 192, 1, 0, 1)
    # net.conv3a = L.Convolution(net.pool2, num_output=192, kernel_size=1, **kwargs)
    # net.relu3a = L.ReLU(net.conv3a, in_place=True)
    from_layer = "conv3a"
    out_layer = "conv3"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 384, 3, 1, 1)
    # net.conv3 = L.Convolution(net.relu3a, num_output=384, pad=1, kernel_size=3, **kwargs)
    # net.relu3 = L.ReLU(net.conv3, in_place=True)
    net.pool3 = L.Pooling(net.conv3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    # stage 4
    from_layer = "pool3"
    out_layer = "conv4a"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 384, 1, 0, 1)
    # net.conv4a = L.Convolution(net.pool3, num_output=384, kernel_size=1, **kwargs)
    # net.relu4a = L.ReLU(net.conv4a, in_place=True)
    from_layer = "conv4a"
    out_layer = "conv4"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
    # net.conv4 = L.Convolution(net.relu4a, num_output=256, pad=1, kernel_size=3, **kwargs)
    # net.relu4 = L.ReLU(net.conv4, in_place=True)

    # stage 5
    from_layer = "conv4"
    out_layer = "conv5a"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1)
    # net.conv5a = L.Convolution(net.relu4, num_output=256, kernel_size=1, **kwargs)
    # net.relu5a = L.ReLU(net.conv5a, in_place=True)
    from_layer = "conv5a"
    out_layer = "conv5"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
    # net.conv5 = L.Convolution(net.relu5a, num_output=256, pad=1, kernel_size=3, **kwargs)
    # net.relu5 = L.ReLU(net.conv5, in_place=True)
    net.pool4 = L.Pooling(net.conv5, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    # final stage( fully_conv )
    from_layer = "pool4"
    out_layer = "fc1_conv"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 6, 1)
    # net.fc1 = L.Convolution(net.pool4, num_output=256, pad=6, kernel_size=3, dilation=6, **kwargs)
    # net.relu_fc1 = L.ReLU(net.fc1, in_place=True)
    from_layer = "fc1_conv"
    out_layer = "fc2_conv"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1)
    # net.fc2 = L.Convolution(net.relu_fc1, num_output=256, kernel_size=1, **kwargs)
    # net.relu_fc2 = L.ReLU(net.fc2, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net

def DeBlurNetBody(net, from_layer, use_batchnorm=False,  freeze_layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    use_relu = False
    # stage 1
    out_layer = "flat_conv0" #to U3
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 64, 5, 2, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    # stage 2
    from_layer = relu_name
    out_layer = "down_conv1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 64, 3, 1, 2)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv1_1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv1_2"  # to U2
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    # stage 3
    from_layer = relu_name
    out_layer = "down_conv2"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv2_1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv2_2"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv2_3"  # to U1
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    # stage 3
    from_layer = relu_name
    out_layer = "down_conv3"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv3_1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv3_2"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv3_3"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    # stage 4
    from_layer = relu_name
    out_layer = "up_conv1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 4, 1, 2)
    eltwise_name = '{}_eltwise'.format("up1")
    net[eltwise_name] = L.Eltwise(net[out_layer], net["flat_conv2_3"])
    relu_name = '{}_relu'.format(eltwise_name)
    net[relu_name] = L.ReLU(net[eltwise_name], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv4_1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv4_2"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv4_3"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    # stage 5
    from_layer = relu_name
    out_layer = "up_conv2"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 4, 1, 2)
    eltwise_name = '{}_eltwise'.format("up2")
    net[eltwise_name] = L.Eltwise(net[out_layer], net["flat_conv1_2"])
    relu_name = '{}_relu'.format(eltwise_name)
    net[relu_name] = L.ReLU(net[eltwise_name], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv5_1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv5_2"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 64, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    # stage 6
    from_layer = relu_name
    out_layer = "up_conv3"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 64, 4, 1, 2)
    eltwise_name = '{}_eltwise'.format("up3")
    net[eltwise_name] = L.Eltwise(net[out_layer], net["flat_conv0"])
    relu_name = '{}_relu'.format(eltwise_name)
    net[relu_name] = L.ReLU(net[eltwise_name], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv6_1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 15, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv6_2"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 3, 3, 1, 1)
    sigmoid_name = '{}_sigmoid'.format(out_layer)
    net[sigmoid_name] = L.Sigmoid(net[out_layer], in_place=True)

    return net

def SuperReNetBody(net, from_layer, gt_data, phase, batch_size, use_batchnorm=False,  freeze_layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    use_relu = False
    # stage 1
    out_layer = "flat_conv0" #to U3
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 64, 5, 2, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    # stage 2
    from_layer = relu_name
    out_layer = "down_conv1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 64, 3, 1, 2)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv1_1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv1_2"  # to U2
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    # stage 3
    from_layer = relu_name
    out_layer = "down_conv2"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv2_1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv2_2"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv2_3"  # to U1
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    # stage 3
    from_layer = relu_name
    out_layer = "down_conv3"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv3_1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv3_2"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv3_3"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    # stage 4
    from_layer = relu_name
    out_layer = "up_conv1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 4, 1, 2)
    eltwise_name = '{}_eltwise'.format("up1")
    net[eltwise_name] = L.Eltwise(net[out_layer], net["flat_conv2_3"])
    relu_name = '{}_relu'.format(eltwise_name)
    net[relu_name] = L.ReLU(net[eltwise_name], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv4_1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv4_2"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv4_3"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    # stage 5
    from_layer = relu_name
    out_layer = "up_conv2"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 4, 1, 2)
    eltwise_name = '{}_eltwise'.format("up2")
    net[eltwise_name] = L.Eltwise(net[out_layer], net["flat_conv1_2"])
    relu_name = '{}_relu'.format(eltwise_name)
    net[relu_name] = L.ReLU(net[eltwise_name], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv5_1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv5_2"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 64, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    # stage 6
    from_layer = relu_name
    out_layer = "up_conv3"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 64, 4, 1, 2)
    eltwise_name = '{}_eltwise'.format("up3")
    net[eltwise_name] = L.Eltwise(net[out_layer], net["flat_conv0"])
    relu_name = '{}_relu'.format(eltwise_name)
    net[relu_name] = L.ReLU(net[eltwise_name], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv6_1"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 15, 3, 1, 1)
    relu_name = '{}_relu'.format(out_layer)
    net[relu_name] = L.ReLU(net[out_layer], in_place=True)

    from_layer = relu_name
    out_layer = "flat_conv6_2"
    DeconvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 3, 3, 1, 1)
    sigmoid_name = '{}_sigmoid'.format(out_layer)
    net[sigmoid_name] = L.Sigmoid(net[out_layer], in_place=True)

    concat_name = 'pred_gt_concat'
    concat_layer= []
    concat_layer.append(net[sigmoid_name])
    concat_layer.append(net[gt_data])
    net[concat_name] = L.Concat(*concat_layer, axis=0)

    if (phase == 'train' ) or (phase == 'test'):
        kwargs = {
                'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
                'propagate_down': True}
        # VGG_FC_REDUCED

        net.conv1_1 = L.Convolution(net['pred_gt_concat'], num_output=64, pad=1, kernel_size=3, **kwargs)

        net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
        net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
        net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

        name = 'pool1'
        net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        net.conv2_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
        net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
        net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
        net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

        name = 'pool2'
        net[name] = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        net.conv3_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
        net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
        net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
        net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
        net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
        net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

        net.slicer = L.Slice(net.relu3_3, axis=0, slice_point=int(batch_size))

        # manually add euclidean loss in prototxt
        # net.loss = L.Euclidean(net.pred, net.gt)

    return net

def ZNetBody(net, from_layer, dropout=True, freeze_layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

def ResNet10Body(net, from_layer, use_prelu=False):
    bn_kwargs = {'param': [dict(lr_mult=0.0), dict(lr_mult=0.0), dict(lr_mult=0.0)]}
    scale_kwargs = {'param': [dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=1.0)],
                    'scale_param': dict(bias_term=True)
                    }
    # data bn:
    net.data_bn_new = L.BatchNorm(net[from_layer], **bn_kwargs)
    net.data_scale = L.Scale(net['data_bn_new'], in_place=True, **scale_kwargs)
    from_layer = 'data_bn_new'

    # conv1
    conv_kwargs = {'param': [dict(lr_mult=1.0, decay_mult=1.0), dict(lr_mult=2.0, decay_mult=1.0)],
                    'weight_filler': dict(type='msra', variance_norm=P.Filler.FAN_OUT),
                    'bias_filler': dict(type='constant', value=0)
                  }
    net.conv1 = L.Convolution(net[from_layer], num_output=64, pad=3, kernel_size=7,
                              stride=2, **conv_kwargs)
    net.conv1_bn_new = L.BatchNorm(net['conv1'], in_place=True, **bn_kwargs)
    net.conv1_scale_new = L.Scale(net['conv1_bn_new'], in_place=True, **scale_kwargs)
    if (use_prelu):
        relu_name = "{}_prelu".format('conv1')
        net[relu_name] = L.PReLU(net['conv1_scale_new'], in_place=True)
    else :
        relu_name = "{}_relu".format('conv1')
        net[relu_name] = L.ReLU(net['conv1_scale_new'], in_place=True)
    net.conv1_pool = L.Pooling(net[relu_name], kernel_size=3, stride=2)
    from_layer = 'conv1_pool'

    # layer_64_1_conv1
    conv_kwargs = {'param': dict(lr_mult=1.0, decay_mult=1.0),
                   'weight_filler': dict(type='msra'),
                   'bias_term': False
                  }
    net.layer_64_1_conv1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3,
                                         stride=1, **conv_kwargs)
    net.layer_64_1_bn2_new = L.BatchNorm(net['layer_64_1_conv1'], in_place=True, **bn_kwargs)
    net.layer_64_1_scale2_new = L.Scale(net['layer_64_1_bn2_new'], in_place=True, **scale_kwargs)
    if (use_prelu):
        relu_name = "{}_prelu2".format('layer_64_1')
        net[relu_name] = L.PReLU(net['layer_64_1_scale2_new'], in_place=True)
    else :
        relu_name = "{}_relu2".format('layer_64_1')
        net[relu_name] = L.ReLU(net['layer_64_1_scale2_new'], in_place=True)
    from_layer = relu_name

    # layer_64_1_conv2
    net.layer_64_1_conv2 = L.Convolution(net[from_layer], in_place=True, num_output=64, pad=1, kernel_size=3,
                                         stride=1, **conv_kwargs)
    net.layer_64_1_sum = L.Eltwise(net['layer_64_1_conv2'], net['conv1_pool'])
    from_layer = 'layer_64_1_sum'

    # layer_128_1_bn1
    net.layer_128_1_bn1_new = L.BatchNorm(net[from_layer], **bn_kwargs)
    net.layer_128_1_scale1_new = L.Scale(net['layer_128_1_bn1_new'], in_place=True, **scale_kwargs)
    if (use_prelu):
        relu_name = "{}_prelu1".format('layer_128_1')
        net[relu_name] = L.PReLU(net['layer_128_1_scale1_new'], in_place=True)
    else :
        relu_name = "{}_relu1".format('layer_128_1')
        net[relu_name] = L.ReLU(net['layer_128_1_scale1_new'], in_place=True)
    from_layer = relu_name

    # layer_128_1_conv1
    net.layer_128_1_conv1 = L.Convolution(net[from_layer], num_output=128, pad=1, kernel_size=3,
                                          stride=2, **conv_kwargs)
    net.layer_128_1_bn2_new = L.BatchNorm(net['layer_128_1_conv1'], in_place=True, **bn_kwargs)
    net.layer_128_1_scale2_new = L.Scale(net['layer_128_1_bn2_new'], in_place=True, **scale_kwargs)
    if (use_prelu):
        relu_name = "{}_prelu2".format('layer_128_1')
        net[relu_name] = L.PReLU(net['layer_128_1_scale2_new'], in_place=True)
    else :
        relu_name = "{}_relu2".format('layer_128_1')
        net[relu_name] = L.ReLU(net['layer_128_1_scale2_new'], in_place=True)
    from_layer = relu_name

    # layer_128_1_conv2
    net.layer_128_1_conv2 = L.Convolution(net[from_layer], num_output=128, pad=1, kernel_size=3,
                                          stride=1, **conv_kwargs)
    # layer_128_1_conv_expand
    net.layer_128_1_conv_expand = L.Convolution(net['layer_128_1_relu1'], num_output=128, pad=0, kernel_size=1,
                                                stride=2, **conv_kwargs)

    # res2 sum
    net.layer_128_1_sum = L.Eltwise(net['layer_128_1_conv2'], net['layer_128_1_conv_expand'])

    # layer_256_1_bn
    net.layer_256_1_bn1_new = L.BatchNorm(net['layer_128_1_sum'], **bn_kwargs)
    net.layer_256_1_scale1_new = L.Scale(net['layer_256_1_bn1_new'], in_place=True, **scale_kwargs)
    if (use_prelu):
        relu_name = "{}_prelu1".format('layer_256_1')
        net[relu_name] = L.PReLU(net['layer_256_1_scale1_new'], in_place=True)
    else :
        relu_name = "{}_relu1".format('layer_256_1')
        net[relu_name] = L.ReLU(net['layer_256_1_scale1_new'], in_place=True)
    from_layer = relu_name

    # layer_256_1_conv1
    net.layer_256_1_conv1 = L.Convolution(net[from_layer], num_output=256, pad=1, kernel_size=3,
                                          stride=2, **conv_kwargs)
    net.layer_256_1_bn2_new = L.BatchNorm(net['layer_256_1_conv1'], in_place=True, **bn_kwargs)
    net.layer_256_1_scale2_new = L.Scale(net['layer_256_1_bn2_new'], in_place=True, **scale_kwargs)
    if (use_prelu):
        relu_name = "{}_prelu2".format('layer_256_1')
        net[relu_name] = L.PReLU(net['layer_256_1_scale2_new'], in_place=True)
    else :
        relu_name = "{}_relu2".format('layer_256_1')
        net[relu_name] = L.ReLU(net['layer_256_1_scale2_new'], in_place=True)
    from_layer = relu_name

    # sum
    net.layer_256_1_conv2 = L.Convolution(net[from_layer], num_output=256, pad=1, kernel_size=3,
                                          stride=1, **conv_kwargs)
    net.layer_256_1_conv_expand = L.Convolution(net['layer_256_1_relu1'], num_output=256, pad=0, kernel_size=1,
                                                stride=2, **conv_kwargs)
    net.layer_256_1_sum = L.Eltwise(net['layer_256_1_conv2'], net['layer_256_1_conv_expand'])

    # layer_512_1_bn1
    net.layer_512_1_bn1_new = L.BatchNorm(net['layer_256_1_sum'], **bn_kwargs)
    net.layer_512_1_scale1_new = L.Scale(net['layer_512_1_bn1_new'], in_place=True, **scale_kwargs)
    if (use_prelu):
        relu_name = "{}_prelu1".format('layer_512_1')
        net[relu_name] = L.PReLU(net['layer_512_1_scale1_new'], in_place=True)
    else :
        relu_name = "{}_relu1".format('layer_512_1')
        net[relu_name] = L.ReLU(net['layer_512_1_scale1_new'], in_place=True)
    from_layer = relu_name

    # layer_512_1_conv1
    net.layer_512_1_conv1 = L.Convolution(net[from_layer], num_output=512, pad=1, kernel_size=3,
                                          stride=2, **conv_kwargs)
    net.layer_512_1_bn2_new = L.BatchNorm(net['layer_512_1_conv1'], in_place=True, **bn_kwargs)
    net.layer_512_1_scale2_new = L.Scale(net['layer_512_1_bn2_new'], in_place=True, **scale_kwargs)
    if (use_prelu):
        relu_name = "{}_prelu2".format('layer_512_1')
        net[relu_name] = L.PReLU(net['layer_512_1_scale2_new'], in_place=True)
    else :
        relu_name = "{}_relu2".format('layer_512_1')
        net[relu_name] = L.ReLU(net['layer_512_1_scale2_new'], in_place=True)
    from_layer = relu_name

    # sum
    net.layer_512_1_conv2 = L.Convolution(net[from_layer], num_output=512, pad=1, kernel_size=3,
                                          stride=1, **conv_kwargs)
    net.layer_512_1_conv_expand = L.Convolution(net['layer_512_1_relu1'], num_output=512, pad=0, kernel_size=1,
                                                stride=2, **conv_kwargs)
    net.layer_512_1_sum = L.Eltwise(net['layer_512_1_conv2'], net['layer_512_1_conv_expand'])

    # last_bn
    net.last_bn_new = L.BatchNorm(net['layer_512_1_sum'], in_place=True, **bn_kwargs)
    net.last_scale_new = L.Scale(net['last_bn_new'], in_place=True, **scale_kwargs)
    if (use_prelu):
        relu_name = "{}_prelu".format('last')
        net[relu_name] = L.PReLU(net['last_scale_new'], in_place=True)
    else :
        relu_name = "{}_relu".format('last')
        net[relu_name] = L.ReLU(net['last_scale_new'], in_place=True)

    # global_pooling
    # net.global_pool = L.Pooling(net['last_relu'], pool=P.Pooling.AVE, global_pooling=True)
