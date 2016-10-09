#!/usr/bin/env python
# encoding: utf-8

import caffe
import numpy as np
import sys

if(len(sys.argv)!=3):
    print "Usage: python binaryproto2npy.py proto.binaryproto out.npy"
    sys.exit()

blob =  caffe.proto.caffe_pb2.BlobProto()
data = open(sys.argv[1],'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
print arr.shape
np.save(sys.argv[2], arr)
