#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/caffemaker/caffe/dataset/lmdb_sub_pyramid_300x300/lmdb_sub_pyramid/
DATA=/home/caffemaker/caffe/dataset/lmdb_sub_pyramid_300x300/
TOOLS=/home/caffemaker/caffe/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/flickr_sub_pyramid_train_lmdb \
  $DATA/flickr_subpyramid_300x300mean.binaryproto

echo "Done."
