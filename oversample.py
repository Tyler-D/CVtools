#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import skimage
import skimage.io

def oversample(img, crop_ratio, horizon_flip=False, vertical_flip=False):
    """
    4 corner crops , horizonly flips and vertically flips
    Parameter:
    -------------
    img: h*w*c image; ndarray, dtype= np.float32
    crop_ratio: (crop_height_ratio, crop_width_ratio); tuple
    horizon_flip: flip the image horizonly;bool
    vertical_flip:

    Return:
    ------------
    crops: N*h*w*c images; ndarray
    """
    img_height = img.shape[0]
    img_width = img.shape[1]
    img_channel = img.shape[-1]
    crop_dims = (int(img_height * crop_ratio[0]), int(img_width * crop_ratio[1]))
    h_indices = (0, img_height - crop_dims[0])
    w_indices = (0, img_width - crop_dims[1])
    crop_idx = np.empty((4, 4), dtype=int)
    idx = 0
    for i in h_indices:
        for j in w_indices:
            #crop_idx : (ymin, xmin, ymax, xmax)
            crop_idx[idx] = (i, j, i + crop_dims[0], j + crop_dims[1])
            idx+=1
    num = 4
    if horizon_flip:
        num += 4
    if vertical_flip:
        num += 4
    crops = np.empty((num, crop_dims[0], crop_dims[1], img_channel), dtype=np.float32)
    idx = 0
    for crop in crop_idx:
        crops[idx] = img[crop[0]:crop[2], crop[1]:crop[3], :]
        img_crop = crops[idx]
        idx += 1
        if horizon_flip:
            crops[idx] = np.flipud(img_crop)
            idx += 1
        if vertical_flip:
            crops[idx] = np.fliplr(img_crop)
            idx += 1
    return crops


#test
# img = skimage.img_as_float(skimage.io.imread("./test.jpg")).astype(np.float32)
# # img = skimage.io.imread("./test.jpg")
# print img.shape
# print type(img)
# crops = oversample(img,(0.8,0.8),True,True)
# print crops.shape
# for idx in xrange(crops.shape[0]):
    # skimage.io.imsave(os.path.join("./", str(idx) + ".jpg"), crops[idx, :, :, :])



