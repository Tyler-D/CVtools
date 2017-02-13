#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2 as cv
import numpy as np

def downup(img, ratio=1, mode='linear'):
   '''
   make train data for super resolution

   Parameter:
   -------
   img: Ndarray. cv loaded img
   ratio: float. resize ratio ~ (0,1)
   mode: interpolation mode. /cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR

   Return:
   -------
   img: down && up img

   '''
   img_height, img_width, channel = img.shape
   new_height = int(img_height * ratio)
   new_width = int(img_width * ratio)

   if mode == 'area':
       img_temp = cv.resize(img, (new_width, new_height), interpolation = cv.INTER_AREA)
       img_new = cv.resize(img_temp, (img_width, img_height), interpolation = cv.INTER_AREA)
   elif mode == 'linear':
       img_temp = cv.resize(img, (new_width, new_height), interpolation = cv.INTER_LINEAR)
       img_new = cv.resize(img_temp, (img_width, img_height), interpolation = cv.INTER_LINEAR)
   elif mode == 'cubic' :
       img_temp = cv.resize(img, (new_width, new_height), interpolation = cv.INTER_CUBIC)
       img_new = cv.resize(img_temp, (img_width, img_height), interpolation = cv.INTER_CUBIC)

   return img_new

def pyrDownUp(img, times=1):
    img_temp = img
    for i in xrange(int(times)):
        img_temp = cv.pyrDown(img_temp)
    img_new = img_temp
    for i in xrange(int(times)):
        img_new = cv.pyrUp(img_new)
    return img_new

# img = cv.imread("./test.jpg")
# img_new = pyrDownUp(img, 2)
# cv.imwrite("./new.jpg", img_new)
