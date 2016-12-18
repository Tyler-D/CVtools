#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2 as cv
import numpy as np

def MotionBlur(img, steps):
    '''
    Parameters:
    -----------
    img: Ndarray. CV loaded image
    steps: tuple(step_x, step_y). Motion Velocity

    Return:
    -------
    img: Ndarray. Blurred image
    '''
    BLACK=[0,0,0]
    img_height, img_width, channel= img.shape
    step_x , step_y = steps

    hori = abs(step_x)
    vert = abs(step_y)
    img_mb = cv.copyMakeBorder(img, vert, vert, hori, hori, cv.BORDER_CONSTANT, value=BLACK)

    img_mask = np.zeros((img_height,img_width, channel))
    if step_x!=0 :
        sign_x = step_x / hori
        for x in xrange(0, hori):
            img_mask += img_mb[vert : vert+img_height, hori-x*sign_x : hori+img_width-x*sign_x]

    if step_y!=0 :
        sign_y = step_y / vert
        for y in xrange(0, vert):
            img_mask += img_mb[vert-y*sign_y : vert+img_height-y*sign_y, hori:hori+img_width]

    img_mask /= (hori+vert)
    return img_mask

