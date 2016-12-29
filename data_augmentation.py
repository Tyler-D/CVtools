#!/usr/bin/env python
# -*- coding: utf-8 -*-

import caffe.io
import os
import numpy as np
import skimage.io
import cv2
import math

def load_augmentate_and_label(img_dir_path, img_path, label_path, label):
    angles = [45, 90, 135, 180, 225, 270, 315]
    label_file = open(label_path, "a+")
    scale = 1.0
    image = caffe.io.load_image(img_path,color=True)
    img = img_path.split("/")[-1]
    w = image.shape[1]
    h = image.shape[0]
    img_name = img.split(".")[0]
    img_type = img.split(".")[-1]
    img_vmirror = cv2.flip(image,1)
    label_str = img + " " + str(label) + "\n"
    label_file.write(label_str)
    img_vmirror_path = os.path.join(img_dir_path,img_name+"_v."+img_type)
    skimage.io.imsave(img_vmirror_path, img_vmirror )
    label_str = img_name+"_v."+img_type + " " + str(label) + "\n"
    label_file.write(label_str)
    for angle in angles:
        rot_mat = cv2.getRotationMatrix2D((w*0.5, h*0.5), angle, scale)
        new_img = cv2.warpAffine(image, rot_mat, (int(math.ceil(w)), int(math.ceil(h))), flags=cv2.INTER_LANCZOS4)
        new_img_path = os.path.join(img_dir_path,img_name+"_"+str(angle)+"."+img_type)
        label_str = img_name + "_" + str(angle) + "." + img_type + " " + str(label) + "\n"
        label_file.write(label_str)
        skimage.io.imsave(new_img_path, new_img)
        new_img_vmirror = cv2.flip(new_img, 1)
        new_img_vmirror_path = os.path.join(img_dir_path, img_name+"_"+str(angle)+"_v."+img_type)
        label_str = img_name+"_"+str(angle)+"_v."+img_type+" "+str(label)+"\n"
        label_file.write(label_str)
        skimage.io.imsave(new_img_vmirror_path, new_img_vmirror)

class DataAug:

    def __init__(self, input_root=None, output_root=None):
        self.input_root = input_root
        self.output_root = output_root
        self.images = []
        self.images_dir = []

    def load_images(self, root, shape=None):
        for img_dir in os.listdir(root):
            img_dir_path = os.path.join(root, img_dir)
            self.images_dir.append(img_dir_path)
            for img in os.listdir(img_dir_path):
                img_path = os.path.join(img_dir_path, img)
                image = caffe.io.load_image(img_path,color=True)
                self.images.append(image)

    def augmentate(self):
        angles = [45, 90, 135, 180, 225, 270, 315]
        scale = 1.0
        for img in self.images:
            print "image shape : ", img.shape
            w = img.shape[1]
            h = img.shape[0]
            img_vmirror = cv2.flip(img,1)
            skimage.io.imsave("testv"+".jpg", img_vmirror )
            for angle in angles:
            #rangle = np.deg2rad(angle)
            # nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
            # nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
                rot_mat = cv2.getRotationMatrix2D((w*0.5, h*0.5), angle, scale)
            # rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
            # rot_mat[0,2] += rot_move[0]
            # rot_mat[1,2] += rot_move[1]
                new_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(w)), int(math.ceil(h))), flags=cv2.INTER_LANCZOS4)
                skimage.io.imsave("test"+str(angle)+".jpg", new_img)
                new_img_vmirror = cv2.flip(new_img, 1)
                skimage.io.imsave("testv"+str(angle)+".jpg", new_img_vmirror)
                # img_rmirror = cv2.flip(new_img, 0)
                # skimage.io.imsave("testh"+str(angle)+".jpg", img_rmirror)

    def load_and_augmentate(self, root):
        angles = [45, 90, 135, 180, 225, 270, 315]
        scale = 1.0
        for img_dir in os.listdir(root):
            img_dir_path = os.path.join(root, img_dir)
            for img in os.listdir(img_dir_path):
                img_path = os.path.join(img_dir_path, img)
                image = caffe.io.load_image(img_path,color=True)
                w = image.shape[1]
                h = image.shape[0]
                img_name = img.split(".")[0]
                img_type = img.split(".")[-1]
                img_vmirror = cv2.flip(image,1)
                img_vmirror_path = os.path.join(img_dir_path,img_name+"_v."+img_type)
                skimage.io.imsave(img_vmirror_path, img_vmirror )
                for angle in angles:
                    rot_mat = cv2.getRotationMatrix2D((w*0.5, h*0.5), angle, scale)
                    new_img = cv2.warpAffine(image, rot_mat, (int(math.ceil(w)), int(math.ceil(h))), flags=cv2.INTER_LANCZOS4)
                    new_img_path = os.path.join(img_dir_path,img_name+"_"+str(angle)+"."+img_type)
                    skimage.io.imsave(new_img_path, new_img)
                    new_img_vmirror = cv2.flip(new_img, 1)
                    new_img_vmirror_path = os.path.join(img_dir_path, img_name+"_"+str(angle)+"_v."+img_type)
                    skimage.io.imsave(new_img_vmirror_path, new_img_vmirror)


    def toArray(self):
        pass

# da = DataAug()
# da.load_and_augmentate("/home/caffemaker/SneakerHead/ModImg/Sneaker_train")
# da.augmentate()

