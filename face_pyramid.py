#/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import shutil

# ROOT_IMG_DIR="/home/caffemaker/caffe/dataset/flickr/img"
# ROOT_ANNO_DIR="/home/caffemaker/caffe/dataset/flickr/anno"
# TARGET_IMG_DIR="/home/caffemaker/caffe/dataset/flickr/img_pyramid"
# # TARGET_IMG_DIR="/mnt/ssd-data-1/flickr/img_pyramid"
# TARGET_ANNO_DIR="/home/caffemaker/caffe/dataset/flickr/anno_pyramid"
# # TARGET_ANNO_DIR="/mnt/ssd-data-1/flickr/anno_pyramid"
# RGB_FILE_LIST="/home/caffemaker/caffe/dataset/flickr/1or2_img_list.txt"

BLACK=[0,0,0]

def imgPyramidBorder(img_path, anno_path, img_name, file_info, discount):
    try :
        img = cv2.imread(img_path)
        img_height = img.shape[0]
        img_width = img.shape[1]

        #pyramidDown the img
        img_pm = cv2.resize(img, None,fx=discount,fy=discount, interpolation=cv2.INTER_AREA)
        img_pm_height = img_pm.shape[0]
        img_pm_width = img_pm.shape[1]

        top=bottom= (img_height - img_pm_height) /2
        left=right = (img_width - img_pm_width) /2

        img_pmb = cv2.copyMakeBorder(img_pm, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
        new_img_path = os.path.join(TARGET_IMG_DIR,img_name + '.jpg')
        cv2.imwrite(new_img_path, img_pmb)

        #modify the annotation to satisfy the new img:
        anno_file = open(anno_path, "r")
        new_anno_path = os.path.join(TARGET_ANNO_DIR, img_name+'.txt')
        new_anno_file = open(new_anno_path, "w")
        for line in anno_file.readlines():
            line = line.split()
            orig_bbox_xmin = int(line[1])
            orig_bbox_ymin = int(line[2])
            orig_bbox_width = int(line[3]) - orig_bbox_xmin
            orig_bbox_height = int(line[4]) - orig_bbox_ymin


            new_bbox_xmin = int(round(orig_bbox_xmin * discount)) + left
            new_bbox_ymin = int(round(orig_bbox_ymin * discount)) + top
            new_bbox_width = int(round(orig_bbox_width * discount))
            new_bbox_height = int(round(orig_bbox_height * discount))

            new_bbox_xmax = new_bbox_xmin + new_bbox_width
            new_bbox_ymax = new_bbox_ymin + new_bbox_height

            new_anno_str = line[0] + ' ' + str(new_bbox_xmin) + ' ' +str(new_bbox_ymin) + ' '+ str(new_bbox_xmax) + ' ' + str(new_bbox_ymax) + "\n"
            new_anno_file.write(new_anno_str)

        new_anno_file.close()
        anno_file.close()
        file_info_str = new_img_path + ' '+ new_anno_path + "\n"
        file_info.write(file_info_str)
        return new_img_path, new_anno_path
    except Exception, e :
        print e

# cnt = 0
# val_num = 1500
# val_file = open("/home/caffemaker/caffe/dataset/flickr/val.txt", "w")
# train_file = open("/home/caffemaker/caffe/dataset/flickr/train.txt", "w")
# discount = [0.6, 0.8, 0.5]
# # end = 0
# with open(RGB_FILE_LIST, "r") as f:
    # for line in f.readlines():
        # # if (cnt == 2) :
            # # break
        # try :
            # # if (end >5):
                # # break
            # if (cnt < val_num):
                # file_info = val_file
            # else :
                # file_info = train_file
            # orig_img_path=line.split()[0]
            # orig_anno_path=line.split()[1]
            # orig_img_name = (orig_img_path.split('/')[-1]).split('.')[0]

            # new_img_path = os.path.join(TARGET_IMG_DIR, orig_img_name+'.jpg')
            # new_anno_path = os.path.join(TARGET_ANNO_DIR, orig_img_name + '.txt')
            # shutil.copy(orig_img_path, new_img_path)
            # shutil.copy(orig_anno_path, new_anno_path)
            # file_info_str = new_img_path + ' '+ new_anno_path + "\n"
            # file_info.write(file_info_str)

            # img_path = orig_img_path
            # anno_path = orig_anno_path
            # for it in xrange(3):
                # new_img_name = orig_img_name +'_' + str(it)
                # img_path, anno_path = imgPyramidBorder(img_path, anno_path, new_img_name, file_info, discount[it])
            # cnt += 1
            # # end += 1
        # except Exception, e:
            # print e
# val_file.close()
# train_file.close()




