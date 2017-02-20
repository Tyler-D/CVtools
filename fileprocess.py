#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import shutil
from data_augmentation import load_augmentate_and_label

def divide(dir, traininfo, testinfo, valinfo, labelinfo,topdown=True):
    pic_flag=0
    test_ratio=0
    val_ratio=0.1
    out_put_dir = "/home/caffemaker/caffe/dataset/webface/"
    for root, dirs, files in  os.walk(dir, topdown):
        for dirsname in dirs:
            label_str = dirsname + " " + str(pic_flag) + "\n"
            labelinfo.write(label_str)
            for root1, dirs1, files1 in  os.walk(os.path.join(root, dirsname)):
                test_num=int(test_ratio * len(files1))
                val_num=test_num + int(val_ratio * len(files1))
                count=0
                for name in files1:
                    if count < test_num:
                        newstr=dirsname + name + " " + str(pic_flag) + "\n"
                        newname=dirsname + name
                        shutil.copy(os.path.join(root1, name), os.path.join(out_put_dir+"/test", newname))
                        testinfo.write(newstr)
                    elif count < val_num:
                        newstr=dirsname + name + " " + str(pic_flag) + "\n"
                        newname=dirsname + name
                        shutil.copy(os.path.join(root1, name), os.path.join(out_put_dir+"/val", newname))
                        valinfo.write(newstr)
                    else :
                        newstr=dirsname + name + " " + str(pic_flag) + "\n"
                        newname=dirsname + name
                        shutil.copy(os.path.join(root1, name), os.path.join(out_put_dir+"/train", newname))
                        traininfo.write(newstr)
                    count+=1
            pic_flag+=1
def createList(dir, traininfo, testinfo, valinfo, labelinfo,topdown=True):
    pic_flag=0
    test_ratio=0
    val_ratio=0.1
    for root, dirs, files in  os.walk(dir, topdown):
        for dirsname in dirs:
            label_str = dirsname + " " + str(pic_flag) + "\n"
            labelinfo.write(label_str)
            for root1, dirs1, files1 in  os.walk(os.path.join(root, dirsname)):
                test_num=int(test_ratio * len(files1))
                val_num=test_num + int(val_ratio * len(files1))
                count=0
                for name in files1:
                    if count < test_num:
                        newstr=os.path.join(root1, name) + " " + str(pic_flag) + "\n"
                        testinfo.write(newstr)
                    elif count < val_num:
                        newstr=os.path.join(root1, name) + " " + str(pic_flag) + "\n"
                        valinfo.write(newstr)
                    else :
                        newstr=os.path.join(root1, name) + " " + str(pic_flag) + "\n"
                        traininfo.write(newstr)
                    count+=1
            pic_flag+=1
def process():
    dir="/home/caffemaker/kaggle/data/train_aug"
    # traininfo_path="/home/caffemaker/SneakerHead/final_set_B/train.txt"
    # testinfo_path="/home/caffemaker/SneakerHead/final_set_B/test.txt"
    # valinfo_path="/home/caffemaker/SneakerHead/final_set_B/val.txt"
    # labelinfo_path="/home/caffemaker/SneakerHead/final_set_B/label.txt"
    # divide_augmentate(dir,traininfo_path,testinfo_path,valinfo_path,labelinfo_path)
    traininfo=open('/home/caffemaker/kaggle/data/train.txt', 'w')
    testinfo=open('/home/caffemaker/kaggle/data/test.txt','w')
    valinfo=open('/home/caffemaker/kaggle/data/val.txt','w')
    labelinfo=open('/home/caffemaker/kaggle/data/label.txt', "w")
    createList(dir, traininfo, testinfo, valinfo, labelinfo)
if __name__ == '__main__':
    process()
