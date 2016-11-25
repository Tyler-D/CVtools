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
def divide_augmentate(dir, traininfo_path, testinfo_path, valinfo_path, labelinfo_path,topdown=True):
    pic_flag=0
    test_ratio=0
    val_ratio=0.1
    out_put_dir = "/home/caffemaker/SneakerHead/final_set_B"
    labelinfo = open(labelinfo_path, "w")
    for root, dirs, files in  os.walk(dir, topdown):
        for dirsname in dirs:
            label_str = dirsname + " " + str(pic_flag) + "\n"
            print "Write label " + dirsname +" "+str(pic_flag)
            labelinfo.write(label_str)
            for root1, dirs1, files1 in  os.walk(os.path.join(root, dirsname)):
                test_num=int(test_ratio * len(files1))
                val_num=test_num + int(val_ratio * len(files1))
                count=0
                for name in files1:
                    if count < test_num:
                        newname=dirsname + name
                        shutil.copy(os.path.join(root1, name), os.path.join(out_put_dir+"/test", newname))
                        load_augmentate_and_label(out_put_dir+"/test",os.path.join(out_put_dir+"/test", newname), testinfo_path, pic_flag)
                    elif count < val_num:
                        newname=dirsname + name
                        shutil.copy(os.path.join(root1, name), os.path.join(out_put_dir+"/val", newname))
                        # print "Copy " + os.path.join(root1, name) + " to" + os.path.join(out_put_dir+"/val", newname)
                        load_augmentate_and_label(out_put_dir+"/val",os.path.join(out_put_dir+"/val", newname), valinfo_path, pic_flag)
                    else :
                        newname=dirsname + name
                        shutil.copy(os.path.join(root1, name), os.path.join(out_put_dir+"/train", newname))
                        # print "Copy " + os.path.join(root1, name) + " to" + os.path.join(out_put_dir+"/train", newname)
                        load_augmentate_and_label(out_put_dir+"/train",os.path.join(out_put_dir+"/train", newname), traininfo_path, pic_flag)
                    count+=1
            pic_flag+=1
def process():
    dir="/mnt/ssd-data-1/CASIA-WebFace/"
    # traininfo_path="/home/caffemaker/SneakerHead/final_set_B/train.txt"
    # testinfo_path="/home/caffemaker/SneakerHead/final_set_B/test.txt"
    # valinfo_path="/home/caffemaker/SneakerHead/final_set_B/val.txt"
    # labelinfo_path="/home/caffemaker/SneakerHead/final_set_B/label.txt"
    # divide_augmentate(dir,traininfo_path,testinfo_path,valinfo_path,labelinfo_path)
    traininfo=open('/home/caffemaker/caffe/dataset/webface/train.txt', 'w')
    testinfo=open('/home/caffemaker/caffe/dataset/webface/test.txt','w')
    valinfo=open('/home/caffemaker/caffe/dataset/webface//val.txt','w')
    labelinfo=open('/home/caffemaker/caffe/dataset/webface/label.txt', "w")
    divide(dir, traininfo, testinfo, valinfo, labelinfo)
if __name__ == '__main__':
    process()
