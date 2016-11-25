#!/bin/bash

ROOT_CAFFE="/home/caffemaker/detection/caffe-ssd/"

f_data_dir="/"
data_dir="/home/caffemaker/caffe/dataset/pic_video/"
example_dir="/home/caffemaker/dataset/pic_video"
label_map_file="$data_dir/labelmap_flickr.prototxt"
anno_type="detection"
label_type="txt"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0
extra_cmd="--encode-type=jpg --encoded"


for subset in train val
do
    python $ROOT_CAFFE/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$label_map_file --label-type=$label_type  --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $f_data_dir $data_dir/$subset.txt $data_dir/lmdb/cam"_"$subset"_"lmdb $example_dir
done

