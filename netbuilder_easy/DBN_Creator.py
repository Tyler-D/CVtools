from __future__ import print_function
import caffe
from model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import subprocess
import sys

##should Convert the upper Convolution layer to Deconvolution Layer Manualy


### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.
caffe_root = "/home/caffemaker/detection/caffe-ssd/"

# Set true if you want to start training right after generating all files.
run_soon = False
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
resume_training = True
# If true, Remove old model files.
remove_old_models = False

train_data = "/home/caffemaker/caffe/dataset/blurred+sharp/train.txt"
train_gt = "/home/caffemaker/caffe/dataset/blurred+shap/train_gt.txt"
test_data = "/home/caffemaker/caffe/dataset/blurred+sharp/val.txt"
test_gt = "/home/caffemaker/caffe/dataset/blurred+sharp/val_gt.txt"
pretrain_model = ""

# Specify the batch sampler.
resize_width = 160
resize_height = 160
resize = "{}x{}".format(resize_width, resize_height)
train_transform_param = {
        'mean_value': [127.5, 127.5, 127.5],
        'scale': 0.0078125,
        'mirror': True}
test_transform_param = {
        'mean_value': [127.5, 127.5, 127.5],
        'scale': 0.0078125}

# If true, use batch norm for all newly added layers.
# Currently only the non batch norm version has been tested.
use_batchnorm = True
# Use different initial learning rate.
if use_batchnorm:
    base_lr = 0.04
else:
    # A learning rate for batch_size = 1, num_gpus = 1.
    base_lr = 0.0005

# Modify the job name if you want.
job_name = "DBN_{}".format(resize)
# The name of the model. Modify it if you want.
model_name = "DBN_{}".format(job_name)

# Directory which stores the model .prototxt file.
save_dir = "/home/caffemaker/caffe/models/DBN/{}".format(job_name)
# Directory which stores the snapshot of models.
snapshot_dir = "/home/caffemaker/caffe/jobs/Zface/{}".format(job_name)
# Directory which stores the job script and log file.
job_dir = "/home/caffemaker/caffe/jobs/Zface/{}".format(job_name)
# Directory which stores the detection results.
output_result_dir = "{}/caffemaker/caffe/jobs/Zface/{}/Main".format(os.environ['HOME'], job_name)

# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(job_dir, model_name)

# Solver parameters.
# Defining which GPUs to use.
gpus = "0"
gpulist = gpus.split(",")
num_gpus = len(gpulist)

# Divide the mini-batch to different GPUs.
batch_size = 32
accum_batch_size = 32
iter_size = accum_batch_size / batch_size
solver_mode = P.Solver.CPU
device_id = 0
batch_size_per_device = batch_size
if num_gpus > 0:
  batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
  iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
  solver_mode = P.Solver.GPU
  device_id = int(gpulist[0])

# Evaluate on whole test set.
num_test_image = 1800
test_batch_size = 1
test_iter = num_test_image / test_batch_size

solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.0005,
    'lr_policy': "step",
    'stepsize': 20000,
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': 60000,
    'snapshot': 1000,
    'display': 10,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': 1000,
    'test_initialization': False,
    }

### Hopefully you don't need to change the following ###
# Check file.
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

# Create train net.
net = caffe.NetSpec()

net.data = L.ImageData(source="/home/caffemaker/caffe/dataset/blurred+sharp/train.txt",
                       batch_size=16, transform_param = train_transform_param)
net.label = L.ImageData(source="/home/caffemaker/caffe/dataset/blurred+sharp/train_gt.txt",
                        batch_size=16, transform_param = train_transform_param)
DeBlurNetBody(net, from_layer='data', use_batchnorm=True)
net.loss = L.EuclideanLoss(net["flat_conv6_2"], net.label)

# Create the MultiBoxLossLayer.

with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

# Create test net.
net = caffe.NetSpec()

net.data = L.ImageData(source="/home/caffemaker/caffe/dataset/blurred+sharp/val.txt",
                        batch_size=1, transform_param = test_transform_param)
net.label = L.ImageData(source="/home/caffemaker/caffe/dataset/blurred+sharp/val_gt.txt",
                        batch_size=1, transform_param = test_transform_param)

DeBlurNetBody(net, from_layer='data', use_batchnorm=True)

net.loss = L.EuclideanLoss(net["flat_conv6_2"], net.label)

with open(test_net_file, 'w') as f:
    print('name: "{}_test"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

# Create deploy net.
# Remove the first and last layer from test net.
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-1]
    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
    print(net_param, file=f)

# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[test_net_file],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(solver_file, 'w') as f:
    print(solver, file=f)

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

if remove_old_models:
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))

# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  f.write(train_src_param)
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

# Copy the python script to job_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, job_dir)

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)
