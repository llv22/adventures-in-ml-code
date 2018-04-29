#!/usr/bin/env python -W ignore::DeprecationWarning
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

## 1. simple setup of default tensorflow session
# tf_config = tf.ConfigProto(log_device_placement=True)
# tf_config.gpu_options.allow_growth = True
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1
# sess = tf.Session(config=tf_config)

## 2. more strategy setup of default tensorflow session
num_cores = 4
GPU = len([v for v in os.environ["CUDA_VISIBLE_DEVICES"].split(',') if len(v)>0])

if GPU > 0:
    num_GPU = 1
    num_CPU = 4
else:
    num_GPU = 0
    num_CPU = 4

tf_config = tf.ConfigProto(
        intra_op_parallelism_threads=num_cores,
        inter_op_parallelism_threads=num_cores, \
        allow_soft_placement=True, \
        # log_device_placement=True, \
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU}
    )
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9

import numpy as np 
import tensorflow as tf 
import datetime 

log_device_placement = True 

n = 10 

A = np.random.rand(10000, 10000).astype('float32') 
B = np.random.rand(10000, 10000).astype('float32') 


c1 = [] 
c2 = [] 

def matpow(M, n): 
    if n < 1: #Abstract cases where n < 1 
        return M 
    else: 
        return tf.matmul(M, matpow(M, n-1)) 

with tf.device('/gpu:0'): 
    a = tf.placeholder(tf.float32, [10000, 10000]) 
    b = tf.placeholder(tf.float32, [10000, 10000]) 
    c1.append(matpow(a, n)) 
    c1.append(matpow(b, n)) 

with tf.device('/cpu:0'): 
  sum = tf.add_n(c1) #Addition of all elements in c1, i.e. A^n + B^n 

t1_1 = datetime.datetime.now() 
with tf.Session(config=tf_config) as sess: 
     print(sess.run(sum, {a:A, b:B}))
t2_1 = datetime.datetime.now()