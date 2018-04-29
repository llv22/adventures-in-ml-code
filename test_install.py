
#!/usr/bin/env python -W ignore::DeprecationWarning
'''This example demonstrates the use of test_install to check if GPU could be used.

References:
    1. tensorflow_cuda_osx.md - https://gist.github.com/Mistobaan/dd32287eeb6859c6668d
'''
import tensorflow as tf

with tf.device('/gpu:0'):
    # Creates a graph.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

# Creates a session with log_device_placement set to True.
tf_config = tf.ConfigProto(
    log_device_placement=True, \
    allow_soft_placement=True, \
)
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=tf_config)

# Runs the op.
print(sess.run(c))