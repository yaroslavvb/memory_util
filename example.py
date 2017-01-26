import tensorflow as tf
import sys, os, math, random

import memory_util
memory_util.vlog(1) 

import tensorflow as tf
import numpy as np

def create_session():
    config = tf.ConfigProto(log_device_placement=False,graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
    return tf.InteractiveSession(config=config)
    
node_mbs = 1
length = 5

dtype = np.float32
n = node_mbs * 250000
a0_ = tf.ones((n,), dtype=dtype)
a0 = tf.Variable(a0_, name="a0")
a = a0
for i in range(1, length):
    name = "a"+str(i)
    a = tf.tanh(a, name=name)

grad = tf.gradients([a], [a0])[0]
sess = create_session()

sess.run(tf.global_variables_initializer())

with memory_util.capture_stderr() as stderr:
    sess.run(grad.op)

peak_memory = memory_util.peak_memory(stderr)
memory_util.print_memory_timeline(stderr)
print("Peak memory: %d"%(peak_memory,))
