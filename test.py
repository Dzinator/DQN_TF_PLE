import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add_node = a + b
add_triple = add_node * 3


W = tf.Variable([0.3], tf.float32)
fixW = tf.assign(W, [-1.])
b = tf.Variable([-0.3], tf.float32)
fixb = tf.assign(b, [1.])

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = W * x + b

squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)


#initialize variables
init = tf.global_variables_initializer()

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()

sess.run(init)

for i in tqdm(range(1000)):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})


print(sess.run([W, b]))