#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:45:35 2017

@author: gonsalves-admin
"""

import tensorflow as tf

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
print(node1,node2)

sess = tf.Session()
print(sess.run([node1, node2]))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print(adder_node)

print(sess.run(adder_node, {a:[2,3],b:[4,4]}))

W = tf.Variable(.3,tf.float32)
W = tf.Variable([2.],tf.float32)
b = tf.Variable([2.],tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[1.,2.]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)o
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))