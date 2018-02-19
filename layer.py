import numpy as np
import tensorflow as tf

class Layers:
	def autoenc(self, x):
		with tf.variable_scope('AutoEncoder'):

			#Convolution 1
			c1 = tf.layers.conv2d(x, 16, 3, strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer())
			r1 = tf.nn.relu(c1)

			#Convolution 2
			c2 = tf.layers.conv2d(r1, 32, 5, strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer())
			r2 = tf.nn.relu(c2)

			#Convolution 3
			c3 = tf.layers.conv2d(r2, 64, 4, strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer())
			r3 = tf.nn.relu(c3)

			#Deconvolution 1
			d1 = tf.layers.conv2d_transpose(r3, 32, 4, strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer())
			rd1 = tf.nn.relu(d1)

			#Deconvolution 2
			d2 = tf.layers.conv2d_transpose(rd1, 16, 5, strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer())
			rd2 = tf.nn.relu(d2)

			#Deconvolution 3
			d3 = tf.layers.conv2d_transpose(rd2, 3, 3, strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer())
			rd3 = tf.nn.relu(d3)

			return rd3
