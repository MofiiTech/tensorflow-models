#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Filename : dcgan.py
# @Date : 2020-03-18
# @Author : Wufei Ma

import os
import sys

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tf.layers.dense(x, units=6*6*128)
        x = tf.nn.tanh(x)
        x = tf.reshape(x, shape=[-1, 6, 6, 128])

        # (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)

        # (batch, 28, 28, 1)
        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
        x = tf.nn.sigmoid(x)
    return x


def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tf.layers.conv2d(x, 64, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)

        x = tf.layers.conv2d(x, 128, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)

        x = tf.layers.dense(x, 2)
    return x


if __name__ == '__main__':

    # Load MNIST dataset
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

    # Training parameters
    num_steps = 20000
    batch_size = 32

    # Network parameters
    image_dim = 784
    gen_hidden_dim = 256
    disc_hidden_dim = 256
    latent_dim = 200
