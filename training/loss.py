# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Loss functions."""

import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

def G_masked_logistic_ns_l1(G, D, opt, training_set, minibatch_size, reals, masks, l1_weight=0):
    _ = opt
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, reals, masks, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, masks, is_training=True)
    logistic_loss = tf.nn.softplus(-fake_scores_out) # -log(sigmoid(fake_scores_out))
    logistic_loss = autosummary('Loss/logistic_loss', logistic_loss)
    l1_loss = tf.reduce_mean(tf.abs(fake_images_out - reals), axis=[1,2,3])
    l1_loss = autosummary('Loss/l1_loss', l1_loss)
    loss = logistic_loss + l1_loss * l1_weight
    return loss, None

def D_masked_logistic_r1(G, D, opt, training_set, minibatch_size, reals, labels, masks, gamma=10.0):
    _ = opt, training_set
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, reals, masks, is_training=True)
    real_scores_out = D.get_output_for(reals, labels, masks, is_training=True)
    fake_scores_out = D.get_output_for(fake_images_out, labels, masks, is_training=True)
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out) # -log(1-sigmoid(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out) # -log(sigmoid(real_scores_out))

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores_out), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        reg = gradient_penalty * (gamma * 0.5)
    return loss, reg

#----------------------------------------------------------------------------
