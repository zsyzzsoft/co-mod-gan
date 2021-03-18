# Large Scale Image Completion via Co-Modulated Generative Adversarial Networks
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://openreview.net/pdf?id=sSjqmfsk95O

"""Paired/Unpaired Inception Discriminative Score (P-IDS/U-IDS)."""

import os
from tqdm import tqdm
import numpy as np
import scipy
import sklearn.svm
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------

class IDS(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, hole_range=[0,1], **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu
        self.hole_range = hole_range

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        inception = misc.load_pkl('https://drive.google.com/uc?id=1MzTY44rLToO5APn8TZmfR7_ENSe5aZUn') # inception_v3_features.pkl
        real_activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)
        fake_activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)

        # Construct TensorFlow graph.
        self._configure(self.minibatch_per_gpu, hole_range=self.hole_range)
        real_img_expr = []
        fake_img_expr = []
        real_result_expr = []
        fake_result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                inception_clone = inception.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                reals, labels = self._get_minibatch_tf()
                reals_tf = tflib.convert_images_from_uint8(reals)
                masks = self._get_random_masks_tf()
                fakes = Gs_clone.get_output_for(latents, labels, reals_tf, masks, **Gs_kwargs)
                fakes = tflib.convert_images_to_uint8(fakes[:, :3])
                reals = tflib.convert_images_to_uint8(reals_tf[:, :3])
                real_img_expr.append(reals)
                fake_img_expr.append(fakes)
                real_result_expr.append(inception_clone.get_output_for(reals))
                fake_result_expr.append(inception_clone.get_output_for(fakes))

        for begin in tqdm(range(0, self.num_images, minibatch_size)):
            self._report_progress(begin, self.num_images)
            end = min(begin + minibatch_size, self.num_images)
            real_results, fake_results = tflib.run([real_result_expr, fake_result_expr])
            real_activations[begin:end] = np.concatenate(real_results, axis=0)[:end-begin]
            fake_activations[begin:end] = np.concatenate(fake_results, axis=0)[:end-begin]
        
        # Calculate FID conviniently.
        mu_real = np.mean(real_activations, axis=0)
        sigma_real = np.cov(real_activations, rowvar=False)
        mu_fake = np.mean(fake_activations, axis=0)
        sigma_fake = np.cov(fake_activations, rowvar=False)
        m = np.square(mu_fake - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
        dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        self._report_result(np.real(dist), suffix='-FID')
        
        svm = sklearn.svm.LinearSVC(dual=False)
        svm_inputs = np.concatenate([real_activations, fake_activations])
        svm_targets = np.array([1] * real_activations.shape[0] + [0] * fake_activations.shape[0])
        svm.fit(svm_inputs, svm_targets)
        self._report_result(1 - svm.score(svm_inputs, svm_targets), suffix='-U')
        real_outputs = svm.decision_function(real_activations)
        fake_outputs = svm.decision_function(fake_activations)
        self._report_result(np.mean(fake_outputs > real_outputs), suffix='-P')

#----------------------------------------------------------------------------
