# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Default metric definitions."""

from dnnlib import EasyDict

#----------------------------------------------------------------------------

metric_defaults = EasyDict([(args.name, args) for args in [
    EasyDict(name='fid200-rt-shoes', func_name='metrics.frechet_inception_distance.FID', num_images=200, minibatch_per_gpu=1, ref_train=True, ref_samples=49825),
    EasyDict(name='fid200-rt-handbags', func_name='metrics.frechet_inception_distance.FID', num_images=200, minibatch_per_gpu=1, ref_train=True, ref_samples=138567),
    EasyDict(name='fid5k',    func_name='metrics.frechet_inception_distance.FID', num_images=5000, minibatch_per_gpu=8),
    EasyDict(name='fid10k',    func_name='metrics.frechet_inception_distance.FID', num_images=10000, minibatch_per_gpu=8),
    EasyDict(name='fid10k-b1',    func_name='metrics.frechet_inception_distance.FID', num_images=10000, minibatch_per_gpu=1),
    EasyDict(name='fid10k-h0',    func_name='metrics.frechet_inception_distance.FID', num_images=10000, minibatch_per_gpu=8, hole_range=[0.0, 0.2]),
    EasyDict(name='fid10k-h1',    func_name='metrics.frechet_inception_distance.FID', num_images=10000, minibatch_per_gpu=8, hole_range=[0.2, 0.4]),
    EasyDict(name='fid10k-h2',    func_name='metrics.frechet_inception_distance.FID', num_images=10000, minibatch_per_gpu=8, hole_range=[0.4, 0.6]),
    EasyDict(name='fid10k-h3',    func_name='metrics.frechet_inception_distance.FID', num_images=10000, minibatch_per_gpu=8, hole_range=[0.6, 0.8]),
    EasyDict(name='fid10k-h4',    func_name='metrics.frechet_inception_distance.FID', num_images=10000, minibatch_per_gpu=8, hole_range=[0.8, 1.0]),
    EasyDict(name='fid36k5',   func_name='metrics.frechet_inception_distance.FID',num_images=36500, minibatch_per_gpu=8),
    EasyDict(name='fid36k5-h0',    func_name='metrics.frechet_inception_distance.FID', num_images=36500, minibatch_per_gpu=8, hole_range=[0.0, 0.2]),
    EasyDict(name='fid36k5-h1',    func_name='metrics.frechet_inception_distance.FID', num_images=36500, minibatch_per_gpu=8, hole_range=[0.2, 0.4]),
    EasyDict(name='fid36k5-h2',    func_name='metrics.frechet_inception_distance.FID', num_images=36500, minibatch_per_gpu=8, hole_range=[0.4, 0.6]),
    EasyDict(name='fid36k5-h3',    func_name='metrics.frechet_inception_distance.FID', num_images=36500, minibatch_per_gpu=8, hole_range=[0.6, 0.8]),
    EasyDict(name='fid36k5-h4',    func_name='metrics.frechet_inception_distance.FID', num_images=36500, minibatch_per_gpu=8, hole_range=[0.8, 1.0]),
    EasyDict(name='fid50k',    func_name='metrics.frechet_inception_distance.FID', num_images=50000, minibatch_per_gpu=8),
    EasyDict(name='ids5k',    func_name='metrics.inception_discriminator_score.IDS', num_images=5000, minibatch_per_gpu=8),
    EasyDict(name='ids10k',    func_name='metrics.inception_discriminative_score.IDS', num_images=10000, minibatch_per_gpu=8),
    EasyDict(name='ids10k-b1',    func_name='metrics.inception_discriminative_score.IDS', num_images=10000, minibatch_per_gpu=1),
    EasyDict(name='ids10k-h0',    func_name='metrics.inception_discriminative_score.IDS', num_images=10000, minibatch_per_gpu=8, hole_range=[0.0, 0.2]),
    EasyDict(name='ids10k-h1',    func_name='metrics.inception_discriminative_score.IDS', num_images=10000, minibatch_per_gpu=8, hole_range=[0.2, 0.4]),
    EasyDict(name='ids10k-h2',    func_name='metrics.inception_discriminative_score.IDS', num_images=10000, minibatch_per_gpu=8, hole_range=[0.4, 0.6]),
    EasyDict(name='ids10k-h3',    func_name='metrics.inception_discriminative_score.IDS', num_images=10000, minibatch_per_gpu=8, hole_range=[0.6, 0.8]),
    EasyDict(name='ids10k-h4',    func_name='metrics.inception_discriminative_score.IDS', num_images=10000, minibatch_per_gpu=8, hole_range=[0.8, 1.0]),
    EasyDict(name='ids36k5',   func_name='metrics.inception_discriminative_score.IDS',num_images=36500, minibatch_per_gpu=8),
    EasyDict(name='ids36k5-h0',    func_name='metrics.inception_discriminative_score.IDS', num_images=36500, minibatch_per_gpu=8, hole_range=[0.0, 0.2]),
    EasyDict(name='ids36k5-h1',    func_name='metrics.inception_discriminative_score.IDS', num_images=36500, minibatch_per_gpu=8, hole_range=[0.2, 0.4]),
    EasyDict(name='ids36k5-h2',    func_name='metrics.inception_discriminative_score.IDS', num_images=36500, minibatch_per_gpu=8, hole_range=[0.4, 0.6]),
    EasyDict(name='ids36k5-h3',    func_name='metrics.inception_discriminative_score.IDS', num_images=36500, minibatch_per_gpu=8, hole_range=[0.6, 0.8]),
    EasyDict(name='ids36k5-h4',    func_name='metrics.inception_discriminative_score.IDS', num_images=36500, minibatch_per_gpu=8, hole_range=[0.8, 1.0]),
    EasyDict(name='ids50k',    func_name='metrics.inception_discriminative_score.IDS', num_images=50000, minibatch_per_gpu=8),
    EasyDict(name='lpips2k',   func_name='metrics.learned_perceptual_image_patch_similarity.LPIPS', num_pairs=2000, minibatch_per_gpu=8),
]])

#----------------------------------------------------------------------------
