# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import copy
import os
import sys

import dnnlib
from dnnlib import EasyDict

from metrics.metric_defaults import metric_defaults

#----------------------------------------------------------------------------

def run(dataset, data_dir, result_dir, num_gpus, total_kimg, mirror_augment, metrics, resume, resume_with_new_nets, disable_style_mod, disable_cond_mod):
    
    train     = EasyDict(run_func_name='training.training_loop.training_loop') # Options for training loop.
    G         = EasyDict(func_name='training.co_mod_gan.G_main')       # Options for generator network.
    D         = EasyDict(func_name='training.co_mod_gan.D_co_mod_gan')  # Options for discriminator network.
    G_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for generator optimizer.
    D_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for discriminator optimizer.
    G_loss    = EasyDict(func_name='training.loss.G_masked_logistic_ns_l1')      # Options for generator loss.
    D_loss    = EasyDict(func_name='training.loss.D_masked_logistic_r1')              # Options for discriminator loss.
    sched     = EasyDict()                                                     # Options for TrainingSchedule.
    grid      = EasyDict(size='8k', layout='random')                           # Options for setup_snapshot_image_grid().
    sc        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000}                                   # Options for tflib.init_tf().

    train.data_dir = data_dir
    train.total_kimg = total_kimg
    train.mirror_augment = mirror_augment
    train.image_snapshot_ticks = train.network_snapshot_ticks = 10
    sched.G_lrate_base = sched.D_lrate_base = 0.002
    sched.minibatch_size_base = 32
    sched.minibatch_gpu_base = 4
    D_loss.gamma = 10
    metrics = [metric_defaults[x] for x in metrics]
    desc = 'co-mod-gan'

    desc += '-' + os.path.basename(dataset)
    dataset_args = EasyDict(tfrecord_dir=dataset)

    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus
    
    if resume is not None:
        resume_kimg = int(os.path.basename(resume).replace('.pkl', '').split('-')[-1])
    else:
        resume_kimg = 0
    
    if disable_style_mod:
        G.style_mod = False

    if disable_cond_mod:
        G.cond_mod = False

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
    kwargs.update(resume_pkl=resume, resume_kimg=resume_kimg, resume_with_new_nets=resume_with_new_nets)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_dir_root = result_dir
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)

#----------------------------------------------------------------------------

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

_examples = '''examples:

  # Train CoModGAN using the FFHQ dataset
  python %(prog)s --data-dir=~/datasets --dataset=ffhq --metrics=ids10k --num-gpus=8

'''

def main():
    parser = argparse.ArgumentParser(
        description='Train CoModGAN.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    parser.add_argument('--dataset', help='Training dataset', required=True)
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)', default=1, type=int, metavar='N')
    parser.add_argument('--total-kimg', help='Training length in thousands of images (default: %(default)s)', metavar='KIMG', default=25000, type=int)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)', default=False, metavar='BOOL', type=_str_to_bool)
    parser.add_argument('--metrics', help='Comma-separated list of metrics or "none" (default: %(default)s)', default='ids10k', type=_parse_comma_sep)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--resume-with-new-nets', default=False, action='store_true')
    parser.add_argument('--disable-style-mod', default=False, action='store_true')
    parser.add_argument('--disable-cond-mod', default=False, action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print ('Error: dataset root directory does not exist.')
        sys.exit(1)

    for metric in args.metrics:
        if metric not in metric_defaults:
            print ('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)

    run(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

