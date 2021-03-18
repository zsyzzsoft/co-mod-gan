"""Learned Perceptual Image Patch Similarity (LPIPS)."""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc


#----------------------------------------------------------------------------

class LPIPS(metric_base.MetricBase):
    def __init__(self, num_pairs=2000, minibatch_per_gpu=8, **kwargs):
        super().__init__(**kwargs)
        self.num_pairs = num_pairs
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu

        graph_def = tf.GraphDef()
        with misc.open_file_or_url('http://rail.eecs.berkeley.edu/models/lpips/net-lin_alex_v0.1.pb') as f:
            graph_def.ParseFromString(f.read())
        
        # Construct TensorFlow graph.
        self._configure(self.minibatch_per_gpu)
        result_expr = []
        for gpu_idx in range(num_gpus):
            def auto_gpu(opr):
                if opr.type in ['SparseToDense', 'Tile', 'GatherV2', 'Pack']:
                    return '/cpu:0'
                else:
                    return '/gpu:%d' % gpu_idx
            with tf.device(auto_gpu):
                Gs_clone = Gs.clone()
                reals, labels = self._get_minibatch_tf()
                reals = tflib.convert_images_from_uint8(reals)
                masks = self._get_random_masks_tf()
                
                latents0 = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                fakes0 = Gs_clone.get_output_for(latents0, labels, reals, masks, **Gs_kwargs)[:, :3, :, :]
                fakes0 = tf.clip_by_value(fakes0, -1.0, 1.0)
                
                latents1 = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                fakes1 = Gs_clone.get_output_for(latents1, labels, reals, masks, **Gs_kwargs)[:, :3, :, :]
                fakes1 = tf.clip_by_value(fakes1, -1.0, 1.0)
                
                distance,  = tf.import_graph_def(
                    graph_def,
                    input_map={'0:0': fakes0, '1:0': fakes1},
                    return_elements = ['Reshape_10']
                )
                result_expr.append(distance.outputs)

        # Run metric
        results = []
        for begin in range(0, self.num_pairs, minibatch_size):
            self._report_progress(begin, self.num_pairs)
            res = tflib.run(result_expr)
            results.append(np.reshape(res, [-1]))
        results = np.concatenate(results)
        self._report_result(np.mean(results))
        self._report_result(np.std(results), suffix='-var')

#----------------------------------------------------------------------------
