# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Tool for creating TFRecords datasets."""

import os
import numpy as np
import tensorflow as tf

#----------------------------------------------------------------------------

class TFRecordExporter:
    def __init__(self, tfrecord_dir, compressed=False):
        self.tfrecord_dir       = tfrecord_dir
        self.num_val_images     = 0
        self.tfr_prefix         = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.shape              = None
        self.resolution_log2    = None
        self.tfr_writer         = None
        self.compressed         = compressed

        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        self.tfr_writer.close()
        self.tfr_writer = None
    
    def set_shape(self, shape):
        self.shape = shape
        self.resolution_log2 = int(np.log2(self.shape[1]))
        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % self.resolution_log2
        self.tfr_writer = tf.python_io.TFRecordWriter(tfr_file, tfr_opt)
    
    def set_num_val_images(self, num_val_images):
        self.num_val_images = num_val_images

    def add_image(self, img):
        if self.shape is None:
            self.set_shape(img.shape)
        if not self.compressed:
            assert list(self.shape) == list(img.shape)
        quant = np.rint(img).clip(0, 255).astype(np.uint8) if not self.compressed else img
        ex = tf.train.Example(features=tf.train.Features(feature={
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
            'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()])),
            'compressed': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.compressed])),
            'num_val_images': tf.train.Feature(int64_list=tf.train.Int64List(value=[self.num_val_images])),
        }))
        self.tfr_writer.write(ex.SerializeToString())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
