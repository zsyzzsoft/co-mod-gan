# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Multi-resolution input data pipeline."""

import os
import glob
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from .mask_generator import tf_mask_generator

#----------------------------------------------------------------------------
# Dataset class that loads data from tfrecords files.

class TFRecordDataset:
    def __init__(self,
        tfrecord_dir,               # Directory containing a collection of tfrecords files.
        resolution      = None,     # Dataset resolution, None = autodetect.
        label_file      = None,     # Relative path of the labels file, None = autodetect.
        max_label_size  = 0,        # 0 = no labels, 'full' = full labels, <int> = N first label components.
        max_images      = None,     # Maximum number of images to use, None = use all images.
        repeat          = True,     # Repeat dataset indefinitely?
        shuffle_mb      = 4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
        prefetch_mb     = 2048,     # Amount of data to prefetch (megabytes), 0 = disable prefetching.
        buffer_mb       = 256,      # Read buffer size (megabytes).
        num_threads     = 2,        # Number of concurrent threads.
        num_val_images  = 10000,
        compressed      = False):

        self.tfrecord_dir       = tfrecord_dir
        self.resolution         = None
        self.resolution_log2    = None
        self.shape              = []        # [channels, height, width]
        self.dtype              = 'uint8'
        self.dynamic_range      = [0, 255]
        self.label_file         = label_file
        self.label_size         = None      # components
        self.label_dtype        = None
        self.pix2pix            = False
        self._np_labels         = None
        self._tf_minibatch_in   = None
        self._tf_labels_var     = None
        self._tf_labels_dataset = None
        self._tf_datasets       = dict()
        self._tf_val_datasets   = dict()
        self._tf_iterator       = None
        self._tf_val_iterator   = None
        self._tf_init_ops       = dict()
        self._tf_val_init_ops   = dict()
        self._tf_minibatch_np   = None
        self._tf_minibatch_val_np  = None
        self._tf_masks_iterator_np = None
        self._cur_minibatch     = -1
        self._cur_lod           = -1
        self._hole_range        = -1

        # List tfrecords files and inspect their shapes.
        assert os.path.isdir(self.tfrecord_dir)
        tfr_files = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.tfrecords')))
        assert len(tfr_files) >= 1
        tfr_shapes = []
        for tfr_file in tfr_files:
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            for record in tf.python_io.tf_record_iterator(tfr_file, tfr_opt):
                ex = tf.train.Example()
                ex.ParseFromString(record)
                features = ex.features.feature
                if 'compressed' in features and features['compressed'].int64_list.value[0]:
                    compressed = True
                if 'num_val_images' in features:
                    num_val_images = features['num_val_images'].int64_list.value[0]
                tfr_shapes.append(features['shape'].int64_list.value)
                break

        # Determine shape and resolution.
        max_shape = max(tfr_shapes, key=np.prod)
        
        if max_shape[0] > 3:
            self.pix2pix = True
        
        self.resolution = resolution if resolution is not None else max_shape[1]
        self.resolution_log2 = int(np.log2(self.resolution))
        self.shape = [max_shape[0], self.resolution, self.resolution]
        tfr_lods = [self.resolution_log2 - int(np.log2(shape[1])) for shape in tfr_shapes]
        assert all(shape[0] == max_shape[0] for shape in tfr_shapes)
        assert all(shape[1] == shape[2] for shape in tfr_shapes)
        assert all(shape[1] == self.resolution // (2**lod) for shape, lod in zip(tfr_shapes, tfr_lods))

        # Autodetect label filename.
        if self.label_file is None:
            guess = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.labels')))
            if len(guess):
                self.label_file = guess[0]
        elif not os.path.isfile(self.label_file):
            guess = os.path.join(self.tfrecord_dir, self.label_file)
            if os.path.isfile(guess):
                self.label_file = guess

        # Load labels.
        assert max_label_size == 'full' or max_label_size >= 0
        self._np_labels = np.zeros([1<<30, 0], dtype=np.float32)
        if self.label_file is not None and max_label_size != 0:
            self._np_labels = np.load(self.label_file)
            assert self._np_labels.ndim == 2
        if max_label_size != 'full' and self._np_labels.shape[1] > max_label_size:
            self._np_labels = self._np_labels[:, :max_label_size]
        if max_images is not None and self._np_labels.shape[0] > max_images:
            self._np_labels = self._np_labels[:max_images]
        self.label_size = self._np_labels.shape[1]
        self.label_dtype = self._np_labels.dtype.name

        # Build TF expressions.
        with tf.name_scope('Dataset'), tf.device('/cpu:0'):
            self._tf_minibatch_in = tf.placeholder(tf.int64, name='minibatch_in', shape=[])
            self._tf_hole_range = tf.placeholder(tf.float32, name='hole_range', shape=[2])
            self._tf_labels_var = tflib.create_var_with_large_initial_value(self._np_labels, name='labels_var')
            self._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(self._tf_labels_var)
            for tfr_file, tfr_shape, tfr_lod in zip(tfr_files, tfr_shapes, tfr_lods):
                if tfr_lod < 0:
                    continue
                dset_raw = tf.data.TFRecordDataset(tfr_file, compression_type='', buffer_size=buffer_mb<<20)
                if max_images is not None:
                    dset_raw = dset_raw.take(max_images)
                for tf_datasets, dset in [(self._tf_val_datasets, dset_raw.take(num_val_images)), (self._tf_datasets, dset_raw.skip(num_val_images))]:
                    if compressed:
                        dset = dset.map(self.parse_and_decode_tfrecord_tf, num_parallel_calls=num_threads)
                    else:
                        dset = dset.map(self.parse_tfrecord_tf, num_parallel_calls=num_threads)
                    dset = tf.data.Dataset.zip((dset, self._tf_labels_dataset))
                    bytes_per_item = np.prod(tfr_shape) * np.dtype(self.dtype).itemsize
                    if shuffle_mb > 0:
                        dset = dset.shuffle(((shuffle_mb << 20) - 1) // bytes_per_item + 1)
                    if repeat:
                        dset = dset.repeat()
                    if prefetch_mb > 0:
                        dset = dset.prefetch(((prefetch_mb << 20) - 1) // bytes_per_item + 1)
                    dset = dset.batch(self._tf_minibatch_in)
                    tf_datasets[tfr_lod] = dset
            self._tf_iterator = tf.data.Iterator.from_structure(self._tf_datasets[0].output_types, self._tf_datasets[0].output_shapes)
            self._tf_init_ops = {lod: self._tf_iterator.make_initializer(dset) for lod, dset in self._tf_datasets.items()}
            self._tf_val_iterator = tf.data.Iterator.from_structure(self._tf_val_datasets[0].output_types, self._tf_val_datasets[0].output_shapes)
            self._tf_val_init_ops = {lod: self._tf_val_iterator.make_initializer(dset) for lod, dset in self._tf_val_datasets.items()}

            self._tf_masks_dataset = tf_mask_generator(self.resolution, self._tf_hole_range).batch(self._tf_minibatch_in).prefetch(64)
            self._tf_masks_iterator = self._tf_masks_dataset.make_initializable_iterator()

    def close(self):
        pass

    # Use the given minibatch size and level-of-detail for the data returned by get_minibatch_tf().
    def configure(self, minibatch_size, lod=0, hole_range=[0,1]):
        lod = int(np.floor(lod))
        assert minibatch_size >= 1 and lod in self._tf_datasets
        if self._cur_minibatch != minibatch_size or self._cur_lod != lod or (hole_range is not None and self._hole_range != hole_range):
            self._tf_init_ops[lod].run({self._tf_minibatch_in: minibatch_size})
            self._tf_val_init_ops[lod].run({self._tf_minibatch_in: minibatch_size})
            self._cur_minibatch = minibatch_size
            self._cur_lod = lod
            if hole_range is not None:
                self._tf_masks_iterator.initializer.run({self._tf_minibatch_in: minibatch_size, self._tf_hole_range: hole_range})
                self._hole_range = hole_range

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self): # => images, labels
        return self._tf_iterator.get_next()
        
    def get_minibatch_val_tf(self): # => images, labels
        return self._tf_val_iterator.get_next()

    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self, minibatch_size, lod=0): # => images, labels
        self.configure(minibatch_size, lod)
        with tf.name_scope('Dataset'):
            if self._tf_minibatch_np is None:
                self._tf_minibatch_np = self.get_minibatch_tf()
            return tflib.run(self._tf_minibatch_np)
            
    def get_minibatch_val_np(self, minibatch_size, lod=0): # => images, labels
        self.configure(minibatch_size, lod)
        with tf.name_scope('Dataset'):
            if self._tf_minibatch_val_np is None:
                self._tf_minibatch_val_np = self.get_minibatch_val_tf()
            return tflib.run(self._tf_minibatch_val_np)

    # Get next minibatch as TensorFlow expressions.
    def get_random_masks_tf(self): # => images, labels
        return self._tf_masks_iterator.get_next()

    # Get next minibatch as NumPy arrays.
    def get_random_masks_np(self, minibatch_size, hole_range=[0,1]):
        self.configure(minibatch_size, hole_range=hole_range)
        with tf.name_scope('Dataset'):
            if self._tf_masks_iterator_np is None:
                self._tf_masks_iterator_np = self.get_random_masks_tf()
            return tflib.run(self._tf_masks_iterator_np)

    # Get random labels as TensorFlow expression.
    def get_random_labels_tf(self, minibatch_size): # => labels
        with tf.name_scope('Dataset'):
            if self.label_size > 0:
                with tf.device('/cpu:0'):
                    return tf.gather(self._tf_labels_var, tf.random_uniform([minibatch_size], 0, self._np_labels.shape[0], dtype=tf.int32))
            return tf.zeros([minibatch_size, 0], self.label_dtype)

    # Get random labels as NumPy array.
    def get_random_labels_np(self, minibatch_size): # => labels
        if self.label_size > 0:
            return self._np_labels[np.random.randint(self._np_labels.shape[0], size=[minibatch_size])]
        return np.zeros([minibatch_size, 0], self.label_dtype)

    # Parse individual image from a tfrecords file into TensorFlow expression.
    @staticmethod
    def parse_tfrecord_tf(record):
        features = tf.parse_single_example(record, features={
            'shape': tf.FixedLenFeature([3], tf.int64),
            'data': tf.FixedLenFeature([], tf.string)})
        data = tf.decode_raw(features['data'], tf.uint8)
        return tf.reshape(data, features['shape'])
        
    @staticmethod
    def parse_and_decode_tfrecord_tf(record):
        features = tf.parse_single_example(record, features={
            'shape': tf.FixedLenFeature([3], tf.int64),
            'data': tf.FixedLenFeature([], tf.string)})
        shape = tf.cast(features['shape'], 'int32')
        data = tf.image.decode_image(features['data'])
        data = tf.image.resize_with_crop_or_pad(data, shape[1], shape[2])
        return tf.broadcast_to(tf.transpose(data, [2, 0, 1]), shape)

#----------------------------------------------------------------------------
# Helper func for constructing a dataset object using the given options.

def load_dataset(class_name=None, data_dir=None, verbose=False, **kwargs):
    kwargs = dict(kwargs)
    if 'tfrecord_dir' in kwargs:
        if class_name is None:
            class_name = __name__ + '.TFRecordDataset'
        if data_dir is not None:
            kwargs['tfrecord_dir'] = os.path.join(data_dir, kwargs['tfrecord_dir'])

    assert class_name is not None
    if verbose:
        print('Streaming data using %s...' % class_name)
    dataset = dnnlib.util.get_obj_by_name(class_name)(**kwargs)
    if verbose:
        print('Dataset shape =', np.int32(dataset.shape).tolist())
        print('Dynamic range =', dataset.dynamic_range)
        print('Label size    =', dataset.label_size)
    return dataset

#----------------------------------------------------------------------------
