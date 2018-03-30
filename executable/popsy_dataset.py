"""Provides data for the popsy dataset (original flowers.py)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

# sys.path.append(r"/Users/Quentin/Documents/Tensorflow_models/research/slim/")  # noqa
import dataset_utils

slim = tf.contrib.slim

# _FILE_PATTERN = 'popsy_%s_*.tfrecord'

# SPLITS_TO_SIZES = {'train': 1840, 'validation': 460}

# _NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
    'filename': 'Image filename (title)',
}


def _get_num_classes(dataset_dir, file_pattern_for_counting):
    """
    Calculates the number of classes from dataset_dir sub_folders
    Is there a way to get from tfrecords like _get_num_samples? (cleaner)
    """
    root_folder = os.path.join(dataset_dir, file_pattern_for_counting)
    dataset_main_folder_list =\
        [name for name in os.listdir(root_folder)
         if os.path.isdir(os.path.join(root_folder, name))]
    return(len(dataset_main_folder_list))


def _get_num_samples(split_name, dataset_dir, file_pattern,
                     file_pattern_for_counting):
    """
    Returns num_sample from the tfrecord_files
    """
    num_samples = 0

    # Count the total number of examples in all of the shards
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file)
                          for file in os.listdir(dataset_dir)
                          if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1
    return(num_samples)


def get_split(split_name, dataset_dir, file_pattern='popsy_%s_*.tfrecord',
              reader=None, file_pattern_for_counting='popsy'):
    """Gets a dataset tuple with instructions for reading images.

    Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
    It is assumed that the pattern contains a '%s' string so that the split
    name can be inserted.
    reader: The TensorFlow reader type.

    Returns:
    A `Dataset` namedtuple.

    Raises:
    ValueError: if `split_name` is not a valid train/validation split.
    """
    if split_name not in ['train', 'validation']:
        raise ValueError('split name %s was not recognized.' % split_name)

    # Full path to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    # count samples
    num_samples = _get_num_samples(split_name, dataset_dir, file_pattern,
                                   file_pattern_for_counting)

    # Count number of classes
    num_classes = _get_num_classes(dataset_dir, file_pattern_for_counting)
    print("num_classes calculated:", num_classes)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
    }

# from https://stackoverflow.com/questions/42509811/gentlely-way-to-read-tfrecords-data-into-batches
# example = tf.train.Example(features=tf.train.Features(feature={
#    'image/height': _int64_feature(height),
#    'image/width': _int64_feature(width),
#    'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
#    'image/channels': _int64_feature(channels),
#    'image/class/label': _int64_feature(label),
#    'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
#    'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
#    'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
#    'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        # from ../slim/data/tfexample_decoder.py in Tensor() class:
        # 'image/class/label' = tensor_key: the name of the `TFExample` feature to read the tensor from.
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern_path,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=num_classes,
        labels_to_names=labels_to_names)
