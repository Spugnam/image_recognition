#!/usr/local/bin//python3

import tensorflow as tf
import inception_preprocessing
from tensorflow.contrib import slim


def load_batch(dataset, batch_size=32, height=299, width=299,
               is_training=False):
    """Loads a single batch of data.
    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.

    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples
      that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image
      samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and
      dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32, common_queue_min=8)
    image_raw, label, filename = data_provider.get(['image', 'label', 'filename'])

    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(
        image_raw, height, width, is_training=is_training)

    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels, filename = tf.train.batch(
        [image, image_raw, label, filename],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)

    return images, images_raw, labels, filename
