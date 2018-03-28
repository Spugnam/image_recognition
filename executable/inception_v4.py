#!/usr/local/bin//python3

import sys
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

sys.path.append(r"/Users/Quentin/Documents/Tensorflow_models/research/slim")  # noqa
from datasets import dataset_utils
import datasets
# from tensorflow.contrib.slim import datasets

# from datasets import inception_v4tils
from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing
from tensorflow.contrib import slim

inception_v4 =\
    "http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz"
checkpoints_dir = '/tmp/checkpoints'

if not os.path.isfile("/tmp/checkpoints/inception_v4.ckpt"):
    if not tf.gfile.Exists(checkpoints_dir):
        tf.gfile.MakeDirs(checkpoints_dir)
    dataset_utils.download_and_uncompress_tarball(inception_v4, checkpoints_dir)


image_size = inception.inception_v4.default_image_size  # 299

with tf.Graph().as_default():
    url = "https://lh3.googleusercontent.com/erfxERAP-fBPlU69XogrSNvdR-prbQvnffZleXH7G-Qmf4COq_KBKjnEa3W6cCd_GmwDqX8VeAdkwoc2FbPs=s500-e365-nu?bk=FGx4i0EUTcXjnhd0s0M4MNYeQ7MTwiwLqfG7sYv8nLjKA66YkJyqkNd0pNqPrs%2BmzI3SPE5mjqvozwUfqTxT0Sr8QE5feFU9JZH5T53OckyAe2hLZ3oU8XO1b1a%2BvNwCHdo0vLhw4kqkwehJMbVLvTS1pBhAtOdeEu5OyRf4s8KCR8qOEBhrXewNkyS742jUDBmMv7ht2puk74HgFIBcboohO2agCYJLe4tdZOHpWaCQkVVD0vEAWGDDY2nrE4HDeJSx4yR%2FIB0%2BjlZHKKv52N%2BC4fEsqAjdkHzX9tjCcqobMumVw4fmAj64ImOmR4b2nWiO3TtpLc01sH%2BVn%2FIk%2FQXDUv6pKUytzipnHQA7LTCtv%2F9R9PNPvvBK8XWA%2BXAI7T2YjoJ3SzYlnGgtDHO0nnBeGcfyVegralByU9udzx%2FD6JR8JRDwKYbnMMZMLtAmYP0IrO021p3J23VKxGv0oOcr9%2BLUAAP8pZ%2F4UT9tSNaOMU%2Bjb18LRtf%2Fy2yYvxICKgdoSvZa6ZRWa1cnFMswOA%3D%3D"  # noqa
    image_string = urllib.urlopen(url).read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(
        image, image_size, image_size, is_training=False)
    processed_images = tf.expand_dims(processed_image, 0)

    # Use the default arg scope to configure the batch norm parameters
    with slim.arg_scope(inception.inception_v4_arg_scope()):
        logits, _ = inception.inception_v4(processed_images, num_classes=1001,
                                           is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
        slim.get_model_variables('InceptionV4'))

    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                            key=lambda x:x[1])]

    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.axis('off')
    plt.show()

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' %
              (probabilities[index] * 100, names[index]))
