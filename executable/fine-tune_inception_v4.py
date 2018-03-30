#!/usr/local/bin//python3

import os
import popsy_dataset
import tensorflow as tf
from utils import load_batch
from inception_v4 import inception_v4, inception_v4_arg_scope

from tensorflow.contrib import slim
image_size = inception_v4.default_image_size

checkpoints_dir = '/tmp/checkpoints'

train_dir = '/tmp/inception_finetuned/'
popsy_dataset_dir = '../data/images'

# # Preliminary Steps to download inception_v4 checkpoint
# import dataset_utils
#
# url = "http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz"
# checkpoints_dir = '/tmp/checkpoints'
#
# if not tf.gfile.Exists(checkpoints_dir):
#     tf.gfile.MakeDirs(checkpoints_dir)
#
# dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.
    """
    checkpoint_exclude_scopes = ["InceptionV4/Logits", "InceptionV4/AuxLogits"]

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
        variables_to_restore)


with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset = popsy_dataset.get_split('train', popsy_dataset_dir)
    images, _, labels, filename = load_batch(
        dataset, batch_size=512, height=image_size, width=image_size)

    # use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception_v4_arg_scope()):
        logits, _ = inception_v4(
            images, num_classes=dataset.num_classes, is_training=True)

    # Specify the loss function:
    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
    slim.losses.softmax_cross_entropy(logits, one_hot_labels)
    total_loss = slim.losses.get_total_loss()

    # Create some summaries to visualize the training process:
    tf.summary.scalar('losses/Total_Loss', total_loss)

    # Specify the optimizer and create the train op:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # Run the training:
    final_loss = slim.learning.train(
        train_op,
        logdir=train_dir,
        init_fn=get_init_fn(),
        number_of_steps=5)


print('Finished training. Last batch loss %f' % final_loss)
