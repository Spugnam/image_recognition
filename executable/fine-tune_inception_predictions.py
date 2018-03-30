#!/usr/local/bin//python3

"""
Apply fine-tuned inception model to own dataset
"""
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import popsy_dataset
from utils import load_batch
from inception_v4 import inception_v4, inception_v4_arg_scope

from tensorflow.contrib import slim

image_size = inception_v4.default_image_size
batch_size = 10

train_dir = '/tmp/inception_finetuned/'
popsy_dataset_dir = '../data/images'
log_dir = './log_dir'

with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    dataset = popsy_dataset.get_split('validation', popsy_dataset_dir)  # changed from train
    images, images_raw, labels, filenames = load_batch(
        dataset, height=image_size, width=image_size)

    # Use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception_v4_arg_scope()):
        logits, _ = inception_v4(
            images, num_classes=dataset.num_classes, is_training=True)

    probabilities = tf.nn.softmax(logits)

    checkpoint_path = tf.train.latest_checkpoint(train_dir)
    init_fn = slim.assign_from_checkpoint_fn(
        checkpoint_path, slim.get_variables_to_restore())

    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            sess.run(tf.initialize_local_variables())
            init_fn(sess)
            np_probabilities, np_images_raw, np_labels, np_filenames = sess.run(
                [probabilities, images_raw, labels, filenames])
            print("np_probabilities", np_probabilities)
            print("np_filenames", np_filenames)

            # manual accuracy TODO use slim.metrics
            accuracy = 0
            for i in range(batch_size):
                image = np_images_raw[i, :, :, :]
                true_label = np_labels[i]
                predicted_label = np.argmax(np_probabilities[i, :])
                # print("predicted_label", predicted_label)
                # print("labels_to_names", dataset.labels_to_names)
                predicted_name = dataset.labels_to_names[predicted_label]
                true_name = dataset.labels_to_names[true_label]
                image_title = np_filenames[i]
                if predicted_name == true_name:
                    accuracy += 1 / batch_size
                print('Title: [{}], Ground Truth: [{}], Prediction [{}]'.format(
                    image_title, true_name, predicted_name), end='\n')
                # plt.figure()
                # plt.imshow(image.astype(np.uint8))
                # plt.title('Ground Truth: [{}], Prediction [{}]'.format(
                #     true_name, predicted_name))
                # plt.axis('off')
                # plt.show()
            print("Total accuracy: {}", accuracy)
