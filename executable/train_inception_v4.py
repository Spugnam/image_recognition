#!/usr/local/bin//python3
"""
Train last layer of inception_v4 model
"""

import os
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables\
        import get_or_create_global_step
import popsy_dataset
import dataset_utils
from utils import load_batch
from inception_v4 import inception_v4, inception_v4_arg_scope

image_size = inception_v4.default_image_size

checkpoints_dir = '/tmp/checkpoints'
inception_v4_model_path = '/tmp/checkpoints/inception_v4.ckpt'

train_dir = '/tmp/inception_finetuned/'
dirname = os.path.dirname
ROOT_DIR = dirname(dirname(os.path.abspath(__file__)))
popsy_dataset_dir = os.path.join(ROOT_DIR, "data/images")
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "tf_logs/" + "run-" + now

# Training Parameters
num_epochs = 2
batch_size = 8
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2


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


def run():
    # Preliminary Steps to download inception_v4 checkpoint
    url = "http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz"
    checkpoints_dir = '/tmp/checkpoints'
    if not tf.gfile.Exists(checkpoints_dir):
        tf.gfile.MakeDirs(checkpoints_dir)
    if not tf.gfile.Exists(inception_v4_model_path):
        dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

    # train model
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        dataset = popsy_dataset.get_split('train', popsy_dataset_dir)
        images, _, labels, filename = load_batch(dataset,
                                                 batch_size=batch_size,
                                                 height=image_size,
                                                 width=image_size)

        # Steps to take before decaying learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # one step = one batch
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        # use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v4_arg_scope()):
            logits, end_points = inception_v4(
                images, num_classes=dataset.num_classes, is_training=True)

        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Define your exponentially decaying learning rate
        learning_rate = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total_Loss', total_loss)

        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Create train_op
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Specify metrics
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, update_op_acc =\
            tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(update_op_acc, probabilities)

        # Create summaries
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', learning_rate)
        my_summary_op = tf.summary.merge_all()

        # Do not run the summary_op automatically (too much memory)
        sv = tf.train.Supervisor(logdir=train_dir,
                                 summary_op=None,
                                 init_fn=get_init_fn())

        # Run the managed session
        with sv.managed_session() as sess:
            for step in range(num_steps_per_epoch * num_epochs):
                print('Accuracy after batch {}: {}'.format(
                    step, update_op_acc.eval(session=sess)))
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch {}/{}'.format(
                        step/num_batches_per_epoch + 1, num_epochs))
                    learning_rate_value, accuracy_value = sess.run([
                        learning_rate, accuracy])
                    logging.info('Learning Rate: {}'.format(learning_rate_value))
                    logging.info('Accuracy: {}'.format(accuracy_value))

                # Training step
                start_time = time.time()
                total_loss, global_step_count, accuracy_value = sess.run(
                        [train_op,
                         global_step,
                         update_op_acc])
                time_elapsed = time.time() - start_time

                logging.info('global step: {}\tloss: {:.4f}\taccuracy: {:.2%}\telapsed time: {:.2f} sec/step)'.format(global_step_count,
                             total_loss,
                             accuracy_value,
                             time_elapsed))

                # Log the summaries every 10 step.
                if step % 10 == 0:
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

            # Log the final training loss and accuracy
            # total_loss, accuracy = sess.run([train_op, accuracy)])
            logging.info('Final Loss: {}'.format(total_loss))
            logging.info('Final Accuracy: {}'.format(sess.run(accuracy)))

            # Save final model
            logging.info('Finished training! Saving model to disk now.')
            # saver.save(sess, "./flowers_model.ckpt")


if __name__ == '__main__':
    run()
