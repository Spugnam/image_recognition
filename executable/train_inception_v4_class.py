#!/usr/bin/env python3

import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables\
        import get_or_create_global_step
from inception_v4 import inception_v4, inception_v4_arg_scope
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

import popsy_dataset
import dataset_utils
from utils import load_batch

# directories
dirname = os.path.dirname
ROOT_DIR = dirname(dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "data/images")
model_url =\
    "http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz"
model_path = '/tmp/checkpoints/inception_v4.ckpt'
train_dir = '/tmp/inception_finetuned/'


class Inception_v4_Classifier():
    def __init__(self,
                 model_url,
                 num_epochs,
                 initial_learning_rate=0.001,
                 learning_rate_decay_factor=0.7,
                 batch_size=100,
                 image_size=inception_v4.default_image_size,
                 random_state=None):
        """Initialize model by storing all the hyperparameters.
        """
        self.model_url = model_url
        self.num_epochs = num_epochs
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.batch_size = batch_size
        self.image_size = image_size
        self.random_state = random_state
        self._session = None
        self._load_inception_v4(model_path)

    def _load_inception_v4(self, model_url):
        """Load inception_v4 checkpoint if necessary
        """
        checkpoints_dir = '/tmp/checkpoints'
        if not tf.gfile.Exists(checkpoints_dir):
            tf.gfile.MakeDirs(checkpoints_dir)

        model_name = os.path.basename(model_url).split('.')[0] + '.ckpt'
        model_path = os.path.join(checkpoints_dir, model_name)
        if not tf.gfile.Exists(model_path):
            dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

    def get_init_fn(self):
        """Returns a function run by the chief worker to warm-start the training.
        """
        checkpoint_exclude_scopes = ["InceptionV4/Logits",
                                     "InceptionV4/AuxLogits"]
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

    def _build_graph(self):
        """Called by fit method
        Saves parameters for access by other methods
        """
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        dataset = popsy_dataset.get_split('train', DATASET_DIR)
        images, _, labels, filename = load_batch(dataset,
                                                 batch_size=self.batch_size,
                                                 height=self.image_size,
                                                 width=self.mage_size)
        # use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v4_arg_scope()):
            logits, end_points = inception_v4(
                images, num_classes=dataset.num_classes, is_training=True)

        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()


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

        # init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self._saver = saver

        # create instance variables
        self.dataset = dataset

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_params(self):
        """Get all variable values (used for early stopping, faster than saving
        to disk)
        """
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in
                zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        """Set all variables to the given values (for early stopping,
        faster than loading from disk)
        """
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(
            gvar_name + "/Assign") for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1]
                       for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name]
                     for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def fit(self, images, labels, n_epochs=100):
        """Fit the model to the training set. If X_valid and y_valid are
        provided, use early stopping.
        """
        self.close_session()

        # Now train the model
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()

        # Steps to take before decaying learning rate and batches per epoch
        num_batches_per_epoch = int(self.dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # one step = one batch
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Define your exponentially decaying learning rate
        learning_rate = tf.train.exponential_decay(
            learning_rate=self.initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=self.learning_rate_decay_factor,
            staircase=True)

        # needed in case of early stopping
        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None

        # logging
        # program description (first argument)
        if len(sys.argv) > 1:
            run_description = sys.argv[1]
        else:
            run_description = ""
            print("No log description entered")
            now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            logdir = "tf_logs/mnist_dnn-" + run_description + "run-" + now
        # tensorboard logs
        file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

        # recover intermediate runs
        checkpoint_path = "./tmp/my_pretrained_inception_model.ckpt"
        checkpoint_epoch_path = checkpoint_path + ".epoch"
        final_model_path = "./my_pretrained_inception_model"

        # Do not run the summary_op automatically (too much memory)
        sv = tf.train.Supervisor(logdir=train_dir,
                                 summary_op=None,
                                 init_fn=self.get_init_fn())

        # Run the managed session
        with sv.managed_session() as sess:
            if os.path.isfile(checkpoint_epoch_path):
                # if checkpoint file exists, restore model and load epoch
                with open(checkpoint_epoch_path, "rb") as f:
                    start_epoch = int(f.read())
                print("Training interrupted. Continuing at epoch", start_epoch)
                self._saver.restore(sess, checkpoint_path)
            else:
                start_epoch = 0
                self._init.run()

            for epoch in (self.num_epoch):
                for step in range(num_steps_per_epoch * self.num_epochs):
                    # Training step
                    start_time = time.time()
                    total_loss, global_step_count, accuracy_value = sess.run(
                            [train_op,
                             global_step,
                             update_op_acc])
                    time_elapsed = time.time() - start_time

                    logging.info('global step: {}\tloss: {:.4f}\
                                 \taccuracy: {:.2%}\
                                 \telapsed time: {:.2f} sec/step)'
                                 .format(global_step_count,
                                         total_loss,
                                         accuracy_value,
                                         time_elapsed))
                summaries = sess.run(my_summary_op)
                sv.summary_computed(sess, summaries)
                # save model
                self._saver.save(sess, checkpoint_path)

                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))
                if total_loss < best_loss:
                    self._saver.save(sess, final_model_path)
                    best_params = self._get_model_params()
                    best_loss = loss_val
                    checks_without_progress = 0
                else:
                    checks_without_progress += 1

                if checks_without_progress > max_checks_without_progress:
                    print("Early stopping!")
                    break

            # If we used early stopping then rollback to the best model found
            if best_params:
                self._restore_model_params(best_params)
            os.remove(checkpoint_epoch_path)  # remove intermediate run
            return self

    # def predict_proba(self, X):
    #     if not self._session:
    #         raise NotFittedError("This %s instance is not fitted yet" %
    #                              self.__class__.__name__)
    #     with self._session.as_default() as sess:
    #         return self._Y_proba.eval(feed_dict={self._X: X})

    # def predict(self, X):
    #     class_indices = np.argmax(self.predict_proba(X), axis=1)
    #     return np.array([[self.classes_[class_index]]
    #                      for class_index in class_indices], np.int32)

    def save(self, path):
        self._saver.save(self._session, path)


if __name__ == "__main__":
    # load data?

    # single run
    clf = Inception_v4_Classifier(model_url,
                                  initial_learning_rate=0.001,
                                  batch_size=100,
                                  image_size=inception_v4.default_image_size,
                                  random_state=None)

    clf.fit(images, labels, n_epochs=2)
