#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.util import compat


def _decode_image_file(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string, channels=1)
  #image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_decoded, label


def main(args):
  input_feature_size = 28 * 28
  output_label_size = 10
  learning_rate = 0.1
  epoch_number = 10
  batch_size = 3

  # Get the train images and labels
  image_list = [
      line.strip() for line in open("train_images.txt", "r").readlines()
  ]
  label_list = [
      int(line.strip()) for line in open("train_labels.txt", "r").readlines()
  ]

  # Construct the dataset op
  image_list_placeholder = tf.placeholder(tf.string, [None])
  label_list_placeholder = tf.placeholder(tf.int64, [None])
  dataset = tf.data.Dataset.from_tensor_slices((image_list_placeholder,
                                                label_list_placeholder))
  dataset = dataset.repeat(epoch_number).map(_decode_image_file).batch(
      batch_size).shuffle(buffer_size=1000)  # Make batch after map

  iterator = dataset.make_initializable_iterator()
  batch_features_op, batch_label_op = iterator.get_next()
  batch_features_op = tf.reshape(batch_features_op, (-1, input_feature_size))
  batch_features_op = tf.cast(batch_features_op, tf.float32)

  # Define model
  def _model(x):

    W = tf.Variable(tf.zeros([input_feature_size, output_label_size]))
    b = tf.Variable(tf.zeros([output_label_size]))
    logits = tf.matmul(x, W) + b

    return logits

  # Define train op
  logits = _model(batch_features_op)
  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=batch_label_op))
  train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  # Define validate op
  train_accuracy_logits = _model(batch_features_op)
  train_softmax = tf.nn.softmax(train_accuracy_logits)
  train_correct_prediction = tf.equal(
      tf.argmax(train_softmax, 1), batch_label_op)
  train_accuracy_op = tf.reduce_mean(
      tf.cast(train_correct_prediction, tf.float32))

  # Define export model op
  model_features_placeholder = tf.placeholder(tf.float32,
                                              [None, input_feature_size])
  #labels_placeholder = tf.placeholder(tf.float32, [None, output_label_size])
  model_logits = _model(model_features_placeholder)
  model_softmax = tf.nn.softmax(model_logits)
  model_prediction = tf.argmax(model_softmax, 1)


  def export_model():
    model_path = "model"
    model_version = 1
    print("Export the model file: {}, version: {}".format(
        model_path, model_version))

    model_signature = signature_def_utils.build_signature_def(
        inputs={"image": utils.build_tensor_info(model_features_placeholder)},
        outputs={
            "softmax": utils.build_tensor_info(model_softmax),
            "prediction": utils.build_tensor_info(model_prediction)
        },
        method_name=signature_constants.PREDICT_METHOD_NAME)
    export_path = os.path.join(
        compat.as_bytes(model_path), compat.as_bytes(str(model_version)))
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder = saved_model_builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        clear_devices=True,
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            model_signature,
        },
        legacy_init_op=legacy_init_op)

    builder.save()

  # Start training
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(
        iterator.initializer,
        feed_dict={
            image_list_placeholder: image_list,
            label_list_placeholder: label_list
        })

    try:
      step_index = 0
      while True:

        _, loss_value = sess.run([train_op, loss])
        print("Run step: {}, loss: {}".format(step_index, loss_value))
        step_index += 1

        train_accuracy_value = sess.run(train_accuracy_op)
        print("Train accuracy: {}".format(train_accuracy_value))

    except tf.errors.OutOfRangeError:
      print("End of training")
      export_model()


def parse_args():
  parser = argparse.ArgumentParser(
      description='Train image classification model')
  parser.add_argument(
      '--source-splited-imageset',
      required=False,
      help='source root dir of splited dataset',
      type=str)
  parser.add_argument(
      '--output-label-size',
      required=False,
      help='output label size',
      type=int)
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  main(parse_args())
