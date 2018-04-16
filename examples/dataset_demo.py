#!/usr/bin/env python

import tensorflow as tf
import numpy as np


def main():
  # Prepare training data
  features_array = np.array([[1, 1.0], [2, 2.0], [3, 3.0], [4, 4.0], [5, 5.0]])
  labels_array = np.array([1, 0, 1, 1, 1])

  # Construct iterator
  features_array_placeholder = tf.placeholder(tf.float32, [None, 2])
  labels_array_placeholder = tf.placeholder(tf.int32, [None])
  dataset = tf.data.Dataset.from_tensor_slices(
      (features_array, labels_array)).repeat(5).batch(3).shuffle(
          buffer_size=1000)
  iterator = dataset.make_initializable_iterator()
  batch_features_op, batch_label_op = iterator.get_next()

  # Run training
  with tf.Session() as sess:

    sess.run(
        iterator.initializer,
        feed_dict={
            features_array_placeholder: features_array,
            labels_array_placeholder: labels_array
        })

    try:
      index = 0
      while True:
        batch_features, batch_label = sess.run(
            [batch_features_op, batch_label_op])
        print("Index: {}, batch features: {}, batch label: {}".format(
            index, batch_features, batch_label))
        index += 1
    except tf.errors.OutOfRangeError:
      print("End of training")


if __name__ == "__main__":
  main()
