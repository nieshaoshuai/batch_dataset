#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout
from keras.optimizers import SGD
from keras.layers import Input, Dense
from keras.models import Model


def main():
  print("Start training")

  x = np.arange(4).reshape(-1, 1).astype('float32')
  ds_x = tf.data.Dataset.from_tensor_slices(x).repeat().batch(4)
  it_x = ds_x.make_one_shot_iterator()

  y = np.arange(5, 9).reshape(-1, 1).astype('float32')
  ds_y = tf.data.Dataset.from_tensor_slices(y).repeat().batch(4)
  it_y = ds_y.make_one_shot_iterator()

  input_vals = Input(tensor=it_x.get_next())
  output = Dense(1, activation='relu')(input_vals)
  model = Model(inputs=input_vals, outputs=output)
  model.compile('rmsprop', 'mse', target_tensors=[it_y.get_next()])
  model.fit(steps_per_epoch=1, epochs=5, verbose=2)

  print("End of training")


if __name__ == "__main__":
  main()
