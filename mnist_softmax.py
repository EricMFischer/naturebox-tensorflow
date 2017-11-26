# https://www.tensorflow.org/get_started/mnist/beginners

# When one learns how to program, there's a tradition that the first thing you do is print "Hello World."
# Just like programming has Hello World, machine learning has MNIST.
# To start learning Tensorflow, let's learn about a very simple model, Softmax (multinomial logistic) Regression.

# What we will accomplish in this tutorial:
# 1. Learn about the MNIST data and softmax regressions
# 2. Create a function that is a model for recognizing digits, based on looking at every pixel in the image
# 3. Use TensorFlow to train the model to recognize digits by having it "look" at thousands of examples (and run our
# first TensorFlow session to do so)
# 4. Check the model's accuracy with our test data

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data (55k points of training data, 10k of test, 5k of validation)
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # "None" - dimension can be any length, for any num of MNIST images
  x = tf.placeholder(tf.float32, [None, 784])
  # W (weights) is a 784x10 matrix because we have 784 input features and 10 outputs, i.e. because we want to multiply
  # the 784-dimensional image vectors by it to produce 10-dimensional vectors of evidence for the difference classes 0-9
  W = tf.Variable(tf.zeros([784, 10]))
  # b (biases) is a 10 dimensional vector for the 10 number classes, and so we can add it to output
  b = tf.Variable(tf.zeros([10]))

  # REGRESSION MODEL
  # We can now implement our regression model. We multiply the vectorized input images 'x' by the weight matrix W, add the bias b.
  y = tf.matmul(x, W) + b

  # y_ will consist of a 2d tensor, where each row is a one-hot 10-dimensional vector indicating which digit class
  # (0-9) the corresponding MNIST image belongs to.
  # The 2nd shape argument is optional, but it allows TensorFlow to catch bugs stemming from inconsistent tensor shapes.
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  # can be numerically unstable.
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of y, and then average across the batch.

  # LOSS FUNCTION
  # Specifying a loss function just as easy. Loss (how far model is from desired outcome) indicates how bad the model's
  # prediction was on a single example; we try to minimize that while training across all the examples. Here, our loss
  # function is the cross-entropy between the target and the softmax activation function applied to the model's prediction.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  # Because TensorFlow knows the entire graph of your computations, it automatically uses the
  # backpropagation algorithm to determine how your variables affect the loss you want to minimize.
  # But you can also apply an optimization algorithm to modify the variables and reduce the loss.
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  # What TensorFlow actually does here, behind the scenes, is to add new operations to your graph which
  # implement backpropagation and gradient descent. Then it gives you back a single operation which, when run,
  # does a step of gradient descent training, slightly tweaking your variables to reduce the loss.

  # Launch the model in an interactive session.
  # TensorFlow relies on a highly efficient C++ backend to do its computation. The connection to this backend
  # is called a session. The common usage for TensorFlow programs is to first create a graph and then launch it in a session.
  # The InteractiveSession class allows you to interleave operations which build a computation graph with ones that run
  # the graph. This is particularly convenient when working in interactive contexts like IPython. If you are not using an
  # InteractiveSession, then you should build the entire computation graph before starting a session and launching the graph.
  sess = tf.InteractiveSession()
  # Computational graph - series of TensorFlow operations arranged into a graph of nodes. To actually evaluate the nodes,
  # we must run the computational graph within a session. A session encapsulates the control and state of the TensorFlow runtime.

  # Create and run an operation to initialize all the variables we created
  # TensorFlow variables are not initialized until you call tf.global_variables_initializer().run()
  tf.global_variables_initializer().run()

  # TRAIN
  for _ in range(1000):
    # Each step of the loop, we get a "batch" of one hundred random data points from our training set.
    # We run train_step feeding in the batches data to replace the placeholders.
    # Using small batches of random data is called stochastic training--in this case, stochastic gradient descent.
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
