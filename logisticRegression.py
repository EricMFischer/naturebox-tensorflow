# https://www.tensorflow.org/get_started/input_fn
# Steps to build logistic regression model:
# 1. Build input function with tf.Estimator that returns a dictionary of tensors, keys being the attributes in members.csv,
# and values being all the possible tensor values for this attribute (e.g. ’num_days_as_member’). This input function will
# be passed to the train() call for training and the evaluate() call for testing.
#
# 2. FeatureColumns provide a spec for the input data of your model, indicating how to transform and represent the data.
#
# 3. Pass a list of FeatureColumn’s to the constructor. Call train() and evaluate()
#
# 4. For training you need a linear estimator, which has methods for training and evaluation on it. Use
# tf.estimator.LinearRegressor to build a linear regression estimator.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import argparse
import shutil
import sys

# Offers data structures and operations for manipulating numerical tables and time series
import pandas as pd
import tensorflow as tf

COLUMNS = ["mem_signup_month", "num_days_as_member",
"acquisition_source", "days_between_first_and_second_order",
"days_between_second_and_third_order", "days_between_third_and_fourth_order",
"days_between_fourth_and_fifth_order", "spicy_preference_index",
"sweet_preference_index", "tart_preference_index",
"has_purchased_a_limited_edition_sku", "has_spent_more_than_50_dollars"]
FEATURES = ["mem_signup_month", "num_days_as_member",
"acquisition_source", "days_between_first_and_second_order",
"days_between_second_and_third_order", "days_between_third_and_fourth_order",
"days_between_fourth_and_fifth_order", "spicy_preference_index",
"sweet_preference_index", "tart_preference_index",
"has_purchased_a_limited_edition_sku"]
LABEL = "has_spent_more_than_50_dollars"

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='/tmp/naturebox_model',
    help='Base directory for the model.')

parser.add_argument(
    '--train_data', type=str, default='/tmp/members_train.csv',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='/tmp/members_test.csv',
    help='Path to the test data.')

# INPUT FUNCTION
# Preprocess your data here, and then return:
# 1) dictionary of feature columns to Tensors with the corresponding feature data
# feature_cols (dictionary): contains key/value pairs that map feature column names to Tensors (or SparseTensors)
# containing the corresponding feature data.
# 2) tensor containing labels
# labels (tensor): contains your label (target) values: the values your model aims to predict.

# get_input_fn: Define a method that returns an input function based on the given data. Note that the returned input
# function will be called while constructing the TensorFlow graph, NOT while running the graph. What it is returning is
# a representation of the input data as the fundamental unit of TensorFlow computations, a Tensor (or SparseTensor).

# num_epochs: controls the number of epochs to iterate over data. For training, set this to None, so the
# input_fn keeps returning data until the required number of train steps is reached.
# For evaluate and predict, set this to 1, so the input_fn will iterate over the data once and then raise
# OutOfRangeError. That error will signal the Estimator to stop evaluate or predict.

# shuffle: Whether to shuffle the data. For evaluate and predict, set this to False, so the input_fn iterates over
# the data sequentially. For train, set this to True.
def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)


def build_model_columns():
  # Categorical and continuous feature columns:
  mem_signup_month = tf.feature_column.categorical_column_with_vocabulary_list(
    'mem_signup_month', [1,2,3,4,5,6,7,8,9,10,11,12])
  num_days_as_member = tf.feature_column.numeric_column("num_days_as_member")
  acquisition_source = tf.feature_column.categorical_column_with_vocabulary_list(
    'acquisition_source', ["unknown", "google", "facebook", "other paid", "friends/family",
    "blog", "radio", "youtube", "podcast", "mailer", "instagram", "target", "catalog/mailer"])
  days_between_first_and_second_order = tf.feature_column.numeric_column("days_between_first_and_second_order")
  days_between_second_and_third_order = tf.feature_column.numeric_column("days_between_second_and_third_order")
  days_between_third_and_fourth_order = tf.feature_column.numeric_column("days_between_third_and_fourth_order")
  days_between_fourth_and_fifth_order = tf.feature_column.numeric_column("days_between_fourth_and_fifth_order")
  spicy_preference_index = tf.feature_column.numeric_column("spicy_preference_index")
  sweet_preference_index = tf.feature_column.numeric_column("sweet_preference_index")
  tart_preference_index = tf.feature_column.numeric_column("tart_preference_index")
  has_purchased_a_limited_edition_sku = tf.feature_column.categorical_column_with_vocabulary_list(
    'has_purchased_a_limited_edition_sku', [0,1])

  return [
      mem_signup_month, num_days_as_member, acquisition_source, days_between_first_and_second_order,
      days_between_second_and_third_order, days_between_third_and_fourth_order,
      days_between_fourth_and_fifth_order, spicy_preference_index, sweet_preference_index,
      tart_preference_index, has_purchased_a_limited_edition_sku,
  ]


# LinearRegressor source code: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/estimator/canned/linear.py
# Creating a linear regressor: https://www.tensorflow.org/api_docs/python/tf/estimator/LinearRegressor
# How logistic regression works: https://www.tensorflow.org/tutorials/wide#how_logistic_regression_works
# Run TensorBoard to inspect the details about the graph and training progression. tensorboard --logdir=/tmp/naturebox_model
def build_estimator(model_dir):
  feature_columns = build_model_columns()

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
    session_config=tf.ConfigProto(device_count={'GPU': 0}))

  # Instantiate a LinearRegressor for the logistic regression model
  return tf.estimator.LinearRegressor(model_dir=model_dir,
                                      feature_columns=feature_columns,
                                      config=run_config)


def main(unused_argv):
  # Clean up the model directory if present
  shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

  # Load datasets (read 3 CSV's into pandas Dataframes)
  training_set = pd.read_csv("members_train.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
  test_set = pd.read_csv("members_test.csv", skipinitialspace=True,
                         skiprows=1, names=COLUMNS)
  # Set of 60 examples for which to predict whether a user has spent more than $50
  prediction_set = pd.read_csv("members_predict.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

  # Converting data into tensors
  # When building a tf.estimator model, the input data is specified by means of an Input Builder function. This builder
  # function will not be called until it is later passed to tf.estimator.Estimator methods such as train and evaluate.
  # The purpose of this function is to construct the input data, which is represented in the form of tf.Tensors or
  # tf.SparseTensors. In more detail, the input builder function returns the following as a pair:
  # features: A dict from feature column names to Tensors or SparseTensors.
  # labels: A Tensor containing the label column.

  # Define feature columns and create regressor
  regressor = build_estimator(FLAGS.model_dir)

  # TRAIN
  regressor.train(input_fn=get_input_fn(training_set), steps=5000)


  # EVALUATE loss over one epoch of test_set
  results = regressor.evaluate(
      input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
  # Print the stats for the evaluation
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))
  loss_score = results["loss"]
  print("Loss: {0:f}".format(loss_score))


  # PREDICT - print out predictions over a slice of prediction_set
  y = regressor.predict(
      input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
  # .predict() returns an iterator of dicts; convert to a list and print
  predictions = list(p["predictions"] for p in itertools.islice(y, 6))
  print("Predictions: {}".format(str(predictions)))


if __name__ == "__main__":
  # The run_context argument is a SessionRunContext that provides information about the upcoming run() call: the
  # originally requested op/tensors, the TensorFlow Session.
  # before_run(run_context)
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
