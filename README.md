# NaturePBox Tensorflow Repo

Welcome to Toy Problems. This repository will be updated every morning with a new
code challenge.

Feel free to use Google to aid you in solving the coding challenges!

## Using this Repository

Launches TensorFlow CPU binary images in a Docker container:

Run TensorFlow programs in a Jupyter notebook:

    docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow

Start a docker container to run TensorFlow programs in shell:

    docker run -it gcr.io/tensorflow/tensorflow bash

## Steps to build logistic regression model

1. Build input function with tf.Estimator that returns a dictionary of tensors, keys being the attributes in members.csv, and values being all the possible tensor values for this attribute (e.g. ’num_days_as_member’). This input function will be passed to the train() call that initiates training and the evaluate() call that initiates testing.
2. FeatureColumns provide a spec for the input data of your model, indicating how to transform and represent the data.
3. Training: You need a linear estimator, which has methods for training and evaluation on it. Use tf.estimator.LinearRegressor to build a linear regression estimator.
4. Pass a list of FeatureColumn’s to the constructor. Call train() and evaluate()
