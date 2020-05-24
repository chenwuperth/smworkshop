"""
Train job interacts with SageMaker using XGB

"""

import os
import datetime

import boto3
import pandas as pd
import numpy as np
import sagemaker
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sagemaker.amazon.amazon_estimator import get_image_uri

rom ...sm_utils import get_sm_execution_role


def reformat_csv(csv_fn):
    """
    Amazon SageMaker XGBoost can train on data in either a CSV or LibSVM format. 
    For CSV format, It should:

    Have the predictor variable in the first column
    Not have a header row
    """
    new_fn = csv_fn.replace('.csv', '_xgb.csv')
    # 1. skip the header
    # 2. replace the first col with the last col
    # 3. drop the last col
    with open(csv_fn, 'r') as fin:
        lines = fin.readlines()
    new_lines = []
    for line in lines[1:]:
        line = line.strip()
        fds = line.split(',')
        fds[0] = fds[-1]
        fds = fds[0:-1]
        new_line = ','.join(fds)
    new_lines.append(new_line)
    with open(new_fn, 'w') as fout:
        fout.write(os.linesep.join(new_lines))
    return new_fn

ON_SAGEMAKER_NOTEBOOK = False

# preparation
sm_boto3 = boto3.client('sagemaker')
sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = sess.default_bucket()  # this could also be a hard-coded bucket name
print('Using bucket ' + bucket)
sm_role = get_sm_execution_role(ON_SAGEMAKER_NOTEBOOK, region)

# Prepare data
data = load_boston()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=42)

trainX = pd.DataFrame(X_train, columns=data.feature_names)
trainX['target'] = y_train

testX = pd.DataFrame(X_test, columns=data.feature_names)
testX['target'] = y_test

trainX.head()

# convert to CSV so SM can consume
trainX.to_csv('boston_train.csv')
testX.to_csv('boston_test.csv')
ntrain_csv = reformat_csv('boston_train.csv')
ntest_csv = reformat_csv('boston_test.csv')

# send data to S3. SageMaker will take training data from s3
trainpath = sess.upload_data(
    path=ntrain_csv, bucket=bucket,
    key_prefix='sagemaker/sklearncontainer')

s3_input_train = sagemaker.s3_input(s3_data=trainpath, content_type='csv')

testpath = sess.upload_data(
    path=ntest_csv, bucket=bucket,
    key_prefix='sagemaker/sklearncontainer')

s3_input_validation = sagemaker.s3_input(s3_data=testpath, content_type='csv')

container = get_image_uri(region, 'xgboost', '0.90-1')
"""

max_depth controls how deep each tree within the algorithm can be built. 
    Deeper trees can lead to better fit, but are more computationally expensive and can lead to overfitting. There is typically some trade-off in model performance that needs to be explored between a large number of shallow trees and a smaller number of deeper trees.
subsample controls sampling of the training data. 
    This technique can help reduce overfitting, but setting it too low can also starve the model of data.
num_round controls the number of boosting rounds. 
    This is essentially the subsequent models that are trained using the residuals of previous iterations. Again, more rounds should produce a better fit on the training data, but can be computationally expensive or lead to overfitting.
eta controls how aggressive each round of boosting is. 
    Larger values lead to more conservative boosting.
gamma controls how aggressively trees are grown. Larger values lead to more conservative models.
"""
xgb = sagemaker.estimator.Estimator(
    container,
    role=sm_role, 
    train_instance_count=1, 
    train_instance_type='ml.m4.xlarge',
    hyperparameters={
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "silent":"0",
        "objective":"reg:linear",
        "num_round":"50"
    })
xgb.fit({'train':s3_input_train, 'test': s3_input_validation}, wait=False)
