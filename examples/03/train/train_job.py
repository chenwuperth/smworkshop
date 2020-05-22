"""
Train job interacts with SageMaker

adapted from 
https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_randomforest

"""

import datetime
import tarfile

import boto3
import pandas as pd
import numpy as np
from sagemaker import get_execution_role
import sagemaker
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

# We use the Estimator from the SageMaker Python SDK
from sagemaker.sklearn.estimator import SKLearn

from ...sm_utils import get_sm_execution_role

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

# send data to S3. SageMaker will take training data from s3
trainpath = sess.upload_data(
    path='boston_train.csv', bucket=bucket,
    key_prefix='sagemaker/sklearncontainer')

testpath = sess.upload_data(
    path='boston_test.csv', bucket=bucket,
    key_prefix='sagemaker/sklearncontainer')


sklearn_estimator = SKLearn(
    entry_point='train.py',
    role = sm_role,
    train_instance_count=1,
    train_instance_type='ml.c5.xlarge',
    framework_version='0.20.0',
    base_job_name='rf-scikit',
    metric_definitions=[
        {'Name': 'median-AE',
         'Regex': "AE-at-50th-percentile: ([0-9.]+).*$"}],
    hyperparameters = {'n-estimators': 100,
                       'min-samples-leaf': 3,
                       'features': 'CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT',
                       'target': 'target'})

# launch training job, with asynchronous call
sklearn_estimator.fit({'train':trainpath, 'test': testpath})

artifact = sm_boto3.describe_training_job(
    TrainingJobName=sklearn_estimator.latest_training_job.name)['ModelArtifacts']['S3ModelArtifacts']

# we will use this URL of the artifact for inference
print(f'Model artifact persisted at {artifact}')
