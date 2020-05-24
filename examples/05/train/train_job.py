"""
Train job interacts with SageMaker
"""
import os
import boto3
import numpy as np
import pandas as pd
import sagemaker

# We use the Estimator from the SageMaker Python SDK
from sagemaker.pytorch.estimator import PyTorch

from ...sm_utils import get_sm_execution_role

ON_SAGEMAKER_NOTEBOOK = False


# preparation
sm_boto3 = boto3.client('sagemaker')
sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = sess.default_bucket()  # this could also be a hard-coded bucket name
print('Using bucket ' + bucket)
sm_role = get_sm_execution_role(ON_SAGEMAKER_NOTEBOOK, region)
trainpath = 's3://sagemaker-ap-southeast-2-454979696062/sagemaker/sklearncontainer/adult.csv'
pytorch_estimator = PyTorch(
    entry_point='train.py',
    source_dir=os.path.abspath(os.path.dirname(__file__)),
    role = sm_role,
    train_instance_count=1,
    train_instance_type='ml.c5.xlarge',
    framework_version='1.5.0',
    base_job_name='fastai-pytorch',
    metric_definitions=[
        {'Name': 'Dice accuracy',
         'Regex': "Dice accuracy: ([0-9.]+).*$"}],
    hyperparameters = {'hidden_layer_1': 200,
                        'hidden_layer_2': 100})

# launch training job, with asynchronous call
pytorch_estimator.fit({'train':trainpath}, wait=False)
