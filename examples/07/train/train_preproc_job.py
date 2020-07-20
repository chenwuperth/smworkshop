"""
Train job interacts with SageMaker
"""
import os
import boto3

import sagemaker
from sagemaker.sklearn.estimator import SKLearn

from ...sm_utils import get_sm_execution_role, parse_train_args

if __name__ == '__main__':
    args = parse_train_args()

    sm_boto3 = boto3.client('sagemaker')
    sess = sagemaker.Session()
    region = sess.boto_session.region_name
    bucket = sess.default_bucket()  # this could also be a hard-coded bucket name
    print('Using bucket ' + bucket)
    sm_role = get_sm_execution_role(region)

    sklearn_preprocessor = SKLearn(
        entry_point='train_preproc.py',
        source_dir=os.path.abspath(os.path.dirname(__file__)),
        role=sm_role,
        train_instance_type="ml.c4.xlarge",
        base_job_name='preproc-scikit')

    prefix = 'inference-pipeline-scikit-linearlearner'

    # curl -O https://s3-us-west-2.amazonaws.com/sparkml-mleap/data/abalone/abalone.csv
    train_input = sess.upload_data(
        path='abalone.csv', 
        bucket=bucket,
        key_prefix='{}/{}'.format(prefix, 'train'))
     # there is no need to validate pre-processing, so no SM_CHANNEL_TEST
    sklearn_preprocessor.fit({'train': train_input})