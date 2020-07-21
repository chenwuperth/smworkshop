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
    sm_role = get_sm_execution_role(False, region)

    fdir = os.path.abspath(os.path.dirname(__file__))

    sklearn_preprocessor = SKLearn(
        entry_point='train_preproc.py',
        source_dir=fdir,
        role=sm_role,
        train_instance_type="ml.c4.xlarge",
        base_job_name='preproc-scikit')

    prefix = 'inference-pipeline-scikit-linearlearner'

    # curl -O https://s3-us-west-2.amazonaws.com/sparkml-mleap/data/abalone/abalone.csv
    train_input = sess.upload_data(
        path=os.path.join(fdir, 'abalone.csv'), 
        bucket=bucket,
        key_prefix='{}/{}'.format(prefix, 'train'))
     # there is no need to validate models for pre-processing, so no SM_CHANNEL_TEST
    sklearn_preprocessor.fit({'train': train_input})
    print(f'train input on S3 - {train_input}')
    # https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/estimator.py#L724-L741
    print(f'SKlearn preprocessor trained model uploaded to - {sklearn_preprocessor.model_data}')