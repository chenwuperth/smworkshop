"""
Train job interacts with SageMaker
"""

import os
import boto3

import sagemaker

from sagemaker.amazon.amazon_estimator import get_image_uri

from ...sm_utils import get_sm_execution_role, parse_train_args

if __name__ == '__main__':
    args = parse_train_args()

    sm_boto3 = boto3.client('sagemaker')
    sess = sagemaker.Session()
    region = sess.boto_session.region_name
    sm_role = get_sm_execution_role(False, region)
    bucket = sess.default_bucket()  # this could also be a hard-coded bucket name
    ll_image = get_image_uri(region, 'linear-learner')

    s3_ll_output_key_prefix = "ll_training_output"
    prefix = 'inference-pipeline-scikit-linearlearner'
    s3_ll_output_location = 's3://{}/{}/{}/{}'.format(bucket, prefix, s3_ll_output_key_prefix, 'll_model')

    ll_estimator = sagemaker.estimator.Estimator(
        ll_image,
        sm_role, 
        train_instance_count=1, 
        train_instance_type='ml.m4.2xlarge',
        train_volume_size = 20,
        train_max_run = 3600,
        input_mode= 'File',
        output_path=s3_ll_output_location,
        sagemaker_session=sess)

    ll_estimator.set_hyperparameters(feature_dim=10, predictor_type='regressor', mini_batch_size=32)

    ll_train_data = sagemaker.session.s3_input(
        preprocessed_train, 
        distribution='FullyReplicated',
        content_type='text/csv', 
        s3_data_type='S3Prefix')

    data_channels = {'train': ll_train_data}
    ll_estimator.fit(inputs=data_channels, logs=True)