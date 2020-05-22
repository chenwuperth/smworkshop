"""
Inference batch transform job interacts with SageMaker

adapted from 
https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_randomforest

"""

from sagemaker.sklearn.model import SKLearnModel

import logging
import sagemaker
import boto3
import time
from urllib.parse import urlparse

from ...sm_utils import get_sm_execution_role, parse_infer_args

ON_SAGEMAKER_NOTEBOOK = False

def run_as_local_main():
    args = parse_infer_args()
    sm_boto3 = boto3.client('sagemaker')
    sess = sagemaker.Session()
    region = sess.boto_session.region_name
    #model_url = 's3://sagemaker-ap-southeast-2-454979696062/rf-scikit-2020-05-22-08-53-00-634/output/model.tar.gz'
    model_url = args.model_file
    model = SKLearnModel(
        model_data=model_url,
        role=get_sm_execution_role(ON_SAGEMAKER_NOTEBOOK, region),
        entry_point='inference.py')
    
    infer_mode = args.infer_mode
    if 'bt' == infer_mode:
        env = {'MODEL_SERVER_TIMEOUT':'120'}
        transformer = model.transformer(
            instance_count=1,
            instance_type='ml.c5.xlarge',
            output_path=args.output_dir,
            max_payload=99,
            env=env,
            max_concurrent_transforms=1,
            tags=[{"Key": "Project", "Value": "SM Example"}],
        )
        transformer.transform(args.input_file, content_type="text/csv")
        transformer.wait()
    elif 'ep' == infer_mode:
        model.deploy()
    else:
        raise Exception(f'Unknown inference mode {infer_mode}')

def run_as_aws_batch_job():
    args = parse_infer_args()


if __name__ == "__main__":
    run_as_local_main()
