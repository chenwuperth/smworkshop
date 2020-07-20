from sagemaker.sklearn.model import SKLearnModel

import os
import logging
import sagemaker
import boto3
import time
from urllib.parse import urlparse

from ...sm_utils import get_sm_execution_role, parse_infer_args

if __name__ == "__main__":
    args = parse_infer_args()
    sm_boto3 = boto3.client('sagemaker')
    sess = sagemaker.Session()
    region = sess.boto_session.region_name
    model_url = args.model_file

    # Define a SKLearn Transformer from the trained SKLearn Estimator
    # transformer = sklearn_preprocessor.transformer(
    #     instance_count=1, 
    #     instance_type='ml.m4.xlarge',
    #     assemble_with = 'Line',
    #     accept = 'text/csv')

    model = SKLearnModel(
        model_data=model_url,
        source_dir=os.path.abspath(os.path.dirname(__file__)),
        role=get_sm_execution_role(False, region),
        entry_point='infer_preproc.py')
    
    infer_mode = args.infer_mode
    if 'bt' == infer_mode:
        env = {'MODEL_SERVER_TIMEOUT':'120'}
        transformer = model.transformer(
            instance_count=1,
            instance_type='ml.m4.xlarge',
            output_path=args.output_dir,
            assemble_with='Line',
            max_payload=99,
            accept='text/csv',
            env=env,
            max_concurrent_transforms=1,
            tags=[{"Key": "Project", "Value": "SM Example"}],
        )
        transformer.transform(args.input_file, content_type="text/csv")
        transformer.wait()
        preprocessed_train = transformer.output_path
        print(f'Preprocessed data at {preprocessed_train}')
    elif 'ep' == infer_mode:
        model.deploy(instance_type='ml.c5.xlarge', initial_instance_count=1)
    else:
        raise Exception(f'Unknown inference mode {infer_mode}')
