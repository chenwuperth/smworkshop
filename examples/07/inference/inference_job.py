import logging
import os
import time
from urllib.parse import urlparse

import boto3
import sagemaker
from sagemaker.pipeline import PipelineModel
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.model import Model
from sagemaker.amazon.amazon_estimator import get_image_uri

from ...sm_utils import get_sm_execution_role, parse_infer_args

from time import gmtime, strftime

"""
Inference batch transform job interacts with SageMaker
"""

if __name__ == "__main__":
    args = parse_infer_args()
    sm_boto3 = boto3.client('sagemaker')
    sess = sagemaker.Session()
    region = sess.boto_session.region_name
    model_url = args.model_file
    preproc_model_url = args.preproc_model
    sm_role = get_sm_execution_role(False, region)

    preproc_model = SKLearnModel(
        model_data=preproc_model_url,
        source_dir=os.path.abspath(os.path.dirname(__file__)),
        role=sm_role,
        entry_point='infer_preproc.py',
        sagemaker_session=sess)

    ll_image = get_image_uri(region, 'linear-learner')

    ll_model = Model(
        model_data=model_url,
        image=ll_image,
        role=sm_role,
        sagemaker_session=sess
    )

    timestamp_prefix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    model_name = 'inference-pipeline-' + timestamp_prefix
    endpoint_name = 'inference-pipeline-ep-' + timestamp_prefix
    sm_model = PipelineModel(
        name=model_name,
        role=sm_role,
        models=[
            preproc_model,
            ll_model
        ],
        sagemaker_session=sess
    )
    print('Deploying SM EndPoint')
    sm_model.deploy(initial_instance_count=1,
                    instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
    print(f'SageMaker deployed to endpoint - {endpoint_name}')

