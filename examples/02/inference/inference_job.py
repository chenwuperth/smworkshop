"""
Inference batch transform job interacts with SageMaker

"""
import sys
import sagemaker
import boto3
import os

import time
from time import gmtime, strftime

from sagemaker.amazon.amazon_estimator import get_image_uri

#from sagemaker.xgboost.model import XGBoostModel

from ...sm_utils import get_sm_execution_role, parse_infer_args

ON_SAGEMAKER_NOTEBOOK = False

args = parse_infer_args()
sm_boto3 = boto3.client('sagemaker')
sess = sagemaker.Session()
region = sess.boto_session.region_name
container = get_image_uri(region, 'xgboost', '0.90-1')
print(container)
#sys.exit(0)
model_url = args.model_file
sm_role = get_sm_execution_role(ON_SAGEMAKER_NOTEBOOK, region)

primary_container = {
    'Image': container,
    'ModelDataUrl': model_url
}
model_name = 'xgboost-boston-house'
create_model_response = sm_boto3.create_model(
    ModelName=model_name,
    ExecutionRoleArn=sm_role,
    PrimaryContainer=primary_container)

print(create_model_response['ModelArn'])

endpoint_config_name = 'BostonHouse-XGBoostEndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_config_name)
create_endpoint_config_response = sm_boto3.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.m5.xlarge',
        'InitialVariantWeight':1,
        'InitialInstanceCount':1,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

endpoint_name = 'XGBoostEndpoint-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_name)
create_endpoint_response = sm_boto3.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name)
print(create_endpoint_response['EndpointArn'])

# resp = sm_boto3.describe_endpoint(EndpointName=endpoint_name)
# status = resp['EndpointStatus']
# while status=='Creating':
#     print("Status: " + status)
#     time.sleep(60)
#     resp = sm_boto3.describe_endpoint(EndpointName=endpoint_name)
#     status = resp['EndpointStatus']

# print("Arn: " + resp['EndpointArn'])
# print("Status: " + status)

# infer_mode = args.infer_mode
# if 'bt' == infer_mode:
#     env = {'MODEL_SERVER_TIMEOUT':'120'}
#     transformer = model.transformer(
#         instance_count=1,
#         instance_type='ml.c5.xlarge',
#         output_path=args.output_dir,
#         max_payload=99,
#         env=env,
#         max_concurrent_transforms=1,
#         tags=[{"Key": "Project", "Value": "SM Example"}],
#     )
#     transformer.transform(args.input_file, content_type="text/csv")
#     transformer.wait()
# elif 'ep' == infer_mode:
#     model.deploy(instance_type='ml.c5.xlarge', initial_instance_count=1)
# else:
#     raise Exception(f'Unknown inference mode {infer_mode}')




