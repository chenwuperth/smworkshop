import boto3
from sagemaker import get_execution_role
import argparse

def get_sm_execution_role(on_sm_notebook, region):

    if on_sm_notebook:
        return get_execution_role()
    # hack - just hardcode our role
    # return "arn:aws:iam::1234567:role/service-role/AmazonSageMaker-ExecutionRole-20171229T134248"
    # return 'AmazonSageMaker-ExecutionRole-20171229T134248'

    # cf - https://github.com/aws/sagemaker-python-sdk/issues/300
    client = boto3.client('iam', region_name=region)
    response_roles = client.list_roles(
        PathPrefix='/',
        # Marker='string',
        MaxItems=999
    )
    for role in response_roles['Roles']:
        if role['RoleName'].startswith('AmazonSageMaker-ExecutionRole-'):
            print('Resolved SageMaker IAM Role to: ' + str(role))
            return role['Arn']
    raise Exception(
        'Could not resolve what should be the SageMaker role to be used')

def parse_infer_args():
    #logging.debug("_parse_args()")

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--infer-mode', type=str, default='bt', help="BatchTransform (bt) or EndPoint (ep)")
    parser.add_argument('--input-file', type=str)
    parser.add_argument('--model-file', type=str)
    parser.add_argument('--output-dir', type=str)

    args, _ = parser.parse_known_args()
    if args.infer_mode == 'bt' and args.input_file is None:
        raise Exception('Batch transform inference needs input-file')
    for arg, text in zip([args.model_file, args.output_dir], ['--model-file', '--output-dir']):
        if arg is None:
            raise Exception(f'{text} is None')
    
    return args

def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', dest='tune', action='store_true', default=False)
    parser.add_argument('--tune-job-name', type=str)
    args, _ = parser.parse_known_args()
    return args