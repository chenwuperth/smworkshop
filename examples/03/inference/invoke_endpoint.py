"""
Invoke deployed endpoint
"""
import os
import boto3

runtime = boto3.client('sagemaker-runtime')
source_dir = os.path.abspath(os.path.dirname(__file__))
csv_fp = 'boston_test.csv'
endpoint_name = 'sagemaker-scikit-learn-2020-05-23-12-17-39-214'

with open(csv_fp, 'r') as fin:
    csv_body = fin.read()

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=csv_body.encode('utf-8'),
    ContentType='text/csv')

print(response['Body'].read())