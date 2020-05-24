"""
Invoke deployed endpoint
"""

import os
import boto3
import csv

def drop_first_column(fname_in, fname_out):
    """
    Need to remove the target variables to simulate the test
    """
    with open(fname_in, 'r') as fin, open(fname_out, 'w') as fout:
        inlines = fin.readlines()
        new_lines = []
        for line in inlines:
            line = line.strip()
            fds = line.split(',')[1:]
            new_line = ','.join(fds)
            new_lines.append(new_line)
        fout.write(os.linesep.join(new_lines))

runtime = boto3.client('sagemaker-runtime')
csv_fp = 'boston_test_xgb.csv'
csv_fp_n = 'boston_test_xgb_n.csv'

if (not os.path.exists(csv_fp_n)):
    drop_first_column(csv_fp, csv_fp_n)

endpoint_name = 'XGBoostEndpoint-2020-05-24-17-23-16'

with open(csv_fp_n, 'r') as fin:
    csv_body = fin.read()

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=csv_body.encode('utf-8'),
    ContentType='text/csv')

print(response['Body'].read())