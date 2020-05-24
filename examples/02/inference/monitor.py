"""
Monitor deployed endpoint
"""

import os, sys
import boto3
import csv
import json
from time import gmtime, strftime

from sagemaker.model_monitor import DataCaptureConfig
from sagemaker import RealTimePredictor
from sagemaker import session

from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.model_monitor import CronExpressionGenerator

import pandas as pd

from ...sm_utils import get_sm_execution_role, parse_infer_args

ON_SAGEMAKER_NOTEBOOK = False

endpoint_name = 'XGBoostEndpoint-2020-05-24-17-23-16'
s3_capture_upload_path = 's3://sagemaker-ap-southeast-2-454979696062/sagemaker-xgboost-2020-05-24-16-41-30-869/output/endpoint-data-capture'

#baseline_prefix = prefix + '/baselining'
#baseline_data_prefix = baseline_prefix + '/data'
#baseline_results_prefix = baseline_prefix + '/results'

baseline_data_uri = 's3://sagemaker-ap-southeast-2-454979696062/sagemaker/sklearncontainer/boston_train_xgb_n.csv'
baseline_results_uri = 's3://sagemaker-ap-southeast-2-454979696062/sagemaker-xgboost-2020-05-24-16-41-30-869/output/baseline-results'

boto_sess = boto3.Session()
sm_session = session.Session(boto_sess)

def capture():
    # Change parameters as you would like - adjust sampling percentage, 
    #  chose to capture request or response or both.
    #  Learn more from our documentation
    data_capture_config = DataCaptureConfig(
                            enable_capture = True,
                            sampling_percentage=50,
                            destination_s3_uri=s3_capture_upload_path,
                            kms_key_id=None,
                            capture_options=["REQUEST", "RESPONSE"],
                            csv_content_types=["text/csv"],
                            json_content_types=["application/json"])

    # Now it is time to apply the new configuration and wait for it to be applied
    predictor = RealTimePredictor(endpoint=endpoint_name)
    predictor.update_data_capture_config(data_capture_config=data_capture_config)
    sm_session.wait_for_endpoint(endpoint=endpoint_name)

def view_captures():
    bucket = 'sagemaker-ap-southeast-2-454979696062'
    data_capture_prefix = s3_capture_upload_path
    s3_client = boto_sess.client('s3')
    current_endpoint_capture_prefix = '{}/{}'.format(data_capture_prefix, endpoint_name)
    result = s3_client.list_objects(Bucket=bucket, Prefix=current_endpoint_capture_prefix)
    capture_files = [capture_file.get("Key") for capture_file in result.get('Contents')]
    print("Found Capture Files:")
    print("\n ".join(capture_files))

    def get_obj_body(obj_key):
        return s3_client.get_object(Bucket=bucket, Key=obj_key).get('Body').read().decode("utf-8")

    capture_file = get_obj_body(capture_files[-1])
    print(capture_file[:2000])

    print(json.dumps(json.loads(capture_file.split('\n')[0]), indent=2))

role = get_sm_execution_role(ON_SAGEMAKER_NOTEBOOK, boto_sess.region_name)

my_default_monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
)
def create_baseline():
    print(f'Baseline data uri: {baseline_data_uri}')
    print(f'Baseline results uri: {baseline_results_uri}')

    my_default_monitor.suggest_baseline(
        baseline_dataset=baseline_data_uri,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=baseline_results_uri,
        wait=True
    )

mon_schedule_name = 'xgb-boston-pred-model-monitor-schedule-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
my_default_monitor.create_monitoring_schedule(
    monitor_schedule_name=mon_schedule_name,
    endpoint_input=endpoint_name,
    output_s3_uri=baseline_results_uri.replace('baseline_results', 'monitor_reports'),
    statistics=baseline_results_uri + '/statistics.json',
    constraints=baseline_results_uri + '/constraints.json',
    schedule_cron_expression=CronExpressionGenerator.hourly(),
    enable_cloudwatch_metrics=True,
)







