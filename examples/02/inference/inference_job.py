"""
Inference batch transform job interacts with SageMaker

"""
import sagemaker
import boto3
import os

from sagemaker.amazon.amazon_estimator import get_image_uri

from sagemaker.xgboost.model import XGBoostModel

from ...sm_utils import get_sm_execution_role, parse_infer_args

ON_SAGEMAKER_NOTEBOOK = False

args = parse_infer_args()
sm_boto3 = boto3.client('sagemaker')
sess = sagemaker.Session()
region = sess.boto_session.region_name
container = get_image_uri(region, 'xgboost', '0.90-1')
model_url = args.model_file
sm_role = get_sm_execution_role(ON_SAGEMAKER_NOTEBOOK, region)

model = XGBoostModel(
        model_data=model_url,
        source_dir=os.path.abspath(os.path.dirname(__file__)),
        entry_point='inference.py',
        framework_version='0.90',
        role=get_sm_execution_role(ON_SAGEMAKER_NOTEBOOK, region))

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
    model.deploy(instance_type='ml.c5.xlarge', initial_instance_count=1)
else:
    raise Exception(f'Unknown inference mode {infer_mode}')

# primary_container = {
#     'Image': container,
#     'ModelDataUrl': model_url
# }

# create_model_response = sm_boto3.create_model(
#     ModelName='xgboost-0.90-1-boston-house',
#     ExecutionRoleArn=sm_role,
#     PrimaryContainer=primary_container)


