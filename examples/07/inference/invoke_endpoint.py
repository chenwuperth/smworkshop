"""
Invoke deployed endpoint
"""
import sagemaker
from sagemaker.predictor import csv_serializer, RealTimePredictor
from sagemaker.content_types import CONTENT_TYPE_CSV, CONTENT_TYPE_JSON

from ...sm_utils import parse_invoke_args

if __name__ == "__main__":
    args = parse_invoke_args()
    ep_name = args.end_point
    sm_sess = sagemaker.Session()

    if (args.delete_ep):
        print(f'Deleting EndPoint {ep_name}')
        sm_client = sm_sess.boto_session.client('sagemaker')
        sm_client.delete_endpoint(EndpointName=ep_name)
    else:
        print(f'Invoking EndPoint {ep_name}')
        payload = 'M, 0.44, 0.365, 0.125, 0.516, 0.2155, 0.114, 0.155'
        actual_rings = 10

        predictor = RealTimePredictor(
            endpoint=ep_name,
            sagemaker_session=sm_sess,
            serializer=csv_serializer,
            content_type=CONTENT_TYPE_CSV,
            accept=CONTENT_TYPE_JSON
        )

        print(predictor.predict(payload))