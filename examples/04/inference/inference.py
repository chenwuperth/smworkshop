"""
inference script for both batch transform and endpoint

Implements four call-backs, see https://sagemaker.readthedocs.io/en/stable/using_pytorch.html#serve-a-pytorch-model

After the SageMaker model server has loaded your model by calling model_fn, SageMaker will serve your model. 
Model serving is the process of responding to inference requests, received by SageMaker InvokeEndpoint API calls. 
The SageMaker PyTorch model server breaks request handling into three steps:

input processing,

prediction, and

output processing.

In a similar way to model loading, you configure these steps by defining functions in your Python source file.

Each step involves invoking a python function, with information about the request and the return value from the previous function in the chain. Inside the SageMaker PyTorch model server, the process looks like:

# Deserialize the Invoke request body into an object we can perform prediction on
input_object = input_fn(request_body, request_content_type)

# Perform prediction on the deserialized object, with the loaded model
prediction = predict_fn(input_object, model)

# Serialize the prediction result into the desired response content type
output = output_fn(prediction, response_content_type)

"""
import os
import numpy as np

import torch
import torch.nn as nn

# inference functions
def model_fn(model_dir):
    """
    This loads returns a PyTorch model model.pth 
    the SageMaker model directory model_dir.
    """
    dim = 13
    model = nn.Sequential(
        nn.Linear(dim, 50, bias=True), nn.ELU(),
        nn.Linear(50, 50, bias=True), nn.ELU(),
        nn.Linear(50, 50, bias=True), nn.Sigmoid(),
        nn.Linear(50, 1)
    )
    #criterion = nn.MSELoss()
    #opt = torch.optim.Adam(net.parameters(), lr=.0005)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model

# please check "default" implementation of the following functions, and change as necessary
# https://github.com/aws/sagemaker-pytorch-serving-container/blob/master/src/sagemaker_pytorch_serving_container/default_inference_handler.py
# 
# TODO try to comment out the "input_fn" function below, see what happens?
# TODO check the default implementation (see link above) to return input on GPU device
def input_fn(request_body, request_content_type):
    """Takes request data and de-serializes the data into an object for prediction.
        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:
            - The request Content-Type, for example "application/json"
            - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.
        The input_fn is responsible to take the request data and pre-process it before prediction.
    Args:
        input_data (obj): the request data.
        content_type (str): the request Content-Type.
    Returns:
        (obj): data ready for prediction.
    """

    #print(request_body)
    lines = request_body.split(os.linesep)
    arrs = []
    for line in lines:
        # we know it is CSV format
        line = line.strip()
        ds = line.split(',')[1:-1] # remove the first and last element
        if (len(ds) != 13):
            print(f'short line {line}')
            continue
        try:
            ds_arr = np.array([float(x) for x in ds])
        except:
            print(f'Fail to convert to float {line}')
            continue
        arrs.append(ds_arr)
    return torch.from_numpy(np.stack(arrs).astype(np.float32)).clone()

# def predict_fn(input_object, model):
#     pass


# def output_fn(prediction, response_content_type):
#     pass
