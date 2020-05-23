"""
inference script for both batch transform and endpoint

Implements four call-backs, see https://sagemaker.readthedocs.io/en/stable/using_sklearn.html

# The SageMaker Scikit-learn model server loads your model by 
#    invoking a model_fn function that you must provide in your script. 
model = model_fn(model_dir)

# Deserialize the Invoke request body into an object we can perform prediction on
input_object = input_fn(request_body, request_content_type)

# Perform prediction on the deserialized object, with the loaded model
prediction = predict_fn(input_object, model)

# Serialize the prediction result into the desired response content type
output = output_fn(prediction, response_content_type)

"""
from sklearn.externals import joblib
import os
import numpy as np

# inference functions
def model_fn(model_dir):
    """
    This loads returns a Scikit-learn Classifier from a model.joblib file in 
    the SageMaker model directory model_dir.
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

# please check "default" implementation of the following functions, and change as necessary
# https://github.com/aws/sagemaker-scikit-learn-container/blob/master/src/sagemaker_sklearn_container/serving.py
# 
# TODO try to comment out the "input_fn" function below, see what happens?
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
    return np.stack(arrs)

# def predict_fn(input_object, model):
#     pass


# def output_fn(prediction, response_content_type):
#     pass
