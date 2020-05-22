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

# inference functions
def model_fn(model_dir):
    """
    This loads returns a Scikit-learn Classifier from a model.joblib file in 
    the SageMaker model directory model_dir.
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

def input_fn(request_body, request_content_type):
    pass

def predict_fn(input_object, model):
    pass


def output_fn(prediction, response_content_type):
    pass
