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
from io import StringIO
import numpy as np
import json
import pandas as pd
from sagemaker_containers.beta.framework import worker, encoders

# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    'sex', # M, F, and I (infant)
    'length', # Longest shell measurement
    'diameter', # perpendicular to length
    'height', # with meat in shell
    'whole_weight', # whole abalone
    'shucked_weight', # weight of meat
    'viscera_weight', # gut weight (after bleeding)
    'shell_weight'] # after being dried

label_column = 'rings'

def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), 
                         header=None)

        if len(df.columns) == len(feature_columns_names) + 1:
            # This is a labelled example, includes the ring label
            df.columns = feature_columns_names + [label_column]
        elif len(df.columns) == len(feature_columns_names):
            # This is an unlabelled example.
            df.columns = feature_columns_names

        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), accept, mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)
    else:
        raise Exception("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    features = model.transform(input_data)

    if label_column in input_data:
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data[label_column], axis=1)
    else:
        # Return only the set of features
        return features


def model_fn(model_dir):
    """Deserialize fitted model
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor