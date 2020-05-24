"""
inference script for both batch transform and endpoint

https://course.fast.ai/deployment_amzn_sagemaker.html
"""

import torch
from fastai.tabular import *
import pandas as pd

# loads the model into memory from disk and returns it
def model_fn(model_dir):
    learn = load_learner(model_dir, fname='model.pkl')
    return learn

def input_fn(request_body, request_content_type):
    with open('local_csv', 'w') as fout:
        fout.write(request_body)
    df = pd.read_csv('local_csv')
    return df

def predict_fn(df, learn):
    size = df.shape[0]
    preds = []
    for it in range(size):
        item = df.iloc[it]
        _, _, a = learn.predict(item)
        preds.append(a[1])
    return np.array(preds).astype(np.float32)