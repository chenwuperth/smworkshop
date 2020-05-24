"""
inference script for both batch transform and endpoint

https://course.fast.ai/deployment_amzn_sagemaker.html
"""

import torch
from fastai.tabular import *

# loads the model into memory from disk and returns it
def model_fn(model_dir):
    learn = load_learner(model_dir, fname='model.pkl')
    return learn