"""
train.py can run on either your local machine or sagemaker AWS instance
It ideally does not include anything to do with SageMaker except some SM environment variable names

Pytorch model (credit) - https://github.com/talasinski/pytorch-Boston-Housing-data/blob/master/BostonHousing.py
"""

import argparse
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

if __name__ == '__main__':
    print('extracting arguments')
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument('--n-epochs', type=int, default=10)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str,
                        default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='boston_train.csv')
    parser.add_argument('--test-file', type=str, default='boston_test.csv')
    # in this script we ask user to explicitly name features
    parser.add_argument('--features', type=str)
    # in this script we ask user to explicitly name the target
    parser.add_argument('--target', type=str)

    args, _ = parser.parse_known_args()

    print('reading data')
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    print('building training and testing datasets')
    X_train = train_df[args.features.split()].values
    X_test = test_df[args.features.split()].values
    y_train = train_df[args.target].values
    y_test = test_df[args.target].values

    # train
    print('training model')

    num_train, dim = X_train.shape
    torch.set_default_dtype(torch.float64)
    net = nn.Sequential(
        nn.Linear(dim, 50, bias=True), nn.ELU(),
        nn.Linear(50, 50, bias=True), nn.ELU(),
        nn.Linear(50, 50, bias=True), nn.Sigmoid(),
        nn.Linear(50, 1)
    )
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=.0005)

    num_epochs = args.n_epochs
    print(type(y_train))
    y_train_t = torch.from_numpy(y_train).clone().reshape(-1, 1)
    x_train_t = torch.from_numpy(X_train).clone()
    losssave = []
    stepsave = []

    for i in range(num_epochs):
        y_hat = net(x_train_t)
        loss = criterion(y_train_t, net(x_train_t))
        losssave.append(loss.item())
        stepsave.append(i)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if (i % 100 == 0):
            print(f'Epoch {i}, loss = {loss} ')

    # print abs error
    print('validating model')
    ypred = net(torch.from_numpy(X_test).detach())
    abs_err = np.abs(ypred.detach().numpy() - y_test)

    # print couple perf metrics
    for q in [10, 50, 90]:
        print('AE-at-' + str(q) + 'th-percentile: '
              + str(np.percentile(a=abs_err, q=q)))

    # persist model
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(net.state_dict(), f)
    
    print('model persisted at ' + os.path.join(args.model_dir, 'model.pth'))
