"""
train.py can run on either your local machine or sagemaker AWS instance
It ideally does not include anything to do with SageMaker except some SM environment variable names

Adapted from https://docs.fast.ai/tabular.html
"""

import argparse
from pathlib import Path

import torch
from fastai.tabular import *

print('extracting arguments')
parser = argparse.ArgumentParser()

# hyperparameters sent by the client are passed as command-line arguments to the script.
# Data, model, and output directories
parser.add_argument('--model-dir', type=str,
                    default=os.environ.get('SM_MODEL_DIR'))
parser.add_argument('--train', type=str,
                    default=os.environ.get('SM_CHANNEL_TRAIN'))
parser.add_argument('--test', type=str,
                    default=os.environ.get('SM_CHANNEL_TEST'))
parser.add_argument('--train-file', type=str, default='adult.csv')
# in this script we ask user to explicitly name the target
parser.add_argument('--hidden_layer_1', type=int, default=200)
parser.add_argument('--hidden_layer_2', type=int, default=100)

args, _ = parser.parse_known_args()

# we simply ignore the dataset URL from S3 for now
#path = untar_data(URLs.ADULT_SAMPLE)
path = Path(args.train)

#df = pd.read_csv(path/'adult.csv')
df = pd.read_csv(os.path.join(args.train, args.train_file))
procs = [FillMissing, Categorify, Normalize]
dep_var = 'salary'
cat_names = ['workclass', 'education', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'native-country']
cont_names = ['age', 'fnlwgt', 'education-num']
test = TabularList.from_df(df.iloc[800:1000].copy(
), path=path, cat_names=cat_names, cont_names=cont_names)
data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
        .split_by_idx(list(range(800, 1000)))
        .label_from_df(cols=dep_var)
        .add_test(test)
        .databunch(num_workers=1))
learn = tabular_learner(data, layers=[args.hidden_layer_1, args.hidden_layer_2], emb_szs={
                        'native-country': 10}, metrics=accuracy)
learn.fit_one_cycle(2, 1e-2)
#return learn

print('validating model')
preds = []
for it in range(800, 1000):
    item = df.iloc[it]
    _, _, a = learn.predict(item)
    preds.append([a[0], a[1]])
preds = np.stack(preds)
preds = torch.from_numpy(preds)
df1 = df.iloc[800:1000].copy()
df1['salary'] = df1['salary'].astype('category')
cat_columns = df1.select_dtypes(['category']).columns
df1[cat_columns] = df1[cat_columns].apply(lambda x: x.cat.codes)
yt = df1['salary'].values[800:1000].astype(np.int32)
yt = torch.from_numpy(yt)
dice_acc = dice(preds, yt).numpy()
print(f'Dice accuracy: {dice_acc}')

# persist model
learn.export(f'{args.model_dir}/model.pkl')
