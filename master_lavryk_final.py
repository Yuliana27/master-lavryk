# -*- coding: utf-8 -*-

import os
import json
import gzip
import shutil
import pandas as pd
from urllib.request import urlopen
import numpy as np
import seaborn as sns
import random
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from google.colab import drive

import networkx as nx
from networkx.algorithms import bipartite
from sentence_transformers import SentenceTransformer
from numpy import inf
from torch import Tensor
import recmetrics
from collections import defaultdict
from termcolor import colored
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.data import (InMemoryDataset, HeteroData, download_url,
                                  extract_zip)
from early_stopping import early_stopping
from graph_data import BeautyGraph
from gnn_model import Model
from loss_curve import loss_curve

#Conecting to drive

drive.mount('/content/drive')

with open('/content/drive/MyDrive/master_final/reviews_Beauty_5.json', 'rb') as f_in:
    with gzip.open('/content/drive/MyDrive/master_final/reviews_Beauty_5.json.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

#Data Extracting

#--Metadata
meta_beauty = []
count = 0

with gzip.open("/content/drive/MyDrive/master_final/meta_Beauty.json.gz", mode="rt") as f:
    for l in f:
      try:
        string = l.strip().replace("'", "\"")
        meta_beauty.append(json.loads(r'{string}'.format(string = string)))
      except:
        pass
print(meta_beauty[0])
print(len(meta_beauty))

#--Review Data
reviews_beauty = []
i=0
with gzip.open('/content/drive/MyDrive/master_final/reviews_Beauty_5.json.gz', mode="rt") as f:
    count=0
    for l in f:
      try:
        i+=1
        ly=""
        ly=json.loads(l.strip())
        reviews_beauty.append(ly)
      except:
        pass
print(len(reviews_beauty))
print(reviews_beauty[0])

#Data Preprocessing

print('Length of Metadata Beauty Dataset: ', len(meta_beauty), '\n')
print('Length of Review Beauty Dataset: ', len(reviews_beauty))

#--Metadata

meta_beauty_df = pd.DataFrame.from_dict(meta_beauty)

meta_beauty_df = meta_beauty_df.drop_duplicates(subset='asin', keep='first')
meta_beauty_df = meta_beauty_df.dropna(subset=['categories']) # no nulls

meta_beauty_df['class'] = meta_beauty_df['categories'].astype(str).str.rsplit(',').str[-1]
meta_beauty_df['class'] = meta_beauty_df['class'].replace('\]', '', regex=True)
meta_beauty_df['class']=meta_beauty_df['class'].replace('\[', '',regex=True)
meta_beauty_df['class'] = meta_beauty_df['class'].replace('\'', '', regex=True)

print('Number of categories ', len(meta_beauty_df['class'].unique()), '\n')
print(meta_beauty_df['class'].unique())

meta_beauty_df.to_pickle('/content/drive/MyDrive/master_final/meta_beauty.pkl')

#--Review Data

reviews_beauty_df = pd.DataFrame.from_dict(reviews_beauty)

reviews_beauty_df= reviews_beauty_df.dropna(subset=['asin'])

reviews_beauty_df.to_pickle('/content/drive/MyDrive/master_final/reviews_beauty.pkl')

#Data Combining

all_beauty_df = pd.merge(reviews_beauty_df,meta_beauty_df, how='inner', on='asin')

all_beauty_df.to_pickle('/content/drive/MyDrive/master_final/all_beauty.pkl')

#Data Selection (necessary for future graph structure)

beauty_df = all_beauty_df[['reviewerID', 'asin', 'overall', 'class']]
beauty_df = beauty_df.rename(columns={'reviewerID': 'reviewer_id', 'overall': 'rating'})
print(beauty_df.rating.unique())

#Data Size Reduction (for the first experiments)

#--Users with 10 or more Transactions
users_selection = beauty_df.groupby('reviewer_id')['asin'].count().reset_index(drop=False)
users_selection['flag'] = users_selection['asin'].apply(lambda x: 1 if x >=10 else 0) # minimum 10 transactions
print(f'Number of users in transaction criteria : {len(users_selection)}')

beauty_df_with_flag = beauty_df.merge(users_selection, on ='reviewer_id')
beauty_df_selected = beauty_df_with_flag[beauty_df_with_flag['flag'] == 1] # only with 10 or more

beauty_df_selected = beauty_df_selected[['reviewer_id', 'asin_x', 'class', 'rating']]
beauty_df_selected = beauty_df_selected.rename(columns={'asin_x': 'asin'})

print('Length of dataset before reduction', len(beauty_df), '\n')
print('Length of dataset after reduction', len(beauty_df_selected))

#Encoding to Numeric Format

user_le = LabelEncoder()
beauty_df_selected['user_id_encoded'] = user_le.fit_transform(beauty_df_selected['reviewer_id'].values)

product_le = LabelEncoder()
beauty_df_selected['product_id_encoded'] = product_le.fit_transform(beauty_df_selected['asin'].values)

beauties = beauty_df_selected.groupby(['product_id_encoded',
                                          'class','asin'])['rating'].count().reset_index(drop=False)

#Data Transformation into Nodes and Edges (heterogeneous structure)

dataset = BeautyGraph(beauties, beauty_df_selected)
data = dataset[0]
print(data)

#Graph Building
data = T.ToUndirected()(data)
del data['beauties', 'rev_rates', 'user'].edge_label

#Data Splitting (before model training)

train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'rates', 'beauties')],
    rev_edge_types=[('beauties', 'rev_rates', 'user')],
)(data)


#Model Building

weight = torch.bincount(train_data['user', 'beauties'].edge_label)
weight = weight.max() / weight
weight[weight == -inf] = 0
weight[weight == inf] = 0

def weighting(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    res = (weight * (pred - target.to(pred.dtype)).pow(2)).mean()
    return res

model = Model(data, hidden_channels=32)

with torch.no_grad():
    model.encoder(train_data.x_dict, train_data.edge_index_dict)

print(model)

#Model Training

def train(optimizer):
    model.train()
    optimizer.zero_grad()

    pred = model(train_data.x_dict, train_data.edge_index_dict, train_data['user', 'beauties'].edge_label_index)
    target = train_data['user', 'beauties'].edge_label
    loss = weighting(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict, data['user', 'beauties'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['user', 'beauties'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse), pred

def training_model(epoch_number, lr, patience):
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  train_pred = defaultdict()
  test_pred = defaultdict()
  val_pred = defaultdict()

  train_losses = []
  val_losses = []
  test_losses = []

  patience = patience
  count_paitence = 0

  for epoch in range(0, epoch_number):
      loss = train(optimizer)

      train_loss, train_pred[epoch] = test(train_data)

      val_loss, val_pred[epoch] = test(val_data)

      test_loss, test_pred[epoch] = test(test_data)

      train_losses.append(train_loss)
      val_losses.append(val_loss)
      test_losses.append(test_loss)

      print(f'Epoch: {epoch}, Loss: {loss:.4f}, Train: {train_loss:.4f} ,Val: {val_loss:.4f}, Test: {test_loss:.4f}')

      if epoch == 0:
          previous_loss = val_loss

      early_stop_check = early_stopping(epoch, val_loss, previous_loss, count_paitence, patience)
      if early_stop_check[0] =='stop':
          break
      else:
          previous_loss = early_stop_check[0]
          count_paitence = early_stop_check[1]

  return train_losses, val_losses, test_losses, test_pred

#Experiment
#--Small dataset (users with at least 10 transactions)
#--EPOCHS = 150

train_losses_150, val_losses_150, test_losses_150, test_pred = training_model(150, 0.01, 5)
loss_curve(train_losses_150, val_losses_150, test_losses_150)

#Example

user_mapping = {idx: i for i, idx in enumerate(beauty_df_selected['reviewer_id'].unique())}
beauty_mapping = {idx: i for i, idx in enumerate(beauties['product_id_encoded'])}

USERID = int(15)
#Num of recommendations
NUM_BEAUTIES = int(10) 

num_beauties = len(data['beauties'].x)
row = torch.tensor([USERID] * num_beauties)
col = torch.arange(num_beauties)
edge_label_index = torch.stack([row, col], dim=0)
pred = model(data.x_dict, data.edge_index_dict, edge_label_index)
pred = pred.clamp(min=0, max=5)
idx_max = torch.topk(pred, NUM_BEAUTIES).indices
print('Recommendations (Beauty Products) - User with ID ' + str(USERID))
for i in idx_max:
    beautyId = beauty_mapping[int(i)]
    print(beauties.loc[beautyId]['asin'], beauties.loc[beautyId]['class'])

#Model Evaluation

test_actual = np.vstack((test_data['user','beauties'].edge_label_index,test_data['user','beauties'].edge_label))
test_predicted = pd.DataFrame(np.vstack((test_actual, test_pred.detach().numpy())).T,
                              columns = ['UserID','BeautyID','actual_rating','predicted_rating'])

test_predicted['UserID'] = test_predicted['UserID'].apply(lambda x: int(x))
test_predicted['BeautyID'] = test_predicted['BeautyID'].apply(lambda x: int(x))
test_predicted['predicted_rating'] = test_predicted['predicted_rating'].apply(lambda x: round(x,0))
test_predicted.to_pickle('/content/drive/MyDrive/master/test_predicted.pkl')

user_pred_true = defaultdict(list)
for i in range(0, len(test_predicted)):
    actual_rating = test_predicted.loc[i,'actual_rating']
    predicted_rating =   test_predicted.loc[i,'predicted_rating']
    userId = test_predicted.loc[i,'UserID']
    user_pred_true[userId].append((predicted_rating, actual_rating))

threshold = 4
precision = dict()
recall = dict()
avg_precision = dict()
avg_recall = dict()
for k in range(1,11):
    for user_id, user_ratings in user_pred_true.items():

        user_ratings.sort(key=lambda x: x[0], reverse=True)
        count_relevant = sum((actual_rating >= threshold) for (predicted_rating, actual_rating) in user_ratings)
        count_recommended_k = sum((predicted_rating >= threshold) for (predicted_rating, actual_rating) in user_ratings[:k])
        count_relevant_and_recommended_k = sum(
            ((actual_rating >= threshold) and (predicted_rating >= threshold))
            for (predicted_rating, actual_rating) in user_ratings[:k]
        )

        precision[user_id] = count_relevant_and_recommended_k /count_recommended_k if count_recommended_k != 0 else 0
        recall[user_id] = count_relevant_and_recommended_k /count_relevant if count_relevant != 0 else 0

    avg_precision[k] = sum(prec for prec in precision.values()) / len(precision)
    avg_recall[k] = sum(rec for rec in recall.values()) / len(recall)

avg_recall_df = pd.DataFrame(list(avg_recall.items()),columns = ['k','avg_recall'])
avg_precision_df = pd.DataFrame(list(avg_precision.items()),columns = ['k','avg_precision'])

#Model Results

preds_ = model(test_data.x_dict, test_data.edge_index_dict, test_data['user', 'beauties'].edge_label_index)
print(preds_)

#Recall

fig = plt.figure(figsize=(10, 8)) 
sns.set(style='darkgrid', )
sns.lineplot(x="k", y="avg_recall", data=avg_recall_df)
plt.ylim(0, 1)
plt.title("Average Recall@K", fontsize=14)
plt.show()

#Precision

fig = plt.figure(figsize=(10, 8))
sns.set(style='darkgrid', )
sns.lineplot(x="k", y="avg_precision", data=avg_precision_df)
plt.ylim(0, 1)
plt.title("Average Precision@K", fontsize=14)
plt.show()