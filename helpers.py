#!/usr/bin/env python

from sklearn import preprocessing
from copy import deepcopy
from flask import request
import geopandas as gpd
import pandas as pd
import numpy as np
import importlib
import sklearn
import geojson
import folium
import random
import flask
import torch
import math


import socialSig
importlib.reload(socialSig)
from helpers import *



# Data prep
def prep_data(match_path, mig_path, gdf_path):

    # match
    match = pd.read_csv(match_path)
    match["B"] = match["shapeID"].str.split("-").str[3]
    match = match[['shapeID', 'MUNI2015', 'B']]
    match.columns = ['shapeID', 'sending', 'B']

    # mig
    mig = pd.read_csv(mig_path)
    mig = pd.merge(match, mig, on = 'sending')
    mig = mig[['US_MIG_05_10', 'B', 'sending']]

    # gdf
    gdf = gpd.read_file(gdf_path)#
    gdf['geometry'] = gdf['geometry'].simplify(8)
    gdf['col'] = [i for i in range(len(gdf))]
    gdf['B'] = match["shapeID"].str.split("-").str[3]
    gdf = pd.merge(gdf, mig, on = 'B')

    gdf_json = gdf.to_json()

    return gdf, gdf_json, mig


def style_function(feature):
    return {
        "stroke": 0,
        "color": "#FFFFFF"
    }


def get_centroids(gdf):
    cur = gdf[gdf['sending'] == int(request.json['adm_id'])]
    cur['centroid'] = cur['geometry'].centroid#.split(",")#[0]
    cur['centroid'] = cur['centroid'].astype(str)

    lat = cur['centroid'].to_list()[0].split(" ")[1].replace("(", "")
    lng = cur['centroid'].to_list()[0].split(" ")[2].replace(")", "")

    return lat, lng

MATCH_PATH = "./gB_IPUMS_match.csv"
MIG_PATH = "./us_migration.csv"
GDF_PATH = "/home/hbaier/Desktop/portal/data/MEX/MEX_ADM2_fixedInternalTopology.shp"

gdf, gdf_json, mig = prep_data(MATCH_PATH, MIG_PATH, GDF_PATH)



def scale(x, out_range=(0, 29)):
    '''
    Takes as input the coordinate weights and scales them between 0 and len(weights)
    '''
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    to_ret = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return to_ret.astype(int)


def construct_indices(weights, dim, length):
    '''
    The coordinate weights are between 0-len(weights) but the size of X is len(weights) * batch size so the torch.taken
    function will only take items at indices between  0 & len(weights) meaning only the first item in the batch. This function
    adds len(weights) to each index so taken grabs from every batch
    ^^ fix that explanation yo lol
    '''
    indices = []
    weights = scale(weights.clone().detach().numpy())
    print(weights.size)
    for i in range(0, dim):
        to_add = i * length
        cur_indices = [i + to_add for i in weights]
        indices.append(cur_indices)
    return torch.tensor(indices, dtype = torch.int64)

def scale_noOverlap(x, out_range=(0, 29)):
    '''
    Takes as input the coordinate weights and scales them between 0 and len(weights)
    Dan removed int rounding from this one.
    '''
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    to_ret = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return to_ret

def construct_noOverlap_indices(weights, dim, length):
    '''
    The coordinate weights are between 0-len(weights) but the size of X is len(weights) * batch size so the torch.taken
    function will only take items at indices between  0 & len(weights) meaning only the first item in the batch. This function
    adds len(weights) to each index so taken grabs from every batch
    Dan then modified whatever the above was to ensure the rounding only occurs to available indices, precluding
    drop out. 
    ^^ fix that explanation yo lol
    '''
    indices = []
    weights = scale_noOverlap(weights.clone().detach().numpy())
    indices = dim*[[x for _,x in sorted(zip(weights,range(0,length)))]]
    for i in range(0,len(indices)):
        indices[i] = [x+(i*length) for x in indices[i]]
    return torch.tensor(indices, dtype = torch.int64)


def update_function(param, grad, loss, learning_rate):
    '''
    Calculates the new coordinate weights based on the LR and gradient
    '''
    return param - learning_rate * grad.mean(axis = 0)


def mae(real, pred):
    '''
    Calculates MAE of an epoch
    '''
    return torch.abs(real - pred).mean()


def show_image(best_epoch):
    '''
    Takes as input an epoch number and displays the SocialSig from that epoch
    '''
    df = pd.read_csv("./figs/im" + str(best_epoch) + ".csv")
    df["0"] = df["0"].str.split("(").str[1].str.split(",").str[0].astype(float)
    plt.imshow(np.reshape(np.array(df["0"]), (10, 10)))


WEIGHTS_PATH = "./trained_weights_nosending5.torch"

def prep_model(WEIGHTS_PATH, DF_PATH):

    devSet = pd.read_csv(MIG_PATH)
    devSet = devSet.loc[:, ~devSet.columns.str.contains('^Unnamed')]
    devSet = devSet.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    devSet = devSet.dropna(axis=1)
    devSet = devSet.drop(['sending'], axis = 1)

    y = torch.Tensor(devSet['US_MIG_05_10'].values)
    X = devSet.loc[:, devSet.columns != "US_MIG_05_10"].values

    mMScale = preprocessing.MinMaxScaler()
    X = mMScale.fit_transform(X)

    model = socialSig.SocialSigNet(X=X, outDim = 1)

    checkpoint = torch.load(WEIGHTS_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    return model



model = prep_model(WEIGHTS_PATH, MIG_PATH)



def pred_municipality(sending_id, model, MIG_PATH, new_vals):
    mig_cur = pd.read_csv(MIG_PATH)
    mig_cur = mig_cur.loc[:, ~mig_cur.columns.str.contains('^Unnamed')]
    mig_cur = mig_cur.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    mig_cur = mig_cur.dropna(axis=1)
    mig_cur = mig_cur.drop(["sending", "US_MIG_05_10"], axis = 1)

    # mig_cur = mig_cur[mig_cur['sending'] != sending_id]
    # print(mig_cur.head())
    # # mig_cur.loc[-1] = new_vals
    
    # mig_cur = mig_cur.append(new_vals)
    # print(mig_cur.tail())

    # sending_index = mig_cur.index[mig_cur.sending == sending_id]

    mMScale = preprocessing.MinMaxScaler()
    mMScale.fit(mig_cur)#.values

    imput = mMScale.transform([new_vals])

    print(imput)

    imput = torch.reshape(torch.tensor(imput, dtype = torch.float32), (1, 1, 29))

    # print(model(imput, 1))

    # checkpoint = torch.load(WEIGHTS_PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    return model(imput, 1)





