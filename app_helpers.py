from flask import request, jsonify, Response
import torchvision.models as models
from sklearn import preprocessing
from pandas import json_normalize
import geopandas as gpd
import pandas as pd
import numpy as np
import importlib
import geojson
import folium
import torch
import flask
import json
import io

import socialSigNoDrop
importlib.reload(socialSigNoDrop)



GEOJSON_PATH = "./data/ipumns_simple_wgs.geojson"
SHP_PATH = "./data/ipumns_shp.shp"
DATA_PATH = "./data/mexico2010.csv"
MIGRATION_PATH = "./data/migration_data.json"
MATCH_PATH = "./data/gB_IPUMS_match.csv"
MODEL_PATH = "./trained_model/notransfer_50epoch_weightedloss_us.torch"
BORDER_STATIONS_PATH = "./data/border_stations5.geojson"
IMAGERY_DIR = "./imagery/"
ISO = "MEX"
IC = "LANDSAT/LT05/C01/T1"



with open(GEOJSON_PATH) as f:
    geodata_collection = geojson.load(f)


with open(BORDER_STATIONS_PATH) as bs:
    border_stations = geojson.load(bs)


def map_column_names(var_names, df):
    for i in range(0, len(df.columns)):
        if df.columns[i] in var_names.keys():
            df = df.rename(columns = {df.columns[i]: var_names[df.columns[i]] })
    return df


def get_column_lists(df, var_names, grouped_vars):
    e_vars = [i for i in grouped_vars['Economic'] if i in df.columns]
    econ = df[e_vars]
    econ = map_column_names(var_names, econ)
    econ = econ.columns
    
    d_vars = [i for i in grouped_vars['Deomographic'] if i in df.columns]
    demog = df[d_vars]
    demog = map_column_names(var_names, demog)
    demog = demog.columns

    f_vars = [i for i in grouped_vars['Family'] if i in df.columns]
    family = df[f_vars]
    family = map_column_names(var_names, family)
    family = family.columns

    em_vars = [i for i in grouped_vars['Employment'] if i in df.columns]
    employ = df[em_vars]
    employ = map_column_names(var_names, employ)
    employ = employ.columns

    h_vars = [i for i in grouped_vars['Health'] if i in df.columns]
    health = df[h_vars]
    health = map_column_names(var_names, health)
    health = health.columns

    edu_vars = [i for i in grouped_vars['Education'] if i in df.columns]
    edu = df[edu_vars]
    edu = map_column_names(var_names, edu)
    edu = edu.columns

    hh_vars = [i for i in grouped_vars['Household'] if i in df.columns]
    hhold = df[hh_vars]
    hhold = map_column_names(var_names, hhold)
    hhold = hhold.columns
    
    return econ, demog, family, health, edu, employ, hhold


def convert_to_pandas(geodata_collection, MATCH_PATH, DATA_PATH):

    # Normalize the geoJSON as a pandas dataframe
    df = json_normalize(geodata_collection["features"])
    df = df.rename(columns = {"properties.shapeID": "shapeID"})
    df["shapeID"] = df["shapeID"].astype(int)

    # Read in the migration data
    dta = pd.read_csv(DATA_PATH)
    dta = dta.rename(columns = {"GEO2_MX": "shapeID"})

    # Mix it all together
    merged = pd.merge(df, dta, on = 'shapeID')

    return merged



def switch_column_names(MATCH_PATH, DATA_PATH):

    # Read in the dataframe for matching and get the B unique ID column
    match_df = pd.read_csv(MATCH_PATH)[['shapeID', 'MUNI2015']]
    match_df["B"] = match_df['shapeID'].str.split("-").str[3]

    # Read in the migration data
    dta = pd.read_csv(DATA_PATH)

    # Match the IPUMS ID's to the gB ID's
    ref_dict = dict(zip(match_df['MUNI2015'], match_df['B']))
    dta['sending'] = dta['sending'].map(ref_dict)

    print(dta.head())

    # # Mix it all together
    # merged = pd.merge(df, dta, on = 'sending')

    return dta



def get_muni_names(selected_municipalities):
    # Read in the dataframe for matching and get the B unique ID column
    match_df = pd.read_csv(MATCH_PATH)[['shapeID', 'shapeName', 'MUNI2015']]
    match_df["B"] = match_df['shapeID'].str.split("-").str[3]

    match_df = match_df[match_df["B"].isin(selected_municipalities)]
    return match_df['shapeName'].to_list()


def predict_row(values_ar, X, muni):

    with open('status.json', 'w') as outfile:
        json.dump({'status': "Status - Predicting " + str(muni)}, outfile)

    print("SHAPE IN FUNC HERE: ", values_ar.shape)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet50 = models.resnet50(pretrained=True)
    model = socialSigNoDrop.scoialSigNet_NoDrop(X=X, outDim = 1, resnet = resnet50).to(device)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    input = torch.reshape(torch.tensor(values_ar, dtype = torch.float32), (1, 202)).to(device)
    model.eval()
    pred = model(input, 1).detach().cpu().numpy()[0][0]

    return pred



def convert_features_to_geojson(merged):
    # # Make lists of all of the features we want available to the Leaflet map
    coords = merged['geometry.coordinates']
    types = merged['geometry.type']
    num_migrants = merged['sum_num_intmig']
    shapeIDs = merged['sending']
    shapeNames = merged['properties.shapeName']

    # For each of the polygons in the data frame, append it and it's data to a list of dicts to be sent as a JSON back to the Leaflet map
    features = []
    for i in range(0, len(merged)):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": types[i],
                "coordinates": coords[i]
            },
            "properties": {'num_migrants': num_migrants[i],
                           'shapeID': shapeIDs[i],
                           'shapeName': shapeNames[i]
                          }
        })

    return features