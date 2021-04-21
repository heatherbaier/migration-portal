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



GEOJSON_PATH = "./geoBoundariesSimplified-3_0_0-MEX-ADM2.geojson"
# DATA_PATH = "./mex_migration_allvars_subset.csv"
DATA_PATH = "./us_migration_allvars.csv"
MATCH_PATH = "./gB_IPUMS_match.csv"
MODEL_PATH = "./transfer_25epoch_weightedloss_us.torch"



with open(GEOJSON_PATH) as f:
    geodata_collection = geojson.load(f)



def map_column_names(var_names, df):
    for i in range(0, len(df.columns)):
        if df.columns[i] in var_names.keys():
            df = df.rename(columns = {df.columns[i]: var_names[df.columns[i]] })
    return df



def get_column_lists(df, var_names, grouped_vars):
    econ = df[grouped_vars['Economic']]
    econ = map_column_names(var_names, econ)
    econ = econ.columns
    
    demog = df[grouped_vars['Deomographic']]
    demog = map_column_names(var_names, demog)
    demog = demog.columns

    family = df[grouped_vars['Family']]
    family = map_column_names(var_names, family)
    family = family.columns

    employ = df[grouped_vars['Employment']]
    employ = map_column_names(var_names, employ)
    employ = employ.columns

    health = df[grouped_vars['Health']]
    health = map_column_names(var_names, health)
    health = health.columns

    edu = df[grouped_vars['Education']]
    edu = map_column_names(var_names, edu)
    edu = edu.columns

    hhold = df[grouped_vars['Household']]
    hhold = map_column_names(var_names, hhold)
    hhold = hhold.columns
    
    return econ, demog, family, health, edu, employ, hhold


def convert_to_pandas(geodata_collection, MATCH_PATH, DATA_PATH):

    # Normalize the geoJSON as a pandas dataframe
    df = json_normalize(geodata_collection["features"])

    # Get the B unique ID column (akgkjklajkljlk)
    df["B"] = df['properties.shapeID'].str.split("-").str[3]

    # Read in the dataframe for matching and get the B unique ID column
    match_df = pd.read_csv(MATCH_PATH)[['shapeID', 'MUNI2015']]
    match_df["B"] = match_df['shapeID'].str.split("-").str[3]

    # Read in the migration data
    dta = pd.read_csv(DATA_PATH)

    # Match the IPUMS ID's to the gB ID's
    ref_dict = dict(zip(match_df['B'], match_df['MUNI2015']))
    df['sending'] = df['B'].map(ref_dict)

    # Mix it all together
    merged = pd.merge(df, dta, on = 'sending')

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

    # # Mix it all together
    # merged = pd.merge(df, dta, on = 'sending')

    return dta



def predict_row(values_ar, X):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet50 = models.resnet50(pretrained=True)
    model = socialSigNoDrop.scoialSigNet_NoDrop(X=X, outDim = 1, resnet = resnet50).to(device)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    input = torch.reshape(torch.tensor(values_ar, dtype = torch.float32), (1, 219)).to(device)
    model.eval()
    pred = model(input, 1).detach().cpu().numpy()[0][0]

    return pred



def convert_features_to_geojson(merged):
    # # Make lists of all of the features we want available to the Leaflet map
    coords = merged['geometry.coordinates']
    types = merged['geometry.type']
    num_migrants = merged['num_persons_to_us']
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