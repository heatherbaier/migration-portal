from flask import request, jsonify, Response
import torchvision.models as models
from sklearn import preprocessing
from pandas import json_normalize
import geopandas as gpd
import pandas as pd
import numpy as np
import torchvision
import importlib
import geojson
import folium
import torch
import flask
import json
import io

import socialSigNoDrop
importlib.reload(socialSigNoDrop)

from model.utils import *
from model.model import *
from model.modules import *
from model.aggregator import *
from model.encoder import *
from model.graphsage import *


# Path variables
GEOJSON_PATH = "./data/ipumns_simple_wgs_wdata8.geojson"
SHP_PATH = "./data/useforportal.shp"
DATA_PATH = "./data/mexico2010.csv"
MIGRATION_PATH = "./data/migration_data.json"
CORR_TABLE_PATH = "./data/corr_table.csv"
MATCH_PATH = "./data/gB_IPUMS_match.csv"
MODEL_PATH = "./trained_model/ram_8_50x50_0.75_model_best.pth.tar"
BORDER_STATIONS_PATH = "./data/border_stations7.geojson"
GRAPH_MODEL = "./trained_model/trained_graph_model.torch"
BAD_IDS = ["105", "115", "122", "126", "147", "153", "1622", "1684", "2027", "2043", "104", "1630", "113", "640", "400", "1631", "2054", "1693", "152", "1608"]

gdf = gpd.read_file(SHP_PATH)
gdf = gdf.dropna(subset = ["geometry"])
graph_id_dict = dict(zip(gdf["shapeID"].to_list(), [str(i) for i in range(0, len(gdf))]))
munis_available = gdf["shapeID"].to_list()

# Set up graph model
graph_checkpoint = torch.load(GRAPH_MODEL)
graph_checkpoint = graph_checkpoint["model_state_dict"]

with open("./data/graph.json") as g:
    graph = json.load(g)

x, adj_lists, y = [], {}, []

a = 0
for muni_id, dta in graph.items():
    x.append(dta["x"])
    y.append(dta["label"])
    adj_lists[str(a)] = dta["neighbors"]
    a += 1
    
x = np.array(x)
y = np.expand_dims(np.array(y), 1)

agg = MeanAggregator(features = x, gcn = False)
enc = Encoder(features = x, feature_dim = x.shape[1], embed_dim = 128, adj_lists = adj_lists, aggregator = agg)
model = SupervisedGraphSage(num_classes = 1, enc = enc)
model.load_state_dict(graph_checkpoint)
model.eval()


def predict(graph, selected_muni_ref_dict, new_census_vals, selected_municipalities):

    x, adj_lists, y = [], {}, []

    a = 0
    for muni_id, dta in graph.items():
        if muni_id in selected_muni_ref_dict.values():
            cur_x = dta["x"]
            cur_x = cur_x[0:len(cur_x) - 202]
            [cur_x.append(v) for v in new_census_vals[muni_id]]
            x.append(cur_x)
        else:
            x.append(dta["x"])
        y.append(dta["label"])
        adj_lists[str(a)] = dta["neighbors"]
        a += 1
        
    x = np.array(x)
    y = np.expand_dims(np.array(y), 1)   

    agg = MeanAggregator(features = x, gcn = False)
    enc = Encoder(features = x, feature_dim = x.shape[1], embed_dim = 128, adj_lists = adj_lists, aggregator = agg)
    model = SupervisedGraphSage(num_classes = 1, enc = enc)
    model.load_state_dict(graph_checkpoint)

    predictions = []
    for muni in selected_municipalities:
        muni_ref = graph_id_dict[muni]            
        print("cur muni, muni_ref: ", muni, muni_ref)
        input = [muni_ref]
        prediction = int(model.forward(input).item())
        predictions.append(prediction)

    
    return predictions



def prep_dataframes(dta, request, selected_municipalities):

    dta_selected = dta[dta['GEO2_MX'].isin([int(i) for i in selected_municipalities])]
    num_og_migrants = dta_selected['sum_num_intmig'].sum()

    # Parse the edited input variables and switch all of the 0's in percent_changes to 1 (neccessary for multiplying later on)
    column_names = request.json['column_names']
    # print("COLUMN NAMES: ", column_names)
    percent_changes = request.json['percent_changes']
    # print("P", column_names)
    percent_changes = [float(i) - 100 if i != '100' else '1' for i in percent_changes]
    print("PERCENT CHANGES: ", percent_changes)




    # Open the var_map JSON and reverse the dictionary
    with open("./var_map.json", "r") as f2:
        var_names = json.load(f2)
    reverse_var_names = dict([(value, key) for key, value in var_names.items()])

    # Change the 'pretty' variable names back to their originals so we can edit the dataframe
    column_names = [reverse_var_names[i] if i in reverse_var_names.keys() else i for i in column_names]
    
    edited_variables = [column_names[i] for i in range(0, len(column_names)) if percent_changes[i] != '1']
    edited_p_changes = [percent_changes[i] / 2 for i in range(0, len(column_names)) if percent_changes[i] != '1']
    edited_p_changes = [i * .01 for i in edited_p_changes]

    print("EDITED VARIABLES: ", edited_variables)
    print("EDITED P CHANGES: ", edited_p_changes)

    corr_table = pd.read_csv(CORR_TABLE_PATH)

    corr_table = corr_table[corr_table['index'].isin(edited_variables)]
    corr_table = corr_table.set_index(['index']).reindex(edited_variables)#.reset_index()
    corr_dict = dict(corr_table.multiply(edited_p_changes, axis='rows').mean())

    print(corr_dict)  

    # Multiply the columns by their respective percent changes
    for i in range(0, len(column_names)):

        # Multiply the percetnage change by .01 (to get it in percent format), then 
        # multiply it by the original variable value to get the amount we need to change it by
        percentage = abs(float(percent_changes[i])) * .01
        change = percentage * dta_selected[column_names[i]]

        # ...then add or subtract as needed
        if float(percent_changes[i]) < 0:
            dta_selected[column_names[i]] = dta_selected[column_names[i]] - change
        elif float(percent_changes[i]) > 0:
            dta_selected[column_names[i]] = dta_selected[column_names[i]] + change
        else:
            dta_selected[column_names[i]] = dta_selected[column_names[i]] + corr_dict[column_names[i]]

    # Get a data frame with all of the data that wasn't edited
    dta_dropped = dta[~dta['GEO2_MX'].isin(selected_municipalities)]

    # Then re-append the updated data to the larger dataframe incorporating user input
    dta_appended = dta_dropped.append(dta_selected)
    dta_appended = dta_appended.drop(['GEO2_MX'], axis = 1)

    # dta_appended = dta_appended.drop(['Unnamed: 0', 'sending'], axis = 1)
    dta_appended = dta_appended.fillna(0)
    dta_appended = dta_appended.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    with open("./us_vars.txt", "r") as f:
        vars = f.read().splitlines()
    vars = [i for i in vars if i in dta_appended.columns]
    dta_appended = dta_appended[vars]

    # Scale the data frame for the model
    X = dta_appended.loc[:, dta_appended.columns != "sum_num_intmig"].values
    mMScale = preprocessing.MinMaxScaler()
    X = mMScale.fit_transform(X)


    return dta_appended, dta_selected, dta_dropped, num_og_migrants, X


# Read in spatial data
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


def convert_features_to_geojson(merged):
    # # Make lists of all of the features we want available to the Leaflet map
    coords = merged['geometry.coordinates']
    types = merged['geometry.type']
    num_migrants = merged['sum_num_intmig']
    shapeIDs = merged['properties.shapeID']
    shapeNames = merged['properties.ipumns_simple_wgs_wdata_geo2_mx1960_2015_ADMIN_NAME']

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