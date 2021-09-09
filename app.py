from flask import request, jsonify, Response, send_from_directory
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
import os

import landsat_prep as lp
import geograph as gg

import socialSigNoDrop
importlib.reload(socialSigNoDrop)
from app_helpers import *
from model.utils import *
from model.model import *
from model.modules import *
from model.aggregator import *
from model.encoder import *
from model.graphsage import *

# Create the application.
APP = flask.Flask(__name__)

with open('status.json', 'w') as outfile:
    json.dump({'status': "Startup"}, outfile)


@APP.route('/', methods=['GET','POST'])
def index():

    # Read in census data
    df = pd.read_csv(DATA_PATH)

    # Read in migration data
    with open(MIGRATION_PATH) as m:
        mig_data = json.load(m)

    total_migrants = sum(list(mig_data.values()))
    municipality_ids = list(mig_data.keys())

    df['avg_age_weight'] = df['avg_age'] * df['sum_num_intmig']
    avg_age = df['avg_age_weight'].sum() / df['sum_num_intmig'].sum()

    # Open the variables JSON and the JSON containing the readable translation of the variables
    with open("./vars.json", "r") as f:
        grouped_vars = json.load(f)

    with open("./var_map.json", "r") as f2:
        var_names = json.load(f2)

    # Get all of the variables to send to Flask
    econ, demog, family, health, edu, employ, hhold = get_column_lists(df, var_names, grouped_vars)

    # Merry Christmas HTML
    return flask.render_template('index.html', 
                                  municipality_ids = municipality_ids, 
                                  econ_data = econ,
                                  demog_data = demog,
                                  family_data = family,
                                  health_data = health,
                                  edu_data = edu,
                                  employ_data = employ,
                                  hhold_data = hhold,
                                  total_migrants = f'{int(total_migrants / 5):,}',
                                  avg_age = round(avg_age, 2))




@APP.route('/geojson-features', methods=['GET'])
def get_all_points():

    # Convert the geoJSON to a dataframe and merge it to the migration data
    feature_df = convert_to_pandas(geodata_collection, MATCH_PATH, DATA_PATH)
    print(feature_df.columns)
    feature_df['sum_num_intmig'] = feature_df['sum_num_intmig'].fillna(0)
    
    # Make lists of all of the features we want available to the Leaflet map
    coords = feature_df['geometry.coordinates']
    types = feature_df['geometry.type']
    num_migrants = feature_df['sum_num_intmig']
    shapeIDs = feature_df['shapeID']
    # shapeNames = feature_df["properties.geo2_mx1960_2015_ADMIN_NAME"]
    shapeNames = feature_df["properties.ipumns_simple_wgs_wdata_geo2_mx1960_2015_ADMIN_NAME"]

    # For each of the polygons in the data frame, append it and it's data to a list of dicts to be sent as a JSON back to the Leaflet map
    features = []
    for i in range(0, len(feature_df)):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": types[i],
                "coordinates": coords[i]
            },
            "properties": {'num_migrants': num_migrants[i],
                           'shapeID': str(shapeIDs[i]),
                           'shapeName': shapeNames[i]
                          }
        })

    print('done jsonifying')

    return jsonify(features)



@APP.route('/border-features', methods=['GET'])
def get_border_features():

    feature_df = json_normalize(border_stations["features"])

    print(feature_df.columns)
    
    # Make lists of all of the features we want available to the Leaflet map
    coords = feature_df['geometry.coordinates']
    types = feature_df['geometry.type']
    num_migrants = feature_df['properties.total_migrants'].astype(float)
    shapeIDs = feature_df['properties.portname']
    # shapeNames = feature_df['properties.shapeName']

    # For each of the polygons in the data frame, append it and it's data to a list of dicts to be sent as a JSON back to the Leaflet map
    features = []
    for i in range(0, len(feature_df)):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": types[i],
                "coordinates": coords[i]
            },
            "properties": {
                           'shapeID': str(shapeIDs[i]),
                           'num_migrants': num_migrants[i]
                          }
        })

    return jsonify(features)



@APP.route('/border-sectors', methods=['GET'])
def get_border_sectors():

    coords_df = pd.read_csv("./data/sector_centroids.csv")

    x = coords_df['xcoord']
    y = coords_df['ycoord']

    coords = [(x[i], y[i]) for i in range(len(x))]
    names = coords_df['sector'].to_list()

    # For each of the polygons in the data frame, append it and it's data to a list of dicts to be sent as a JSON back to the Leaflet map
    features = []
    for i in range(0, len(coords_df)):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": coords[i]
            },
            "properties": {
                           'shapeID': str(names[i])
                          }
        })

    print(features)

    return jsonify(features)



@APP.route('/predict_migration', methods=['GET', 'POST'])
def predict_migration():

    with open('status.json', 'w') as outfile:
        json.dump({'status': "Status - Starting predictions."}, outfile)

    # Parse the selected municipalities and get their unique B ID's
    selected_municipalities = request.json['selected_municipalities']

    print("LEN SELECTED MUNIS: ", len(selected_municipalities))

    # TEMPORARY UNTIL YOU GET THE BIG IMAGES DOWNLOADED
    selected_municipalities = [sm for sm in selected_municipalities if sm in munis_available]
    selected_municipalities = [sm for sm in selected_municipalities if graph_id_dict[sm] not in BAD_IDS]

    # Read in the migration data and subset it to the selected municipalities
    dta = pd.read_csv(DATA_PATH)
    dta = dta.dropna(subset = ['GEO2_MX'])

    dta_ids = dta["GEO2_MX"].to_list()
    selected_municipalities = [sm for sm in selected_municipalities if int(sm) in dta_ids]

    # If no muni's are selected, select them all
    if len(selected_municipalities) == 0:
        selected_municipalities = dta['sending'].to_list()
        print("Selected municipalities since none were selected: ", selected_municipalities)

    dta_selected, dta_dropped, muns_to_pred = prep_dataframes(dta, request, selected_municipalities)

    #######################################################################
    # Create some sort of dictionary with references to the graph_id_dict # 
    #######################################################################
    selected_muni_ref_dict = {}
    for muni in selected_municipalities:
        muni_ref = graph_id_dict[muni]
        selected_muni_ref_dict[muni] = muni_ref

    #######################################################################
    # Create a dictionary with graph_id_dict                              #
    # references mapped to the new census data                            #
    #######################################################################
    new_census_vals = {}
    for sm in range(0, len(selected_municipalities)):
        new_census_vals[selected_muni_ref_dict[selected_municipalities[sm]]] = muns_to_pred[sm]

    #######################################################################
    # Predict the new data                                                # 
    #######################################################################
    predictions = predict(graph, selected_muni_ref_dict, new_census_vals, selected_municipalities)

    #######################################################################
    # Update the new predictions in the dta_selected dataframe and append #
    # that to all of the data in dta_dropped that wan't selected to       #
    # create a full dataframe with everything                             #
    #######################################################################
    dta_selected['sum_num_intmig'] = predictions
    dta_final = dta_selected.append(dta_dropped)
    print("ALL DATA SHAPE: ", dta_final.shape)
    print("DTA FINAL HEAD: ", dta_final.head())

    #######################################################################
    # Normalize the geoJSON as a pandas dataframe and merge in the new    #
    # census & migration data                                             #
    #######################################################################
    dta_final['GEO2_MX'] = dta_final['GEO2_MX'].astype(str)
    geoDF = json_normalize(geodata_collection["features"])
    merged = pd.merge(geoDF, dta_final, left_on = "properties.shapeID", right_on = "GEO2_MX")
    merged['sum_num_intmig'] = merged['sum_num_intmig'].fillna(0)

    #######################################################################
    # Aggregate statistics and send to a JSON                             #
    #######################################################################

    total_pred_migrants = merged['sum_num_intmig'].sum()
    merged['avg_age_weight'] = merged['avg_age'] * merged['sum_num_intmig']
    avg_age = merged['avg_age_weight'].sum() / merged['sum_num_intmig'].sum()
    migration_statistics = {'avg_age': avg_age, "total_pred_migrants": float(total_pred_migrants)}
    with open('predicted_migrants.json', 'w') as outfile:
        json.dump(migration_statistics, outfile)

    #######################################################################
    # Convert features to a gejson for rendering in Leaflet               #
    #######################################################################
    features = convert_features_to_geojson(merged)

    with open('status.json', 'w') as outfile:
        json.dump({'status': "Status - Rendering new migration map..."}, outfile)

    return jsonify(features)




@APP.route('/update_stats', methods=['GET'])
def update_stats():

    # Read in migration data
    df = pd.read_csv(DATA_PATH)

    with open("./predicted_migrants.json") as json_file:
        predictions = json.load(json_file)

    # Get the number of migrants (over a 5 year period) to send to HTML for stat box
    total_og_migrants = df['sum_num_intmig'].sum()
    total_pred_migrants = int(predictions['total_pred_migrants'])
    change = (total_pred_migrants - total_og_migrants) / 5
    p_change = ( change / (total_og_migrants / 5) ) * 100
    
    # Calculate average age stuff
    df['avg_age_weight'] = df['avg_age'] * df['sum_num_intmig']
    og_avg_age = df['avg_age_weight'].sum() / df['sum_num_intmig'].sum()

    avg_age = predictions['avg_age']
    avg_age_change = avg_age - og_avg_age
    p_avg_age_change = ((round(avg_age, 2) - og_avg_age) / og_avg_age) * 100


    with open("./correlations.json", "r") as f:
        corrs = json.load(f)

    with open("./vars.json", "r") as f:
        var_cats = json.load(f)    

    corr_means = []
    corr_category_dict = {}
    for category in var_cats.keys():
        cat_columns = var_cats[category]
        cat_vals = [v for k,v in corrs.items() if k in cat_columns]
        if len(cat_vals) == 0:
            cat_mean_corr = 0
        else:
            cat_mean_corr = np.mean(cat_vals)
            corr_category_dict[category] = [cat_columns, cat_vals]
        corr_means.append(cat_mean_corr)
        print(category, cat_columns, cat_mean_corr)

    return {'change': int(change),
            'p_change': round(p_change, 2),
            'predicted_migrants': round(total_pred_migrants / 5, 0),
            'avg_age': round(avg_age, 0),
            'avg_age_change': round(avg_age_change, 0),
            'pavg_age_change': round(p_avg_age_change, 0),
            'corr_means': corr_means,
            'corr_category_dict': corr_category_dict}




@APP.route('/status_update', methods=['GET'])
def status_update():
    with open("./status.json", "r") as f:
        status = json.load(f)
    return {"status": status['status']}




@APP.route('/download_data', methods=['GET'])
def download_data():
    return send_from_directory("./data/",
                               "portal_data.csv", as_attachment = True)




if __name__ == '__main__':
    APP.debug=True
    APP.run()


