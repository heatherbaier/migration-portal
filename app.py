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
                                  total_migrants = int(total_migrants),
                                  avg_age = round(avg_age, 2))




@APP.route('/geojson-features', methods=['GET'])
def get_all_points():

    # Convert the geoJSON to a dataframe and merge it to the migration data
    feature_df = convert_to_pandas(geodata_collection, MATCH_PATH, DATA_PATH)
    feature_df['sum_num_intmig'] = feature_df['sum_num_intmig'].fillna(0)
    
    # Make lists of all of the features we want available to the Leaflet map
    coords = feature_df['geometry.coordinates']
    types = feature_df['geometry.type']
    num_migrants = feature_df['sum_num_intmig']
    shapeIDs = feature_df['shapeID']
    shapeNames = feature_df['properties.geo2_mx1960_2015_ADMIN_NAME']

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



@APP.route('/predict_migration', methods=['GET', 'POST'])
def predict_migration():

    with open('status.json', 'w') as outfile:
        json.dump({'status': "Status - Starting predictions."}, outfile)

    # Parse the selected municipalities and get their unique B ID's
    selected_municipalities = request.json['selected_municipalities']

    # TEMPORARY UNTIL YOU GET THE BIG IMAGES DOWNLOADED
    selected_municipalities = [sm for sm in selected_municipalities if sm in munis_available]
    selected_municipalities = [sm for sm in selected_municipalities if graph_id_dict[sm] not in BAD_IDS]

    print("Selected municipalities: ", selected_municipalities)

    # Read in the migration data and subset it to the selected municipalities
    dta = pd.read_csv(DATA_PATH)
    dta = dta.dropna(subset = ['GEO2_MX'])

    # If no muni's are selected, select them all
    if len(selected_municipalities) == 0:
        selected_municipalities = dta['sending'].to_list()
        print("Selected municipalities since none were selected: ", selected_municipalities)

    dta_appended, dta_selected, dta_dropped, num_og_migrants, X = prep_dataframes(dta, request, selected_municipalities)

    # Grab just the municaplities that we edited
    muns_to_pred = X[-len(selected_municipalities):]    

    selected_muni_ref_dict = {}
    for muni in selected_municipalities:
        muni_ref = graph_id_dict[muni]
        selected_muni_ref_dict[muni] = muni_ref

    new_census_vals = {}
    for sm in range(0, len(selected_municipalities)):
        new_census_vals[selected_muni_ref_dict[selected_municipalities[sm]]] = muns_to_pred[sm]

    predictions = predict(graph, selected_muni_ref_dict, new_census_vals, selected_municipalities)

    # Update the migration numbers in the dataframe and re-append it tot the wider dataframe
    dta_selected['sum_num_intmig'] = predictions
    num_pred_migrants = dta_selected['sum_num_intmig'].sum()

    dta_final = dta_dropped.append(dta_selected)
    dta_final['GEO2_MX'] = dta_final['GEO2_MX'].astype(str)

    # Normalize the geoJSON as a pandas dataframe
    geoDF = json_normalize(geodata_collection["features"])
    merged = pd.merge(geoDF, dta_final, left_on = "properties.shapeID", right_on = "GEO2_MX")

    # Aggregate stats and send to a JSON
    total_migrants = merged['sum_num_intmig'].sum()
    merged['avg_age_weight'] = merged['avg_age'] * merged['sum_num_intmig']
    avg_age = merged['avg_age_weight'].sum() / merged['sum_num_intmig'].sum()
    total_migrants = {'avg_age': avg_age, "num_og_migrants": num_og_migrants, "num_pred_migrants": float(num_pred_migrants)}
    with open('predicted_migrants.json', 'w') as outfile:
        json.dump(total_migrants, outfile)

    merged['sum_num_intmig'] = merged['sum_num_intmig'].fillna(0)
    features = convert_features_to_geojson(merged)

    with open('status.json', 'w') as outfile:
        json.dump({'status': "Status - Rendering new migration map..."}, outfile)

    return jsonify(features)




@APP.route('/update_stats', methods=['GET'])
def update_stats():

    # Read in migration data
    df = pd.read_csv(DATA_PATH)

    # Get the number of migrants to send to HTML for stat box
    total_migrants = df['sum_num_intmig'].sum()
    og_avg_age = df['avg_age'].mean()

    with open("./predicted_migrants.json") as json_file:
        predictions = json.load(json_file)

    num_pred_migrants = int(predictions['num_pred_migrants'])
    num_og_migrants = int(predictions['num_og_migrants'])

    # predicted_migrants = predictions['total_migrants']
    predicted_migrants = (total_migrants - num_og_migrants) + num_pred_migrants
    avg_age = predictions['avg_age']
    avg_age = avg_age / ((total_migrants - num_og_migrants) + num_pred_migrants)

    p_change = ((round(predicted_migrants, 0) - total_migrants) / total_migrants) * 100
    change = round(predicted_migrants, 0) - total_migrants
    avg_age_change = avg_age - og_avg_age
    p_avg_age_change = ((round(avg_age, 0) - og_avg_age) / og_avg_age) * 100

    return {'change': change,
            'p_change': round(p_change, 2),
            'predicted_migrants': round(predicted_migrants, 0),
            'avg_age': round(avg_age, 0),
            'avg_age_change': round(avg_age_change, 0),
            'pavg_age_change': round(p_avg_age_change, 0)}




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