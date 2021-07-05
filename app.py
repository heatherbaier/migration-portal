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
    print("Average age: ", df['avg_age'].mean())
    # print("Average age: ", df['avg_age_weight'].sum() / df['sum_num_intmig'].sum())
    avg_age = df['avg_age'].mean()

    # Open the variables JSON and the JSON containing the readable translation of the variables
    with open("./vars.json", "r") as f:
        grouped_vars = json.load(f)

    with open("./var_map.json", "r") as f2:
        var_names = json.load(f2)

    # Get all of the variables to send to Flask
    econ, demog, family, health, edu, employ, hhold = get_column_lists(df, var_names, grouped_vars)

    print("about to render homepage")

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
                                  total_migrants = total_migrants,
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
            "properties": {'num_migrants': num_migrants[i],
                           'shapeID': str(shapeIDs[i]),
                           'shapeName': ''
                          }
        })

    return jsonify(features)



@APP.route('/border-features', methods=['GET'])
def get_border_features():

    feature_df = json_normalize(border_stations["features"])

    print(feature_df)
    
    # Make lists of all of the features we want available to the Leaflet map
    coords = feature_df['geometry.coordinates']
    types = feature_df['geometry.type']
    # num_migrants = feature_df['sum_num_intmig']
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
    
    print("Selected municipalities: ", selected_municipalities)

    # Read in the migration data and subset it to the selected municipalities
    dta = pd.read_csv(DATA_PATH)
    dta = dta.dropna(subset = ['GEO2_MX'])

    # If no muni's are selected, select them all
    if len(selected_municipalities) == 0:
        selected_municipalities = dta['sending'].to_list()
        print("Selected municipalities since none were selected: ", selected_municipalities)


    dta_selected = dta[dta['GEO2_MX'].isin([int(i) for i in selected_municipalities])]
    print(dta_selected.shape)

    print("NUM MIGRANTS HERE: ", dta_selected['sum_num_intmig'].sum())
    num_og_migrants = dta_selected['sum_num_intmig'].sum()

    # Parse the edited input variables and switch all of the 0's in percent_changes to 1 (neccessary for multiplying later on)
    column_names = request.json['column_names']
    percent_changes = request.json['percent_changes']
    # percent_changes = request.json['percent_changes']
    percent_changes = [float(i) - 100 if i != '100' else '1' for i in percent_changes]

    # Open the var_map JSON and reverse the dictionary
    with open("./var_map.json", "r") as f2:
        var_names = json.load(f2)
    reverse_var_names = dict([(value, key) for key, value in var_names.items()])

    # Change the 'pretty' variable names back to their originals so we can edit the dataframe
    column_names = [reverse_var_names[i] if i in reverse_var_names.keys() else i for i in column_names]

    # Multiply the columns by their respective percent changes
    for i in range(0, len(column_names)):

        if float(percent_changes[i]) < 0:
            percentage = abs(float(percent_changes[i])) * .01
            to_subtract = percentage * dta_selected[column_names[i]]
            dta_selected[column_names[i]] = dta_selected[column_names[i]] - to_subtract
        else:
            percentage = abs(float(percent_changes[i])) * .01
            to_add = percentage * dta_selected[column_names[i]]
            dta_selected[column_names[i]] = dta_selected[column_names[i]] + to_add

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
    # dta_appended = dta_appended.drop(["sum_num_intmig"], axis = 1)

    print("Final census data frame shape: ", dta_appended.shape)

    # Scale the data frame for the model
    X = dta_appended.loc[:, dta_appended.columns != "sum_num_intmig"].values
    mMScale = preprocessing.MinMaxScaler()
    X = mMScale.fit_transform(X)

    # Grab just the municaplities that we edited
    muns_to_pred = X[-len(selected_municipalities):]    

    selected_muni_ref_dict = {}
    for muni in selected_municipalities:
        muni_ref = graph_id_dict[muni]
        selected_muni_ref_dict[muni] = muni_ref


    new_census_vals = {}

    for sm in range(0, len(selected_municipalities)):
        new_census_vals[selected_muni_ref_dict[selected_municipalities[sm]]] = muns_to_pred[sm]

    print(new_census_vals)

    x, adj_lists, y = [], {}, []

    a = 0
    for muni_id, dta in graph.items():
        if muni_id in selected_muni_ref_dict.values():
            cur_x = dta["x"]
            cur_x = cur_x[0:len(cur_x) - 202]
            [cur_x.append(v) for v in new_census_vals[muni_id]]
            print("yes!", len(cur_x))
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
    model.eval()


    predictions = []
    for muni in selected_municipalities:
        print("Current municipality: ", muni)
        muni_ref = graph_id_dict[muni]
        print("Ref ID:  ", muni_ref)
        input = [muni_ref]
        prediction = model.forward(input).item()
        predictions.append(prediction)
        print("Prediction: ", prediction)


    # Update the migration numbers in the dataframe and re-append it tot the wider dataframe
    dta_selected['sum_num_intmig'] = predictions

    print("NUM MIGRANTS AFTER PRED: ", dta_selected['sum_num_intmig'].sum())
    num_pred_migrants = dta_selected['sum_num_intmig'].sum()

    dta_final = dta_dropped.append(dta_selected)
    dta_final['GEO2_MX'] = dta_final['GEO2_MX'].astype(str)

    # Normalize the geoJSON as a pandas dataframe
    geoDF = json_normalize(geodata_collection["features"])

    merged = pd.merge(geoDF, dta_final, left_on = "properties.shapeID", right_on = "GEO2_MX")

    print("NUMBER OF PERSONS TO US: ", merged['sum_num_intmig'].sum())
    total_migrants = merged['sum_num_intmig'].sum()

    merged['avg_age_weight'] = merged['avg_age'] * merged['sum_num_intmig']
    print("Average age: ", merged['avg_age'].mean())
    print("Average age: ", merged['avg_age_weight'].sum() / merged['sum_num_intmig'].sum())
    avg_age = merged['avg_age_weight'].sum()# / merged['sum_num_intmig'].sum()

    total_migrants = {'avg_age': avg_age, "num_og_migrants": num_og_migrants, "num_pred_migrants": num_pred_migrants}
    
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

    num_pred_migrants = predictions['num_pred_migrants']
    num_og_migrants = predictions['num_og_migrants']

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