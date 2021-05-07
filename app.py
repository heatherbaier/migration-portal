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
from app_helpers import *


# Create the application.
APP = flask.Flask(__name__)

with open('status.json', 'w') as outfile:
    json.dump({'status': "Startup"}, outfile)


@APP.route('/', methods=['GET','POST'])
def index():

    # Read in migration data
    df = pd.read_csv(DATA_PATH)

    # Get the number of migrants to send to HTML for stat box
    total_migrants = df['sum_num_intmig'].sum()

    municipality_ids = df['sending'].unique()
    # df_var_cols = [i for i in df.columns if i not in ['sending','number_moved']]

    df['avg_age_weight'] = df['avg_age'] * df['sum_num_intmig']
    print("Average age: ", df['avg_age'].mean())
    print("Average age: ", df['avg_age_weight'].sum() / df['sum_num_intmig'].sum())
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
                                  total_migrants = total_migrants,
                                  avg_age = round(avg_age, 2))




@APP.route('/geojson-features', methods=['GET'])
def get_all_points():

    # Convert the geoJSON to a dataframe and merge it to the migration data
    feature_df = convert_to_pandas(geodata_collection, MATCH_PATH, DATA_PATH)

    print(sum(feature_df['geometry.coordinates'].isna()))
    print(sum(feature_df['geometry.type'].isna()))
    print(sum(feature_df['sum_num_intmig'].isna()))
    print(sum(feature_df['properties.shapeID'].isna()))
    print(sum(feature_df['properties.shapeName'].isna()))


    feature_df['sum_num_intmig'] = feature_df['sum_num_intmig'].fillna(0)
    

    # Make lists of all of the features we want available to the Leaflet map
    coords = feature_df['geometry.coordinates']
    types = feature_df['geometry.type']
    num_migrants = feature_df['sum_num_intmig']
    shapeIDs = feature_df['properties.shapeID']
    shapeNames = feature_df['properties.shapeName']

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
                           'shapeID': shapeIDs[i],
                           'shapeName': shapeNames[i]
                          }
        })

    return jsonify(features)



@APP.route('/predict_migration', methods=['GET', 'POST'])
def predict_migration():

    with open('status.json', 'w') as outfile:
        json.dump({'status': "Status - Starting predictions."}, outfile)

    # Parse the selected municipalities and get their unique B ID's
    selected_municipalities = request.json['selected_municipalities']
    # if (len(selected_municipalities) != 0) & (selected_municipalities[0].startswith("MEX")):
    selected_municipalities = [i.split("-")[3] if i.startswith("MEX") else i for i in selected_municipalities]
    print("Selected municipalities: ", selected_municipalities)

    # Read in the migration data and subset it to the selected municipalities
    dta = switch_column_names(MATCH_PATH, DATA_PATH)

    if len(selected_municipalities) == 0:
        selected_municipalities = dta['sending'].to_list()
        print("Selected municipalities since none were selected: ", selected_municipalities)

    dta_selected = dta[dta['sending'].isin(selected_municipalities)]
    dta_selected = dta_selected.dropna(subset = ['sending'])
    print(dta_selected.shape)

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
    dta_dropped = dta[~dta['sending'].isin(selected_municipalities)]

    # Then re-append the updated data to the larger dataframe incorporating user input
    dta_appended = dta_dropped.append(dta_selected)
    dta_appended = dta_appended.drop(['sending'], axis = 1)

    # dta_appended = dta_appended.drop(['Unnamed: 0', 'sending'], axis = 1)
    dta_appended = dta_appended.fillna(0)
    dta_appended = dta_appended.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    with open("./us_vars.txt", "r") as f:
        vars = f.read().splitlines()
    vars = [i for i in vars if i in dta_appended.columns]
    dta_appended = dta_appended[vars]

    print("SHAPE HERE: ", dta_appended.shape)

    # Scale the data frame for the model
    X = dta_appended.loc[:, dta_appended.columns != "sum_num_intmig"].values
    mMScale = preprocessing.MinMaxScaler()
    X = mMScale.fit_transform(X)

    # Grab just the municaplities that we edited
    muns_to_pred = X[-len(selected_municipalities):]

    muni_names = get_muni_names(selected_municipalities)

    # Predict each of them
    predictions = [predict_row(muns_to_pred[i], X, muni_names[i]) for i in range(0, len(muns_to_pred))]
    
    # Update the migration numbers in the dataframe and re-append it tot the wider dataframe
    dta_selected['sum_num_intmig'] = predictions
    dta_final = dta_dropped.append(dta_selected)

    # Normalize the geoJSON as a pandas dataframe
    geoDF = json_normalize(geodata_collection["features"])

    # Get the B unique ID column (akgkjklajkljlk)
    geoDF["B"] = geoDF['properties.shapeID'].str.split("-").str[3]
    geoDF = geoDF.rename(columns = {"B":"sending"})

    # Mix it all together
    merged = pd.merge(geoDF, dta_final, on = 'sending')

    print("NUMBER OF PERSONS TO US: ", merged['sum_num_intmig'].sum())
    total_migrants = merged['sum_num_intmig'].sum()

    merged['avg_age_weight'] = merged['avg_age'] * merged['sum_num_intmig']
    print("Average age: ", merged['avg_age'].mean())
    print("Average age: ", merged['avg_age_weight'].sum() / merged['sum_num_intmig'].sum())
    avg_age = merged['avg_age_weight'].sum() / merged['sum_num_intmig'].sum()

    total_migrants = {'total_migrants': total_migrants, 'avg_age': avg_age}
    
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
    predicted_migrants = predictions['total_migrants']
    avg_age = predictions['avg_age']

    p_change = ((round(predicted_migrants, 0) - total_migrants) / total_migrants) * 100
    change = round(predicted_migrants, 0) - total_migrants
    avg_age_change = avg_age - og_avg_age
    p_avg_age_change = ((round(avg_age, 0) - og_avg_age) / og_avg_age) * 100


    return {'change': change,
            'p_change': round(p_change, 2),
            'predicted_migrants': round(predicted_migrants, 0),
            'avg_age': round(avg_age, 2),
            'avg_age_change': round(avg_age_change, 2),
            'pavg_age_change': round(p_avg_age_change, 2)}




@APP.route('/status_update', methods=['GET'])
def status_update():
    with open("./status.json", "r") as f:
        status = json.load(f)
    return {"status": status['status']}




if __name__ == '__main__':
    APP.debug=True
    APP.run()