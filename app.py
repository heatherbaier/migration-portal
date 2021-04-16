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




@APP.route('/', methods=['GET','POST'])
def index():

    # Read in migration data
    df = pd.read_csv(DATA_PATH)

    # Get the number of migrants to send to HTML for stat box
    total_migrants = df['num_persons_to_us'].sum()

    # Get the data for the number of migrants histogram
    hist = {}
    hist_data, hist_labels = np.histogram(df['num_persons_to_us'], bins = 20)
    hist['data'] = list(hist_data)
    hist["labels"] = [round(i, 2) for i in list(hist_labels)]

    municipality_ids = df['sending'].unique()
    df_var_cols = [i for i in df.columns if i not in ['sending','number_moved']]

    df['avg_age_weight'] = df['avg_age'] * df['num_persons_to_us']
    print("Average age: ", df['avg_age'].mean())
    print("Average age: ", df['avg_age_weight'].sum() / df['num_persons_to_us'].sum())
    avg_age = df['avg_age_weight'].sum() / df['num_persons_to_us'].sum()

    # Open the variables JSON and the JSON containing the pretty translation of the variables
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
                                  avg_age = round(avg_age, 2),
                                  hist = hist)




@APP.route('/geojson-features', methods=['GET'])
def get_all_points():

    # Convert the geoJSON to a dataframe and merge it to the migration data
    feature_df = convert_to_pandas(geodata_collection, MATCH_PATH, DATA_PATH)

    # Make lists of all of the features we want available to the Leaflet map
    coords = feature_df['geometry.coordinates']
    types = feature_df['geometry.type']
    num_migrants = feature_df['num_persons_to_us']
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
    print("we made it!")

    # Parse the selected municipalities and get their unique B ID's
    selected_municipalities = request.json['selected_municipalities']
    selected_municipalities = [i.split("-")[3] for i in selected_municipalities]
    print("Selected municipalities: ", selected_municipalities)

    # Read in the migration data and subset it tot he selected municipalities
    dta = switch_column_names(MATCH_PATH, DATA_PATH)
    dta_selected = dta[dta['sending'].isin(selected_municipalities)]
    print(dta_selected.shape)

    # Parse the edited input variables and switch all of the 0's in percent_changes to 1 (neccessary for multiplying later on)
    column_names = request.json['column_names']
    percent_changes = request.json['percent_changes']
    percent_changes = [i if i != '0' else '1' for i in percent_changes]


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


    # TEMP DUMMY VARAIBLES (THIS NEEDS TO BE FIXED WHEN YOU INPUT THE CORRECT TRAINED MODEL)
    dta_dropped['DUMMY'] = [i for i in range(0, len(dta_dropped))]
    dta_selected['DUMMY'] = [i for i in range(0, len(dta_selected))]

    # Then re-append the updated data to the larger dataframe incorporating user input
    dta_appended = dta_dropped.append(dta_selected)
    dta_appended = dta_appended.drop(['sending'], axis = 1)

    # Scale the data frame for the model
    X = dta_appended.loc[:, dta_appended.columns != "num_persons_to_us"].values
    mMScale = preprocessing.MinMaxScaler()
    X = mMScale.fit_transform(X)

    # Grab just the municaplities that we edited
    muns_to_pred = X[-len(selected_municipalities):]

    # Predict each of them
    predictions = [predict_row(i, X) for i in muns_to_pred]
    
    # Update the migration numbers in the dataframe and re-append it tot the wider dataframe
    dta_selected['num_persons_to_us'] = predictions
    dta_final = dta_dropped.append(dta_selected)

    # Normalize the geoJSON as a pandas dataframe
    geoDF = json_normalize(geodata_collection["features"])

    # Get the B unique ID column (akgkjklajkljlk)
    geoDF["B"] = geoDF['properties.shapeID'].str.split("-").str[3]
    geoDF = geoDF.rename(columns = {"B":"sending"})

    # Mix it all together
    merged = pd.merge(geoDF, dta_final, on = 'sending')

    print("NUMBER OF PERSONS TO US: ", merged['num_persons_to_us'].sum())
    total_migrants = merged['num_persons_to_us'].sum()



    merged['avg_age_weight'] = merged['avg_age'] * merged['num_persons_to_us']
    print("Average age: ", merged['avg_age'].mean())
    print("Average age: ", merged['avg_age_weight'].sum() / merged['num_persons_to_us'].sum())
    avg_age = merged['avg_age_weight'].sum() / merged['num_persons_to_us'].sum()


    total_migrants = {'total_migrants': total_migrants, 'avg_age': avg_age}
    

    with open('predicted_migrants.json', 'w') as outfile:
        json.dump(total_migrants, outfile)


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

    return jsonify(features)
        




@APP.route('/update_stats', methods=['GET'])
def update_stats():

    # Read in migration data
    df = pd.read_csv(DATA_PATH)

    # Get the number of migrants to send to HTML for stat box
    total_migrants = df['num_persons_to_us'].sum()
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


if __name__ == '__main__':
    APP.debug=True
    APP.run()