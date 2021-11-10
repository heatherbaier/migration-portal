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
import scipy
import torch
import flask
import json
import io
import os

from app_helpers import *
from model.utils import *
from model.aggregator import *
from model.encoder import *
from model.graphsage import *

# Create the application.
APP = flask.Flask(__name__)

# Setup status file
with open('status.json', 'w') as outfile:
    json.dump({'status': "Startup"}, outfile)


@APP.route('/', methods=['GET','POST'])
def index():

    """
    Landing page
    """

    # Read in census and migration data
    df = pd.read_csv(DATA_PATH)

    with open(MIGRATION_PATH) as m:
        mig_data = json.load(m)

    # Get total # of migrants and a list of muni ID's
    total_migrants = sum(list(mig_data.values()))
    municipality_ids = list(mig_data.keys())

    # Calculate the average age of migrants per muni
    df['avg_age_weight'] = df['avg_age'] * df['sum_num_intmig']
    avg_age = df['avg_age_weight'].sum() / df['sum_num_intmig'].sum()

    # Open the variables JSON and the JSON containing the readable translation of the variables
    with open("./vars.json", "r") as f:
        grouped_vars = json.load(f)

    with open("./var_map.json", "r") as f2:
        var_names = json.load(f2)

    # Get all of the variables to send to Flask for dropdown options
    econ, demog, family, health, edu, employ, hhold = get_column_lists(df, var_names, grouped_vars)

    # Merry Christmas HTML!!
    return flask.render_template('index1.html', 
                                  municipality_ids = municipality_ids, 
                                  econ_data = econ,
                                  demog_data = demog,
                                  family_data = family,
                                  health_data = health,
                                  edu_data = edu,
                                  employ_data = employ,
                                  hhold_data = hhold,
                                  total_migrants = f'{int(total_migrants / 5):,}',
                                  avg_age = round(avg_age, 2),
                                  model_error = f'{int((total_migrants / 5) * MODEL_ERROR):,}')


@APP.route('/scenario', methods=['GET','POST'])
def scenario():

    """
    Landing page
    """

    # Read in census and migration data
    df = pd.read_csv(DATA_PATH)

    with open(MIGRATION_PATH) as m:
        mig_data = json.load(m)

    # Get total # of migrants and a list of muni ID's
    total_migrants = sum(list(mig_data.values()))
    municipality_ids = list(mig_data.keys())

    # Calculate the average age of migrants per muni
    df['avg_age_weight'] = df['avg_age'] * df['sum_num_intmig']
    avg_age = df['avg_age_weight'].sum() / df['sum_num_intmig'].sum()

    # Open the variables JSON and the JSON containing the readable translation of the variables
    with open("./vars.json", "r") as f:
        grouped_vars = json.load(f)

    with open("./var_map.json", "r") as f2:
        var_names = json.load(f2)

    # Get all of the variables to send to Flask for dropdown options
    econ, demog, family, health, edu, employ, hhold = get_column_lists(df, var_names, grouped_vars)

    # Merry Christmas HTML!!
    return flask.render_template('scenario.html', 
                                  municipality_ids = municipality_ids, 
                                  econ_data = econ,
                                  demog_data = demog,
                                  family_data = family,
                                  health_data = health,
                                  edu_data = edu,
                                  employ_data = employ,
                                  hhold_data = hhold,
                                  total_migrants = f'{int(total_migrants / 5):,}',
                                  avg_age = round(avg_age, 2),
                                  model_error = f'{int((total_migrants / 5) * MODEL_ERROR):,}')





@APP.route('/cat_select', methods=['GET','POST'])
def cat_select():

    """
    Landing page
    """

    category = request.json["selected_cat"]

    # Read in census and migration data
    df = pd.read_csv(DATA_PATH)

    # Calculate the average age of migrants per muni
    df['avg_age_weight'] = df['avg_age'] * df['sum_num_intmig']

    # Open the variables JSON and the JSON containing the readable translation of the variables
    with open("./vars.json", "r") as f:
        grouped_vars = json.load(f)

    with open("./var_map.json", "r") as f2:
        var_names = json.load(f2)

    econ = get_column_lists(df, var_names, grouped_vars, category)

    econ = list(econ)

    print("category:", category, econ)

    return {'categories': econ}




@APP.route('/drilldown', methods=['GET', 'POST'])
def drilldown():

    drilldown_muni = request.json['drilldown_muni']

    print("drilldown_muni: ", drilldown_muni)

    df = pd.read_csv(DATA_PATH)

    num_migs = df[df['GEO2_MX'] == int(drilldown_muni)]['sum_num_intmig'].values[0]

    all_migs = df['sum_num_intmig'].values

    mig_perc = scipy.stats.percentileofscore(all_migs, num_migs) 

    mig_hist_counts, mig_hist_bins = np.histogram(df['sum_num_intmig'])

    print("NUM MIGS: ", num_migs, mig_perc)

    print({"mig_perc": round(mig_perc, 0),
            "mig_hist_counts": list(mig_hist_counts),
            "mig_hist_bins": list(mig_hist_bins)
            })

    return {"mig_perc": round(mig_perc, 0),
            "mig_hist_counts": [str(i) for i in list(mig_hist_counts)],
            "mig_hist_bins": [str(i) for i in list(mig_hist_bins)]
            }



@APP.route('/geojson-features', methods=['GET'])
def get_all_points():

    """
    Grabs the polygons from the geojson, converts them to JSON format with geometry and data 
    features and sends back to the webpage to render on the Leaflet map
    """

    # Convert the geoJSON to a dataframe and merge it to the migration data
    feature_df = convert_to_pandas(geodata_collection, MATCH_PATH, DATA_PATH)
    feature_df['sum_num_intmig'] = feature_df['sum_num_intmig'].fillna(0)
    feature_df['perc_migrants'] = feature_df['sum_num_intmig'] / feature_df['total_pop']

    print(feature_df.columns)

    
    # Make lists of all of the features we want available to the Leaflet map
    coords = feature_df['geometry.coordinates']
    types = feature_df['geometry.type']
    num_migrants = feature_df['perc_migrants']
    shapeIDs = feature_df['shapeID']
    shapeNames = feature_df["properties.ipumns_simple_wgs_wdata_geo2_mx1960_2015_ADMIN_NAME"]

    # For each of the polygons in the data frame, append it and it's data to a list 
    # of dicts to be sent as a JSON back to the Leaflet map
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




@APP.route('/border-sectors', methods=['GET'])
def get_border_sectors():

    """
    Grabs the centroids of the 9 border sectors and their percentage of migrants data
    """

    # Read in centroids and fractional data
    coords_df = pd.read_csv("./data/sector_centroids.csv")
    with open("./data/sector_fractions.json", "r") as f:
        fractions = json.load(f)
    dta = pd.read_csv(DATA_PATH)

    # Calcualte total migrants and max # of migrants (for normalization)
    total_mig = dta["sum_num_intmig"].sum()
    max_total_mig = dta["sum_num_intmig"].max()

    # Convert relavant data into lists
    x, y = coords_df['xcoord'], coords_df['ycoord']
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
                           'shapeID': str(names[i]),
                           'num_migrants': round(fractions[str(names[i])] * 100, 2),
                           'num_migrants_normed': (total_mig * fractions[str(names[i])]) / max_total_mig
                          }
        })

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
    # selected_municipalities = [sm for sm in selected_municipalities if graph_id_dict[sm] not in BAD_IDS]

    # Read in the migration data and subset it to the selected municipalities
    dta = pd.read_csv(DATA_PATH)
    dta = dta.dropna(subset = ['GEO2_MX'])

    dta_ids = dta["GEO2_MX"].to_list()
    selected_municipalities = [sm for sm in selected_municipalities if int(sm) in dta_ids]

    # If no muni's are selected, select them all
    if len(selected_municipalities) == 0:
        selected_municipalities = [str(i) for i in dta['GEO2_MX'].to_list()]
        selected_municipalities = [sm for sm in selected_municipalities if sm in munis_available]
        # selected_municipalities = [sm for sm in selected_municipalities if graph_id_dict[sm] not in BAD_IDS]
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
    dta_final[['GEO2_MX', 'sum_num_intmig']].to_csv("./map_layers/sum_num_intmig.csv", index = False)
    geoDF = json_normalize(geodata_collection["features"])
    merged = pd.merge(geoDF, dta_final, left_on = "properties.shapeID", right_on = "GEO2_MX")
    merged['sum_num_intmig'] = merged['sum_num_intmig'].fillna(0)
    merged['perc_migrants'] = merged['sum_num_intmig'] / merged['total_pop']

    dta_final['perc_migrants'] = dta_final['sum_num_intmig'] / dta_final['total_pop']
    dta_final[['GEO2_MX', 'perc_migrants']].to_csv("./map_layers/perc_migrants.csv", index = False)

    og_df = pd.read_csv(DATA_PATH)
    og_df = og_df[['GEO2_MX', 'sum_num_intmig', 'total_pop']].rename(columns = {'sum_num_intmig': 'sum_num_intmig_og'})
    og_df['GEO2_MX'] = og_df['GEO2_MX'].astype(str)
    change_df = pd.merge(og_df, dta_final[['GEO2_MX', 'sum_num_intmig']])
    change_df['absolute_change'] = change_df['sum_num_intmig'] - change_df['sum_num_intmig_og']
    change_df[['GEO2_MX', 'absolute_change']].to_csv("./map_layers/absolute_change.csv", index = False)
    change_df['perc_change'] = (change_df['sum_num_intmig'] - change_df['sum_num_intmig_og']) / change_df['sum_num_intmig_og']
    change_df = change_df.replace([np.inf, -np.inf], np.nan)
    change_df = change_df.fillna(0)
    change_df[['GEO2_MX', 'perc_change']].to_csv("./map_layers/perc_change.csv", index = False)

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
    features = convert_features_to_geojson(merged, column = 'perc_migrants')

    with open('status.json', 'w') as outfile:
        json.dump({'status': "Status - Rendering new migration map..."}, outfile)

    return jsonify(features)



@APP.route('/update_map', methods=['GET', 'POST'])
def update_map():

    """
    Called when a user changes whic type of data to display ont he map (i.e. % v absolute & change v total)
    """

    print("Variable to map: ", request.json['variable'])

    data_path = os.path.join("map_layers", request.json['variable'] + ".csv")
    dta_final = pd.read_csv(data_path)
    dta_final['GEO2_MX'] = dta_final['GEO2_MX'].astype(str)

    geoDF = json_normalize(geodata_collection["features"])
    merged = pd.merge(geoDF, dta_final, left_on = "properties.shapeID", right_on = "GEO2_MX")

    features = convert_features_to_geojson(merged, column = request.json['variable'])
    
    return jsonify(features)



@APP.route('/update_stats', methods=['GET'])
def update_stats():

    """
    Function used to update the statistc boxes at the top of the page & the graphs below the map
    """

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
        cat_vals = [round(abs(v), 4) for k,v in corrs.items() if k in cat_columns]
        if len(cat_vals) == 0:
            cat_mean_corr = 0
        else:
            cat_mean_corr = round(np.mean(cat_vals), 4)
            corr_category_dict[category] = [cat_columns, [round(v, 4) for k,v in corrs.items() if k in cat_columns]]
        corr_means.append(cat_mean_corr)
        print(category, cat_columns, cat_mean_corr)

    migs_for_bs = pd.read_csv("./map_layers/sum_num_intmig.csv")
    migs_for_bs = migs_for_bs["sum_num_intmig"].sum()

    with open("./data/sector_fractions.json", "r") as f:
        bs_fractions = json.load(f)

    for k,v in bs_fractions.items():
        bs_fractions[k] = bs_fractions[k] * migs_for_bs
    

    changes = pd.read_csv("./map_layers/absolute_change.csv").sort_values(by = ["absolute_change"], ascending = False)
    changes["GEO2_MX"] = changes["GEO2_MX"].astype(str)
    with open("./data/shapeName_shapeID_dict.json", "r") as f:
        id_map = json.load(f)
    changes["GEO2_MX"] = changes["GEO2_MX"].astype(str).map(id_map)
    top_munis = changes["GEO2_MX"].to_list()[0:10]
    top_changes = changes["absolute_change"].round(2).to_list()[0:10]

    bottom_munis = changes["GEO2_MX"].to_list()[-10:][::-1]
    bottom_changes = changes["absolute_change"].round(2).to_list()[-10:][::-1]


    return {'change': int(change),
            'p_change': round(p_change, 2),
            'predicted_migrants': round(total_pred_migrants / 5, 0),
            'avg_age': round(avg_age, 0),
            'avg_age_change': round(avg_age_change, 0),
            'pavg_age_change': round(p_avg_age_change, 0),
            'corr_means': corr_means,
            'corr_category_dict': corr_category_dict,
            'bs_fractions_labels': list(bs_fractions.keys()),
            'bs_fractions_values': list(bs_fractions.values()),
            'model_error': f'{int((round(total_pred_migrants, 0) / 5) * MODEL_ERROR):,}',
            'top_munis': top_munis,
            'top_changes': top_changes,
            'bottom_munis': bottom_munis,
            'bottom_changes': bottom_changes,
            }







@APP.route('/get_border_data', methods=['GET'])
def get_border_data():

    """
    Function to set up the intital display of migrants to each border sector (i.e. baseline)
    Called on document setup
    """

    migs_for_bs = pd.read_csv(DATA_PATH)
    migs_for_bs = migs_for_bs["sum_num_intmig"].sum()

    with open("./data/sector_fractions.json", "r") as f:
        bs_fractions = json.load(f)

    for k,v in bs_fractions.items():
        bs_fractions[k] = bs_fractions[k] * migs_for_bs

    return {'bs_fractions_labels': list(bs_fractions.keys()),
            'bs_fractions_values': list(bs_fractions.values())}




@APP.route('/status_update', methods=['GET'])
def status_update():
    with open("./status.json", "r") as f:
        status = json.load(f)
    return {"status": status['status']}




@APP.route('/download_data', methods=['GET'])
def download_data():
    return send_from_directory("./data/",
                               "mexico2010.csv", as_attachment = True)




if __name__ == '__main__':
    APP.debug=True
    APP.run()


