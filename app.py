from flask import request, jsonify, Response
from pandas import json_normalize
import geopandas as gpd
import pandas as pd
import numpy as np
import geojson
import folium
import flask
import json
import io


# Create the application.
APP = flask.Flask(__name__)


GEOJSON_PATH = "./geoBoundariesSimplified-3_0_0-MEX-ADM2.geojson"
DATA_PATH = "./mex_migration_allvars_subset.csv"
MATCH_PATH = "./gB_IPUMS_match.csv"




with open(GEOJSON_PATH) as f:
    geodata_collection = geojson.load(f)



def map_column_names(var_names, df):
    for i in range(0, len(df.columns)):
        if df.columns[i] in var_names.keys():
            df = df.rename(columns = {df.columns[i]: var_names[df.columns[i]] })
    return df



@APP.route('/', methods=['GET','POST'])
def index():
    """ Displays the index page accessible at '/' """


    if request.method == "GET":

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
        cur_data = df[df['sending'] == 20240]

        # Open the variables JSON and the JSON containing the pretty translation of the variables
        with open("./vars copy.json", "r") as f:
            grouped_vars = json.load(f)

        with open("./var_map.json", "r") as f2:
            var_names = json.load(f2)

        # Get all of the variables to send to Flask
        # TO-DO: PUT THIS IN A FUNCTION SOMEWHERE
        # WAIT IS THIS EVEN NECCESSARY IF WE ARE ONLY DOING PERCENTAGE INCREASES RIP
        econ = cur_data[grouped_vars['Economic']]
        econ = map_column_names(var_names, econ)
        econ = zip(econ.columns, econ.iloc[0].to_list())
        
        demog = cur_data[grouped_vars['Deomographic']]
        demog = map_column_names(var_names, demog)
        demog = zip(demog.columns, demog.iloc[0].to_list())

        family = cur_data[grouped_vars['Family']]
        family = map_column_names(var_names, family)
        family = zip(family.columns, family.iloc[0].to_list())

        employ = cur_data[grouped_vars['Employment']]
        employ = map_column_names(var_names, employ)
        employ = zip(employ.columns, employ.iloc[0].to_list())

        health = cur_data[grouped_vars['Health']]
        health = map_column_names(var_names, health)
        health = zip(health.columns, health.iloc[0].to_list())

        edu = cur_data[grouped_vars['Education']]
        edu = map_column_names(var_names, edu)
        edu = zip(edu.columns, edu.iloc[0].to_list())

        hhold = cur_data[grouped_vars['Household']]
        hhold = map_column_names(var_names, hhold)
        hhold = zip(hhold.columns, hhold.iloc[0].to_list())

    # Merry Christmas HTML
    return flask.render_template('dashboard copy.html', 
                                  municipality_ids = municipality_ids, 
                                  econ_data = econ,
                                  demog_data = demog,
                                  family_data = family,
                                  health_data = health,
                                  edu_data = edu,
                                  employ_data = employ,
                                  hhold_data = hhold,
                                  total_migrants = total_migrants,
                                  hist = hist)



    # return dict(adms.iloc[0].T)







def convert_to_pandas(geodata_collection, MATCH_PATH, DATA_PATH):\

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







# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


# @APP.route('/', methods=['GET','POST'])
# def index():
#     """ Displays the index page accessible at '/' """
#     df = pd.read_csv("./us_migration.csv")
#     df_var_cols = [i for i in df.columns if i not in ['sending','US_MIG_05_10']]

#     if request.method == 'POST':
#         print("In POST")
#         print("Selected province: ", request.json['adm_id'])
#         adms = df[df['sending'] == int(request.json['adm_id'])]
#         adms = adms[df_var_cols]
#         return dict(adms.iloc[0].T)

#     if request.method == 'GET':
#         print("In GET")
#         adms = df['sending'].unique()
#         df = df[df_var_cols]
#         dta = zip(df.columns, df.iloc[0].to_list())

#         m = folium.Map(location=[23.6345, -92.5528], zoom_start=5)

#         folium.Choropleth(
#             geo_data=gdf_json,
#             name="choropleth",
#             data=mig,
#             columns=["B", "US_MIG_05_10"],
#             key_on="feature.properties.B",
#             fill_color="YlGn",
#             fill_opacity=1,
#             weight= 1,
#             legend_name="Number of migrants",
#         ).add_to(m)

#         polys = folium.features.GeoJson(gdf_json, style_function=style_function).add_to(m)

#         return flask.render_template('index.html', dta = dta, map = m._repr_html_(), adms = adms)#, shp = features[0]['geometry']['coordinates'])




# @APP.route('/update_map', methods=['GET','POST'])
# def update_map():

#     print('here!!')

#     if request.method == 'POST':

#         lat, lng = get_centroids(gdf)

#         m = folium.Map(location=[float(lng), float(lat)], zoom_start=9)
#         folium.Choropleth(
#             geo_data=gdf_json,
#             name="choropleth",
#             data=mig,
#             columns=["B", "US_MIG_05_10"],
#             key_on="feature.properties.B",
#             fill_color="YlGn",
#             fill_opacity=1,
#             weight= 1,
#             legend_name="Number of migrants",
#         ).add_to(m)

#         polys = folium.features.GeoJson(gdf_json, style_function=style_function).add_to(m)

#         return m._repr_html_()



# @APP.route('/pred_muni', methods=['GET','POST'])
# def pred_muni():
#     # if request.method == 'POST':
#     print("in pred muni function!")
    

#     new_vals = request.json['values']
#     new_vals = [float(i) for i in new_vals]
#     print(new_vals)

#     pred = pred_municipality(int(request.json['adm_id']), model, MIG_PATH, new_vals)

#     pred = pred.item()
#     print(pred)

#     return {'pred': pred}



# @APP.route('/plot_social_sig', methods=['GET','POST'])
# def plot_png():
#     df = pd.read_csv("./figs2/im" + str(1) + ".csv")
#     mpimg.imsave("./static/images/im2.png", np.reshape(np.array(df["0"]), (224,224)))
#     return {'url': "/static/images/im2.png"}

# # def create_figure():
# #     df = pd.read_csv("./figs2/im" + str(1) + ".csv")
# #     mpimg.imsave("./pics9/im" + str(1) + ".png", np.reshape(np.array(df["0"]), (224,224)))
# #     return {'url': "./pics9/im" + str(1) + ".png"}




if __name__ == '__main__':
    APP.debug=True
    APP.run()






# Economic:
# 'sending_salary_worker', 'sending_self_employed', 'sending_sum_income','sending_unknown_employment_status',
# 'sending_unpaid_worker','sending_weighted_avg_income',
# 'sending_weighted_avg_income_abroad',
# 'sending_weighted_avg_no_income_abroad',
# 'sending_weighted_avg_unknown_income_abroad',


# Demographic
# 'sending_citizen_unspecified', 'sending_citizenship_unknown',
# 'sending_indigeneity','sending_marriage_unknown', 'sending_married', 'sending_no_indigeneity',
# 'sending_not_citizen','sending_separated','sending_single','sending_unknown_indigeneity','sending_widowed


# Household
# 'sending_household_not_owned', 'sending_household_owned',
# 'sending_household_owned_unknown', 'sending_internet', 'sending_internet_unknown',
# 'sending_no_internet','sending_rural','sending_urban',


# Administrative:
# 'sending_total_pop',