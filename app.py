#!/usr/bin/env python

import geopandas as gpd
from flask import request
import pandas as pd
import geojson
import folium
import flask

# from helpers import *

import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import json



# Create the application.
APP = flask.Flask(__name__)



def map_column_names(var_names, df):
    for i in range(0, len(df.columns)):
        if df.columns[i] in var_names.keys():
            df = df.rename(columns = {df.columns[i]: var_names[df.columns[i]] })
    return df



@APP.route('/', methods=['GET','POST'])
def index():
    """ Displays the index page accessible at '/' """


    if request.method == "GET":
        df = pd.read_csv("./mex_migration_allvars_subset.csv")
        municipality_ids = df['sending'].unique()
        df_var_cols = [i for i in df.columns if i not in ['sending','number_moved']]
        # print(request)
        cur_data = df[df['sending'] == 20240]

        print(cur_data)

        with open("./vars copy.json", "r") as f:
            grouped_vars = json.load(f)

        with open("./var_map.json", "r") as f2:
            var_names = json.load(f2)

        print(var_names)        


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


    

    
    # adms = adms[df_var_cols]

    return flask.render_template('dashboard copy.html', 
                                  municipality_ids = municipality_ids, 
                                  econ_data = econ,
                                  demog_data = demog,
                                  family_data = family,
                                  health_data = health,
                                  edu_data = edu,
                                  employ_data = employ,
                                  hhold_data = hhold)#, municipality_ids = municipality_ids)



    # return dict(adms.iloc[0].T)




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