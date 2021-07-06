from pandas import json_normalize
import pandas as pd
import geojson

DATA_PATH = "./data/mexico2010.csv"
BORDER_STATIONS_PATH = "./data/border_stations7.geojson"

with open(BORDER_STATIONS_PATH) as bs:
    border_stations = geojson.load(bs)

feature_df = json_normalize(border_stations["features"])
print("Total migrants at border stations: ", feature_df['properties.total_migrants'].sum())


df = pd.read_csv(DATA_PATH)
print("Total migrants: ", df['sum_num_intmig'].sum())
