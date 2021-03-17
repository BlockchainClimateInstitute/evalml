import importlib
import os
import warnings
from collections import namedtuple
from functools import reduce

import numpy as np
import pandas as pd
import woodwork as ww
from sklearn.utils import check_random_state

from evalml.exceptions import (
    EnsembleMissingPipelinesError,
    MissingComponentError
)
from evalml.utils import get_logger

logger = get_logger(__file__)

target = 'Price_p'


def deg2rad(degrees):
    return math.pi * degrees / 180.0

def rad2deg(radians):
    return 180.0 * radians / math.pi

def get_latmin(lat):
    halfSide = 1000 * box_in_km

    # Radius of Earth at given latitude
    An = WGS84_a * WGS84_a * math.cos(lat)
    Bn = WGS84_b * WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    radius = math.sqrt((An * An + Bn * Bn) / (Ad * Ad + Bd * Bd))

    # Radius of the parallel at given latitude
    pradius = radius * math.cos(lat)
    latMin = lat - halfSide / radius
    latMax = lat + halfSide / radius
    return latMin

def get_latmax(lat):
    halfSide = 1000 * box_in_km

    # Radius of Earth at given latitude
    An = WGS84_a * WGS84_a * math.cos(lat)
    Bn = WGS84_b * WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    radius = math.sqrt((An * An + Bn * Bn) / (Ad * Ad + Bd * Bd))

    # Radius of the parallel at given latitude
    pradius = radius * math.cos(lat)
    latMin = lat - halfSide / radius
    latMax = lat + halfSide / radius
    return latMax

def get_pradius(lat):
    halfSide = 1000 * box_in_km

    # Radius of Earth at given latitude
    An = WGS84_a * WGS84_a * math.cos(lat)
    Bn = WGS84_b * WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    radius = math.sqrt((An * An + Bn * Bn) / (Ad * Ad + Bd * Bd))

    # Radius of the parallel at given latitude
    pradius = radius * math.cos(lat)
    return pradius

def get_postcodeOutcode_from_postcode(postcode):
    return postcode.split()[0]

def get_postcode_from_address(string):
    return string.split(' ')[-2] + ' ' + string.split(' ')[-1]

def get_postcodeArea_from_outcode(postcodeArea):
    if postcodeArea[1].isnumeric():
        return postcodeArea[0]
    else:
        return postcodeArea[0:2]
      
def ReadParquetFile(bucketName, fileLocation):
    df = pd.DataFrame()
    prefix_objs = your_bucket.objects.filter(Prefix=fileLocation)
    for s3_file in prefix_objs:
        obj = s3_file.get()
        df = df.append(pd.read_parquet(io.BytesIO(obj['Body'].read())))
    return df
  
  
def download_postcodes(path='../data/raw/ukpostcodes'):
  
  zipurl = 'https://www.freemaptools.com/download/full-postcodes/ukpostcodes.zip'

  # Download the file from the URL
  zipresp = urlopen(zipurl)

  # Create a new file on the hard drive
  tempzip = open("/tmp/tempfile.zip", "wb")

  # Write the contents of the downloaded file into the new file
  tempzip.write(zipresp.read())

  # Close the newly-created file
  tempzip.close()

  # Re-open the newly-created file with ZipFile()
  zf = ZipFile("/tmp/tempfile.zip")

  # Extract its contents into <extraction_path>
  # note that extractall will automatically create the path
  zf.extractall(path)

  # close the ZipFile instance
  zf.close()
  
def preprocess_data(data):
    match_types = ['1. Address Matched', '2. Address Matched No Spec', '3. No in Address Matched']
    keep_cols = [x for x in data.columns if '_e' in x or x in ['Longitude_m','Latitude_m', 'Postcode']]
    keep_cols.append('Price_p')
    data = data[data['TypeOfMatching_m'].isin(match_types)]
    data = data[keep_cols]
    data['POSTCODE'] = data['FULLADRESS_e'].apply(get_postcode_from_address)
    data['POSTCODE_OUTCODE'] = data['Postcode'].apply(get_postcodeOutcode_from_postcode)
    data['POSTCODE_AREA'] = data['POSTCODE_OUTCODE'].apply(get_postcodeArea_from_outcode)
    
    # drop outliers, convert floor level to integers
    data['Rooms'] = (data['NUMBER_HABITABLE_ROOMS_e'].astype(float) + data['NUMBER_HEATED_ROOMS_e'].astype(float)) / float(2)
    data = data[data['Rooms'] < 20]
    data = data[data['NUMBER_HABITABLE_ROOMS_e'].astype(float) < 15]
    data = data[data['NUMBER_HEATED_ROOMS_e'].astype(float) < 12]
    data = data[data['FLAT_STOREY_COUNT_e'].astype(float).fillna(0) < 20]
    data = data[data['TOTAL_FLOOR_AREA_e'].astype(float) < 800]
    data = data[data['TOTAL_FLOOR_AREA_e'].astype(float).fillna(0) < 400]
    data = data[data['TOTAL_FLOOR_AREA_e'].astype(float) > 0]
    data['PROPERTY_TYPE_e'] = data['PROPERTY_TYPE_e'].astype(str).replace('nan',np.nan).fillna('No PROPERTY_TYPE_e').astype(str)
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('NO DATA!','0')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('NODATA!','0')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('Basement','-1')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('Ground','0')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('1st','1')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('2nd','2')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('3rd','3')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('4th','4')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('5th','5')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('6th','6')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('9th','9')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('11th','11')
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].replace('12th','12')
    floor_levels = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','-1']
    floor_levels = [x for x in data['FLOOR_LEVEL_e'].values if x not in floor_levels]
    data['FLOOR_LEVEL_e'] = np.where(data['FLOOR_LEVEL_e'].isin(floor_levels), np.nan, data['FLOOR_LEVEL_e']).astype(float)
    data['NUMBER_HEATED_ROOMS_e'] = data['NUMBER_HEATED_ROOMS_e'].astype(str).replace('nan',np.nan).fillna('0').astype(float)
    data = data[['POSTCODE', 
                 'POSTCODE_OUTCODE', 
                 'POSTCODE_AREA', 
                 'POSTTOWN_e', 
                 'PROPERTY_TYPE_e', 
                 'TOTAL_FLOOR_AREA_e', 
                 'NUMBER_HEATED_ROOMS_e', 
                 'FLOOR_LEVEL_e', 
                 'Latitude_m', 
                 'Longitude_m', 
                 target]]
    
    data = data.reset_index()
    data = data.drop('index',axis=1)
    data = data.reset_index()
    data = data.rename({'index':'unit_indx'},axis=1)
    data = data.reset_index()
    data = data.drop('index',axis=1)
    data['TOTAL_FLOOR_AREA_e'] = data['TOTAL_FLOOR_AREA_e'].astype(float)
    data['FLOOR_LEVEL_e'] = data['FLOOR_LEVEL_e'].astype(float)
    data['NUMBER_HEATED_ROOMS_e'] = data['NUMBER_HEATED_ROOMS_e'].astype(float)
    data[target] = data[target].astype(float)
    X = data.drop('POSTCODE_AREA',axis=1)
    y = data['POSTCODE_AREA']
    X_train, X_holdout, y_train, y_holdout = evalml.preprocessing.utils.split_data(X, y, problem_type='multiclass', test_size=0.25)
    
    validate = X_holdout.to_dataframe().reset_index().drop('index',axis=1)
    validate['POSTCODE_AREA'] = y_holdout.to_series().reset_index()['POSTCODE_AREA']

    train = X_train.to_dataframe().reset_index().drop('index',axis=1)
    train['POSTCODE_AREA'] = y_train.to_series().reset_index()['POSTCODE_AREA']

    return train,  validate
