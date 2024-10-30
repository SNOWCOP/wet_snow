#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:36:50 2024

@author: vpremier
"""


import asf_search as asf
import geopandas as gpd
from shapely.geometry import box
import datetime as dt
import os
import numpy as np
import pandas as pd


def query_s1(path_to_shapefile, start, end,
                platform = asf.PLATFORM.SENTINEL1,
                beammode = asf.BEAMMODE.IW,
                product_type = asf.PRODUCT_TYPE.GRD_HD,
                track = None, ow = False):
    
    #dates
    date_start = dt.datetime.strptime(start,"%Y-%m-%d")
    date_end = dt.datetime.strptime(end,"%Y-%m-%d")
    
    gdf = gpd.read_file(path_to_shapefile)
    
    # convert crs (otherwise may result in an error)
    if not gdf.crs == 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')  
        
    # Extract the Bounding Box Coordinates
    bounds = gdf.total_bounds
    
    #Create GeoDataFrame of the Bounding Box 
    gdf_bounds = gpd.GeoSeries([box(*bounds)])

    #Get WKT Coordinates
    wkt_aoi = gdf_bounds.to_wkt().values.tolist()[0] 
    
    # Define seacrh query
    results = asf.search(
        platform = platform,
        processingLevel=[beammode,product_type],
        start = date_start,
        end = date_end,
        intersectsWith = wkt_aoi
        )    

    # check if the file has already been downloaded
    results_fltd = results.copy()
    
    for r in results.data:
        
        fileName = r.properties['fileName']
        pathNumber = r.properties['pathNumber']

            
        if track is not None:
            if pathNumber != track:
                results_fltd.remove(r)
        
        
    print(f'Total matching Sentinel-1 data for the AOI: {len(results_fltd)}')
    
    
    # convert to a dataframe
    fileList = []
    pathNumber = []
    dateList = []
    for r in results.data:
        
        fileList.append(r.properties['fileName'])
        pathNumber.append(r.properties['pathNumber'])
        dateList.append(r.properties['fileName'].split('_')[4][:8])
          
        
    df = pd.DataFrame(list(zip(dateList,fileList,pathNumber)),
                      columns=['Date','fileName','track'])
    df = df.sort_values(by='Date')
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df






