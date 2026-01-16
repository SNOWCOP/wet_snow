#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 12:16:14 2026

@author: vpremier
wit the contribution of Sander van Rijsewijk
"""
import geopandas as gpd
import openeo
from shapely.geometry import Point, mapping
from query import query_s1
import pandas as pd
from datetime import timedelta
import time
import numpy as np

# ----------------------------- Functions ------------------------------------


def multitemporal_filter(gamma0_datacube, kernel_size=5, border_value=0):

    # Apply mean kernel on remaining acquisitions
    kernel = np.ones((kernel_size, kernel_size)) / \
        (kernel_size**2)  # Mean kernel
    kernel_list = kernel.tolist()
    local_mean = gamma0_datacube.apply_kernel(
        kernel=kernel_list,
        border=0  # "replicate" not supported on vito - be careful about edge effects!
    )

    # Apply multitemporal filter equation
    norm_ratio = gamma0_datacube.divide(
        local_mean)  # datacube with I_i / E[I_i]
    norm_sum = norm_ratio.reduce_temporal("mean")  # 1/N * sum of I_i / E[I_i]
    filtered = norm_sum.multiply(local_mean)  # E[I_k]/N * sum of I_i / E[I_i]
    filtered = 10 * filtered.log10()  # Convert to dB
    return filtered


def extract_dates(df):
    dates = (
        df
        .groupby("track")["Date"]
        .apply(
            lambda dates: [
                [d.strftime("%Y-%m-%d"), (d + timedelta(days=1)
                                          ).strftime("%Y-%m-%d")]
                for d in sorted(dates.unique())
            ]
        )
        .to_dict()
    )
    return dates


def load_sparse_dates(conn, temporal_extent, bbox, date_pairs, bands):
    # load datacube for a specific track
    cube = None
    for temporal_extent in date_pairs:
        s1_datacube = conn.load_collection(
            "SENTINEL1_GRD",
            temporal_extent=temporal_extent,
            spatial_extent={'west': bbox[0],
                            'east': bbox[2],
                            'south': bbox[1],
                            'north': bbox[3],
                            'crs': 4326},
            bands=bands,
        )
        if cube is None:
            cube = s1_datacube
        else:
            cube = cube.merge_cubes(s1_datacube, overlap_resolver="max")
            
    return cube


def mean_filter_boolean(cube, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    kernel_list = kernel.tolist()

    local_mean = cube.apply_kernel(
        kernel=kernel_list,
        border=0  # zero padding (edge effects!)
    )

    # Step 2 — Threshold back to boolean
    return local_mean > 0.5


def compute_W(LIA, k, theta_1, theta_2):
    cond1 = LIA < theta_1
    cond2 = (LIA >= theta_1) & (LIA <= theta_2)
    cond3 = LIA > theta_2

    W = (
        cond1 * 1.0
        + cond2 * (k * (1 + (45 - LIA) / (theta_2-theta_1)))
        + cond3 * k
    )

    return W

# ----------------------------- End of Functions ------------------------------------


# study area
shp_path = r'/mnt/CEPH_PROJECTS/SNOWCOP/AOI/s1_area_extent/s1_area_extent.shp'
ref_year = 2023
track = 18
threshold = -2 # in dB

# parameters
theta_1 = 20
theta_2 = 45 
k = 0.5

# Define date range
start_date = f"{ref_year}-04-01"
end_date = f"{ref_year + 1}-04-01"

# Define the reference period
start_ref_date = f"{ref_year + 1}-01-01"
end_ref_date = f"{ref_year + 1}-01-31"


# derive dates and track information from the ASF catalogue
s1_list = query_s1(shp_path, start_date, end_date)
s1_list["Date"] = pd.to_datetime(s1_list["Date"])
print(s1_list.head())

# get the track list
tracks = s1_list["track"].unique().tolist()
dates = extract_dates(s1_list)
print(f"The area covers {len(tracks)} tracks, i.e. {tracks}")


# query the reference period
s1_ref_list = query_s1(shp_path, start_ref_date, end_ref_date)
s1_ref_list["Date"] = pd.to_datetime(s1_ref_list["Date"])
dates_ref = extract_dates(s1_ref_list)
print(s1_ref_list.head())


# -----------------------------------------------------------------

# Connect and authenticate
conn = openeo.connect(url="openeo.vito.be")
conn.authenticate_oidc()

start_time = time.time()

# Geometry from shapefile
gdf = gpd.read_file(shp_path)
# To EPSG:4326 since load_collection expects this by default
gdf = gdf.to_crs(epsg=4326)

bbox = gdf.bounds.iloc[0]

# load all the images
date_pairs = dates[track]
s1 = load_sparse_dates(conn, [start_date, end_date], bbox, date_pairs, ["VV", "VH"])


# convert to backscatter
s1_backscatter = s1.sar_backscatter(
    coefficient="gamma0-terrain",
    elevation_model="COPERNICUS_30",
    local_incidence_angle=True)


# Seperate backscatter and angle bands
s1_vv = s1_backscatter.filter_bands("VV")
s1_vh = s1_backscatter.filter_bands("VH")
s1_lia = s1_backscatter.filter_bands("local_incidence_angle")


# multi-temporal filter
s1_vv_fltd = multitemporal_filter(s1_vv)
s1_vh_fltd = multitemporal_filter(s1_vh)

# Select the reference 
s1_vv_winter = s1_vv_fltd.filter_temporal(start_ref_date, end_ref_date)
s1_vv_ref = s1_vv_winter.reduce_dimension(reducer="mean",dimension="bands")

s1_vh_winter = s1_vh_fltd.filter_temporal(start_ref_date, end_ref_date)
s1_vh_ref = s1_vh_winter.reduce_dimension(reducer="mean",dimension="bands")

#Compute the difference w.r.t. the reference (dual polarization)
R_vv = s1_vv - s1_vv_ref
R_vh = s1_vh - s1_vh_ref

W = compute_W(s1_lia, k, theta_1, theta_2)
R_c = W*R_vh + (1-W)*R_vv

# Get the wet/dry snow map
wetdry_map = R_c < threshold
#wetdry_map.download("./output/wetdry.nc")

# In the ATBD: apply a median filter, here replaced with a mean filter
wetdry_map_fltd = mean_filter_boolean(wetdry_map, kernel_size=3)      
    

# masking out layover and shadow areas, lia>75 or <15
lia_mask = (s1_lia >= 15) & (s1_lia <= 75)
masked_cube = wetdry_map_fltd.mask(lia_mask)

# masking forest and water bodies

# repeat for each track
masked_cube.download("./output/wetsnowmaps_track83.tif")


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken is: {elapsed_time:.3f} seconds")









