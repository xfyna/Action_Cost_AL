# -*- coding: utf-8 -*-
"""
Last edited 10/28/2024

@author: sapph
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from osgeo import gdal
from numpy.linalg import inv
import math
import torch
import gpytorch
import sklearn.metrics
import time
import os
import glob
import pickle
import re
import cv2
import sys
import pickle
import random

def find_local_extrema(z_plot, grid_size, surface_type, adaptive_threshold=True):
    """
    Identifies local minima (wells) and maxima (peaks) in a given surface.
    
    Parameters:
    - z_plot: Flattened surface data (1D array).
    - grid_size: Size of the 2D grid (int).
    - surface_type: Type of surface ('parabola', 'townsend', 'lunar', or 'generic').
    - adaptive_threshold: Whether to apply adaptive thresholding (bool).
    
    Returns:
    - num_local_min: Number of local minima (wells).
    - num_local_max: Number of local maxima (peaks).
    - minima_coords: Coordinates of local minima (list of tuples).
    - maxima_coords: Coordinates of local maxima (list of tuples).
    """
    # Step 1: Reshape and apply Gaussian smoothing based on surface type
    z_reshaped = z_plot.reshape(grid_size, grid_size)
    
    # Adjust smoothing (sigma) based on surface type
    if surface_type == 'parabola':
        sigma = 1  # Light smoothing for simpler surfaces
    elif surface_type == 'townsend':
        sigma = 0.3  # Moderate smoothing for complex surfaces
    elif surface_type == 'lunar':
        sigma = 0.3  # Medium smoothing for moderately complex surfaces
    else:
        sigma = 1.0  # Default smoothing
    
    z_smoothed = gaussian_filter(z_reshaped, sigma=sigma)
    
    # Step 2: Identify local minima (wells)
    local_minima_indices = argrelextrema(z_smoothed, np.less, order=3)  # Higher order for broader minima detection
    local_maxima_indices = argrelextrema(z_smoothed, np.greater, order=3)  # Detect local maxima (peaks)
    
    # Convert minima and maxima indices to 2D coordinates
    minima_coords = np.array(local_minima_indices).T
    maxima_coords = np.array(local_maxima_indices).T
    
    # Step 3: Get minima and maxima values from smoothed surface
    minima_values = z_smoothed[local_minima_indices]
    maxima_values = z_smoothed[local_maxima_indices]
    
    # Step 4: Apply adaptive thresholding based on surface type
    if adaptive_threshold:
        z_min, z_max = np.min(z_plot), np.max(z_plot)
        
        # Dynamic threshold based on surface type
        if surface_type == 'parabola':
            min_threshold = z_min + 0.01 * (z_max - z_min)  # Lower threshold for smooth surfaces
            max_threshold = z_max - 0.01 * (z_max - z_min)
        elif surface_type == 'townsend':
            min_threshold = z_min + 0.1 * (z_max - z_min)  # Higher threshold for complex surfaces
            max_threshold = z_max - 0.1 * (z_max - z_min)
        elif surface_type == 'lunar':
            min_threshold = z_min + 0.1 * (z_max - z_min)  # Medium threshold for lunar surface
            max_threshold = z_max - 0.1 * (z_max - z_min)
        else:
            min_threshold = z_min + 0.1 * (z_max - z_min)  # Default threshold
            max_threshold = z_max - 0.1 * (z_max - z_min)
        
        minima_coords = minima_coords[minima_values < min_threshold]
        maxima_coords = maxima_coords[maxima_values > max_threshold]
    
    # Step 5: Return the number of wells and peaks, along with their coordinates
    num_local_min = len(minima_coords)
    num_local_max = len(maxima_coords)
    
    return num_local_min, num_local_max, minima_coords, maxima_coords

def nearest_neighbor(x_sample,x_list,y_list):
    disp = np.sqrt((x_sample[0]-x_list[:,0])**2+(x_sample[1]-x_list[:,1])**2)
    i_min = np.argmin(disp)
    y_sample = y_list[i_min]
    return y_sample

# For the Parabola surface
def generate_parabola_surface():
    grid_bounds = [(-1, 1), (-1, 1)]
    grid_size = 21
    x_1 = np.linspace(grid_bounds[0][0], grid_bounds[0][1], grid_size)
    x_2 = x_1
    grid_diff = x_1[1] - x_1[0]
    x_plot, y_plot = np.meshgrid(x_1, x_2)
    z_plot = x_plot**2 + y_plot**2
    x_vec = np.array([x_plot.ravel()])
    y_vec = np.array([y_plot.ravel()])
    x_true = np.concatenate((x_vec.transpose(),y_vec.transpose()), axis=1)
    y_true = x_true[:, 0]**2 + x_true[:, 1]**2 
    n_samples = len(y_true)
    y_obs = y_true
    length = 109
    z_plot = y_true

    return z_plot, grid_size

# For the Townsend surface
def generate_townsend_surface():
    grid_bounds = [(-2.5, 2.5), (-2.5, 2.5)]
    grid_size = 21
    x_1 = np.linspace(grid_bounds[0][0], grid_bounds[0][1], grid_size)
    x_2 = x_1
    grid_diff = x_1[1] - x_1[0]
    x_plot, y_plot = np.meshgrid(x_1, x_2)
    z_plot = -(np.cos((x_plot-0.1)*y_plot))**2-x_plot*np.sin(3*x_plot+y_plot)
    x_vec = np.array([x_plot.ravel()])
    y_vec = np.array([y_plot.ravel()])
    x_true = np.concatenate((x_vec.transpose(),y_vec.transpose()), axis=1)
    y_true = -(np.cos((x_true[:,0]-0.1)*x_true[:,1]))**2-x_true[:,0]*np.sin(3*x_true[:,0]+x_true[:,1])
    n_samples = len(y_true)
    y_obs = y_true
    length = 109
    z_plot = y_true
    return z_plot, grid_size

# For the Lunar surface
def generate_lunar_surface():
#generation of the lunar surface
    length = 155
    file_name1 = 'Shoemaker_5mDEM.tif'
    # You need to multiply 0.5 for each pixel value to get the actual elevation.
    Aimg = gdal.Open(file_name1)
    A = Aimg.GetRasterBand(1).ReadAsArray()

    file_name2 = 'Shoemaker_280mIceExposures.tif'
    Bimg = gdal.Open(file_name2)
    B = Bimg.GetRasterBand(1).ReadAsArray()

    file_name3 = 'Shoemaker_250mLAMP-OnOffRatio.tif'
    Cimg = gdal.Open(file_name3)
    C = Cimg.GetRasterBand(1).ReadAsArray()

    # make DEMs and other maps
    # to build a DEM, each index in row and column is 5 m
    (n_y,n_x) = np.shape(A)
    spacing = 5.0
    x_vec_grid5 = np.array(range(n_x))*spacing
    y_vec_grid5 = np.array(range(n_y))*spacing
    x_mat5, y_mat5 = np.meshgrid(x_vec_grid5, y_vec_grid5)
    z_mat5 = A/2
    z_mat5 = np.where(z_mat5==32767/2, np.nan, z_mat5) 
    z_min5 = min(z_mat5[~np.isnan(z_mat5)])
    z_max5 = max(z_mat5[~np.isnan(z_mat5)])
    grid_diff = 0.25

    # unravel grid data
    x_DEM5 = x_mat5.ravel()
    y_DEM5 = y_mat5.ravel()
    z_DEM5 = z_mat5.ravel()

    #  parse ice data distance 280 m
    (n_y,n_x) = np.shape(B)
    spacing = 280.0
    x_vec_grid280 = np.array(range(n_x))*spacing
    y_vec_grid280 = np.array(range(n_y))*spacing
    x_mat280, y_mat280 = np.meshgrid(x_vec_grid280, y_vec_grid280)
    z_mat280 = z_mat5[::56,::56]
    z_mat280 = z_mat280[0:n_y,0:n_x]

    # unravel grid data
    x_DEM280 = x_mat280.ravel()
    y_DEM280 = y_mat280.ravel()
    z_DEM280 = z_mat280.ravel()
    ice_DEM280 = B.ravel()

    #  parse LAMP data distance 250m
    (n_y,n_x) = np.shape(C)
    spacing = 250.0
    x_vec_grid250 = np.array(range(n_x))*spacing
    y_vec_grid250 = np.array(range(n_y))*spacing
    x_mat250, y_mat250 = np.meshgrid(x_vec_grid250, y_vec_grid250)
    z_mat250 = z_mat5[::50,::50]
    # unravel grid data
    x_DEM250 = x_mat250.ravel()
    y_DEM250 = y_mat250.ravel()
    z_DEM250 = z_mat250.ravel()

    C = np.where(C==-9999, np.nan, C) 
    c_min = min(C[~np.isnan(C)])
    c_max = max(C[~np.isnan(C)])
    c_DEM250 = C.ravel()

    # let's make LAMP data the elevation
    LAMP_DEM280 = np.zeros(len(x_DEM280))
    x_list = np.array([x_DEM250,y_DEM250]).transpose()
    for i in range(len(x_DEM280)):
        x_sample = np.array([x_DEM280[i],y_DEM280[i]])
        LAMP_DEM280[i] = nearest_neighbor(x_sample,x_list,c_DEM250)
    # % clean up data
    # training data input is DEM position 
    x_true = np.array([x_DEM250/1000, y_DEM250/1000, z_DEM250/1000]).transpose()
    # training data output is LAMP
    y_obs =  np.double(c_DEM250)

    # get rid of elevation nan values
    y_obs =  y_obs[~np.isnan(x_true[:,2])]
    x_true = x_true[~np.isnan(x_true[:,2]),:]
    # get rid of LAMP data
    x_true = x_true[~np.isnan(y_obs),:]
    y_obs =  y_obs[~np.isnan(y_obs)]

    x_true_doub = x_true
    y_obs_doub = y_obs

    for i in range(x_true.shape[0]):
        y_obs_doub[i] = np.float64(y_obs[i])
        for j in range(x_true.shape[1]):
            x_true_doub[i, j] = np.float64(x_true[i, j])

    r_disp = 6
    #constant center location
    x_center_all = np.mean(x_true,0)
    x_disp = np.sqrt((x_true[:,0]-x_center_all[0])**2 + (x_true[:,1]-x_center_all[1])**2 + (x_true[:,2]-x_center_all[2])**2)
    i_min = np.argmin(x_disp)
    x_center = x_true[i_min,:]
    
    x_true = x_true_doub - x_center
    
    y_obs = y_obs[np.argwhere(x_true[:,0]>=-r_disp/2)[:,0]]
    x_true = x_true[np.argwhere(x_true[:,0]>=-r_disp/2)[:,0]]
    y_obs = y_obs[np.argwhere(x_true[:,1]>=-r_disp/2)[:,0]]
    x_true = x_true[np.argwhere(x_true[:,1]>=-r_disp/2)[:,0]]
    y_obs = y_obs[np.argwhere(x_true[:,0]<=r_disp/2)[:,0]]
    x_true = x_true[np.argwhere(x_true[:,0]<=r_disp/2)[:,0]]
    y_obs = y_obs[np.argwhere(x_true[:,1]<=r_disp/2)[:,0]]
    x_true = x_true[np.argwhere(x_true[:,1]<=r_disp/2)[:,0]]
    
    y_true = y_obs 
    x_true_2 = x_true
    x_true = x_true[:, :2]
    n_samples = len(y_obs)
    sigma = math.sqrt(0.02)
    z_plot = y_obs  # Assuming you want elevation as output
    grid_size = 25 
    
    return z_plot, grid_size

# Generate surfaces
z_parabola, grid_size_parabola = generate_parabola_surface()
z_townsend, grid_size_townsend = generate_townsend_surface()
z_lunar, grid_size_lunar = generate_lunar_surface()

# Find local minima and maxima for each surface
parabola_min, parabola_max, parabola_min_coords, parabola_max_coords = find_local_extrema(z_parabola, grid_size_parabola, 'parabola')
townsend_min, townsend_max, townsend_min_coords, townsend_max_coords = find_local_extrema(z_townsend, grid_size_townsend, 'townsend')
lunar_min, lunar_max, lunar_min_coords, lunar_max_coords = find_local_extrema(z_lunar, grid_size_lunar, 'lunar')

# Calculate y_max and y_min for each surface
y_max_min = {
    'parabola': {
        'y_max': np.max(z_parabola),
        'y_min': np.min(z_parabola)
    },
    'townsend': {
        'y_max': np.max(z_townsend),
        'y_min': np.min(z_townsend)
    },
    'lunar': {
        'y_max': np.max(z_lunar),
        'y_min': np.min(z_lunar)
    }
}

# Print results
print(f"Parabola surface has {parabola_min} local minima and {parabola_max} local maxima.")
print(f"Townsend surface has {townsend_min} local minima and {townsend_max} local maxima.")
print(f"Lunar surface has {lunar_min} local minima and {lunar_max} local maxima.")

# Print y_max and y_min for each surface
for surface, values in y_max_min.items():
    print(f"{surface.capitalize()} surface: y_max = {values['y_max']}, y_min = {values['y_min']}")
