# -*- coding: utf-8 -*-
"""
Last edited 10/28/2024

@primary author: frank
@secondary editor: sapph
"""

import numpy as np
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

#cmd argument required when run
func_name = sys.argv[1]
trial_number = sys.argv[2]

#model parameters
kernel_type = 'RBF'
poly_rank = 4
r_disp = 6.0
explore_name = 'Conventional'
con_name = 'Unconstrained'
noise_name = 'Noisy'

results = {
    'distance_a': [],
    'variance_a': [],
    'rmse_a': [],
    'trial_number': trial_number
}

def nearest_neighbor(x_sample,x_list,y_list):
    disp = np.sqrt((x_sample[0]-x_list[:,0])**2+(x_sample[1]-x_list[:,1])**2)
    i_min = np.argmin(disp)
    y_sample = y_list[i_min]
    return y_sample

#%% surfaces
#let's create a 2D convex surface 
if func_name == 'Parabola':
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
    sigma = math.sqrt(0.02)
    y_obs = y_true
    length = 109
    if noise_name == 'Noisy':
        y_obs = y_true + np.random.rand(n_samples) * sigma
    
# and now a nonconvex surface, townsend function (https://en.wikipedia.org/w/index.php?title=Test_functions_for_optimization&oldid=787014841)
elif func_name == 'Townsend':
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
    sigma = math.sqrt(0.02)
    y_obs = y_true
    length = 649
    if noise_name == 'Noisy':
        y_obs = y_true + np.random.rand(n_samples) * sigma
    # min resides at (2.5510, 0.0258)

elif func_name == 'Lunar':
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

    #%% focus on a 1.5 km radius at the bottom of the crater. center about the lowest point
    r_disp = 6

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

#file naming convention & output
trial_name = str(func_name)+'_'+str(noise_name)+'_'+str(con_name)+'_'+str(explore_name)+'_'+str(kernel_type)+'_'+str(trial_number)
parent_dir = '../GPAL/'
image_path = os.path.join(parent_dir, trial_name + '/')
os.mkdir(image_path)
stdoutOrigin=sys.stdout 
sys.stdout = open(image_path+"log.txt", "w")

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def nearest_neighbor(x_sample,x_list,y_list):
    disp = np.sqrt((x_sample[0]-x_list[:,0])**2+(x_sample[1]-x_list[:,1])**2)
    i_min = np.argmin(disp)
    y_sample = y_list[i_min]
    return y_sample

# define the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):    
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
    
        if kernel_type == 'RBF':
        # RBF Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 2)) 
        elif kernel_type == 'Matern':
        # Matern Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims = 2))
        elif kernel_type == 'Periodic':
        # Periodic Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel_type == 'Piece_Polynomial':
        # Piecewise Polynomial Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PiecewisePolynomialKernel(ard_num_dims = 2))
        elif kernel_type == 'RQ':
        # RQ Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims = 2))
        elif kernel_type == 'Cosine': # !
        # Cosine Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel(ard_num_dims = 2))
        elif kernel_type == 'Linear':
        # Linear Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(ard_num_dims = 2))
        elif kernel_type == 'Polynomial':
        # Polynomial Kernel 
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(ard_num_dims = 2, power = 4))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
 
def kernel_print(i, training_iter, loss, model):
    if kernel_type == 'RBF' or kernel_type == 'Matern' or kernel_type == 'Piece_Polynomial':
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.detach().numpy()[0][0],
                model.covar_module.base_kernel.lengthscale.detach().numpy()[0][1],
                model.likelihood.noise.detach().numpy()
            ))
    elif kernel_type == 'Periodic': 
    # Periodic Kernel
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f  period_length: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.detach().numpy(),
                model.covar_module.base_kernel.period_length.detach().numpy()
            ))
    elif kernel_type == 'RQ':
    # RQ Kernel
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f %.3f   alpha: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.detach().numpy()[0][0],
                model.covar_module.base_kernel.lengthscale.detach().numpy()[0][1],
                model.covar_module.base_kernel.alpha.detach().numpy()
            )) 
    elif kernel_type == 'Cosine': # !
    # Cosine Kernel
        print('Iter %d/%d - Loss: %.3f   period_length: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.period_length.detach().numpy()
            ))
    elif kernel_type == 'Linear':
    # Linear Kernel
        print('Iter %d/%d - Loss: %.3f   variance: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.variance.item()
            ))
    elif kernel_type == 'Polynomial':
    # Polynomial Kernel 
        print('Iter %d/%d - Loss: %.3f   offset: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.offset.detach().numpy()
            ))
   
def unique_sample(i_sample,i_set,i_train,i_max,x):
    if i_sample <= i_max and i_sample >= 0:
        if i_sample not in i_train:
            i_new = i_sample
        else:
            i_set_unique = set(i_set)-set(i_train)
            if not i_set_unique:
                return []
            i_set_unique = list(i_set_unique)
            x_start = x[i_sample,:]
            x_disp = np.sqrt((x[i_set_unique,0]-x_start[0])**2 + (x[i_set_unique,1]-x_start[1])**2)
            # disp_i = np.abs(np.array(i_set_unique)-np.array(i_sample))
            i_new =i_set_unique[np.argmin(x_disp)]
    elif i_sample > i_max:
        i_new = unique_sample(i_sample-1,i_set,i_train,i_max)
    else:
        i_new = unique_sample(i_sample+1,i_set,i_train,i_max)
    return i_new

def sample_disp_con(x,x_start,r_disp):
    # x_start = x[i_start,:]
    x_disp = np.sqrt((x[:,0]-x_start[0])**2 + (x[:,1]-x_start[1])**2)
    i_con = np.argwhere(x_disp<=r_disp)
    i_con = np.sort(i_con)
    return list(i_con[:,0])

def GPtrain(x_train, y_train, training_iter):
    
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x_train, y_train, likelihood)
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # train GP model
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        # kernel_print(i, training_iter, loss, model)
        optimizer.step()
    
    return likelihood, model, optimizer, output, loss

def kernel_print(i, training_iter, loss, model):
    if kernel_type == 'RBF' or kernel_type == 'Matern' or kernel_type == 'Piece_Polynomial':
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.detach().numpy()[0][0],
                model.covar_module.base_kernel.lengthscale.detach().numpy()[0][1],
                model.likelihood.noise.detach().numpy()
            ))
    elif kernel_type == 'Periodic': 
    # Periodic Kernel
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f  period_length: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.detach().numpy(),
                model.covar_module.base_kernel.period_length.detach().numpy()
            ))
    elif kernel_type == 'RQ':
    # RQ Kernel
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f %.3f   alpha: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.detach().numpy()[0][0],
                model.covar_module.base_kernel.lengthscale.detach().numpy()[0][1],
                model.covar_module.base_kernel.alpha.detach().numpy()
            )) 
    elif kernel_type == 'Cosine': # !
    # Cosine Kernel
        print('Iter %d/%d - Loss: %.3f   period_length: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.period_length.detach().numpy()
            ))
    elif kernel_type == 'Linear':
    # Linear Kernel
        print('Iter %d/%d - Loss: %.3f   variance: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.variance.item()
            ))
    elif kernel_type == 'Polynomial':
    # Polynomial Kernel 
        print('Iter %d/%d - Loss: %.3f   offset: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.offset.detach().numpy()
            ))

def GPeval(x_test, model, likelihood):
    
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x_test))
    
    f_preds = model(x_test)
    y_preds = likelihood(model(x_test))
    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_var = np.diag(f_preds.lazy_covariance_matrix.numpy())
    f_covar = f_preds.covariance_matrix
    
    with torch.no_grad():
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        
    return observed_pred, lower, upper

def RMS(y_1,y_2):
    return math.sqrt(sklearn.metrics.mean_squared_error(y_1, y_2))

def sample_disp_con(x,x_start,r_disp):
    # x_start = x[i_start,:]
    x_disp = np.sqrt((x[:,0]-x_start[0])**2 + (x[:,1]-x_start[1])**2) #+ (x[:,2]-x_start[2])**2)
    i_con = np.argwhere(x_disp<=r_disp)
    i_con = np.sort(i_con)
    return list(i_con[:,0])

#%% plotting
def plotBoth():
    
    #autoscale
    ax1 = fig.add_subplot(3, 2, 1, projection='3d')
    ax1.set_title('Conventional AL [Unconstrained] Prediction vs. Truth')
    ax1.plot_trisurf(x_true[:, 0], x_true[:, 1], y_obs, cmap='inferno', linewidth=0, alpha=0.25, vmax=max(y_obs), vmin=min(y_obs))
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_zlabel("$y$")
    
    if 'X_train_full_a' in globals() and 'i_train_GP_a' in globals() and 'y_pred_GP_a' in globals():
        X_train_GP_a = X_train_full_a[i_train_GP_a, :]
        y_train_GP_a = y_train_full_a[i_train_GP_a]
        X_test_GP_a = X_train_full_a[i_test_GP_a, :]
        y_test_GP_a = y_train_full_a[i_test_GP_a]

        ax1.scatter3D(X_train_GP_a[-1, 0], X_train_GP_a[-1, 1], y_train_GP_a[-1], s=100, color='black', marker='*', zorder=1)
        ax1.plot3D(X_train_GP_a[:, 0], X_train_GP_a[:, 1], y_train_GP_a, color='black')
        ax1.plot_trisurf(X_test_GP_a[:, 0], X_test_GP_a[:, 1], y_pred_GP_a, color='grey', alpha=0.25)  
      
    # RMSE vs. Distance
    ax4 = fig.add_subplot(3, 2, 2)
    ax4.set_title('RMS Error vs. Distance')
    if 'rover_distance_a' in globals() and 'y_RMS_GP_a' in globals():
        A_RMSD = ax4.plot(np.linspace(0, rover_distance_a[-1], len(y_RMS_GP_a)), y_RMS_GP_a, color = 'blue', marker ='.')
    ax4.set_ylabel('RMS Error')
    ax4.set_xlabel('Distance')
    ax4.legend(['Conventional'], loc='lower right')
    
    # RMSE vs. Samples
    ax5 = fig.add_subplot(3, 2, 3)
    ax5.set_title('RMS Error vs. Samples')
    if 'std_GP_a' in globals() and 'y_RMS_GP_a' in globals():
        A_RMSS = ax5.plot(range(0, len(std_GP_a)), y_RMS_GP_a, color = 'blue', marker ='.')
    ax5.set_ylabel('RMS Error')
    ax5.set_xlabel('Samples')
    
    # Variance vs. Samples
    ax6 = fig.add_subplot(3, 2, 4)
    ax6.set_title('Variance vs. Samples')
    if 'std_GP_a' in globals() and 'std_GP_a' in globals():
        A_VAR = ax6.plot(range(0, len(std_GP_a)), std_GP_a, color = 'blue', marker ='.')
    ax6.set_ylabel('Variance')
    ax6.set_xlabel('Samples')
       
    # Distance vs. Samples
    ax7 = fig.add_subplot(3, 2, 5)
    ax7.set_title('Distance vs. Samples')
    if 'rover_dist_a' in globals() and 'X_train_GP_a' in globals():
        A_DIS = ax7.plot(range(0, len(rover_dist_a)), rover_dist_a, color = 'blue', marker ='.')
    ax7.set_ylabel('Distance')
    ax7.set_xlabel('Samples')
    ax7.legend(['Conventional'], loc='lower right')
    
    return ax1, ax4, ax5, ax6, ax7

#%% randomly initialize location
i_0 = random.randrange(n_samples)
i_train_a = []

i_train_a.append(i_0)

i_train_full_a = list(range(0,n_samples))

# randomly sample next 10 data points with a displacement constraint of 10int
r_NN = np.sqrt(3)*grid_diff
r_con = r_NN
if func_name == 'Lunar':
    for i in range(10):
        i_sample_set_a = sample_disp_con(x_true_2,x_true_2[i_train_a[-1]],r_NN) # Conventional [nearest neighbor (within 0.25 km)]  
        i_sample_a = i_sample_set_a[random.randrange(len(i_sample_set_a))]
        i_train_a.append(int(i_sample_a))

else:
    for i in range(10):
        i_sample_set_a = sample_disp_con(x_true,x_true[i_train_a[-1]],r_NN) # nearest neighbor (within 0.25 km)
        # i_sample_set = sample_disp_con(x_true,x_true[i_train[-1]],r_con) # within 1 km
        i_sample_a = i_sample_set_a[random.randrange(len(i_sample_set_a))]
        i_train_a.append(int(i_sample_a))   

i_train_GP_a = list(set(i_train_a))

#%% hyperparameters for exploration training
sample_iter = int(n_samples/4)

#Conventional
var_iter_a = []
var_iter_local_a = []
var_iter_global_a = []
rmse_local_obs_a = []
rmse_global_obs_a = []
rmse_local_true_a = []
rmse_global_true_a = []
lengthscale_a = []

if kernel_type == 'RBF' or kernel_type == 'Matern'  or kernel_type == 'Piece_Polynomial':
# RBF Kernel
    noise = []
elif kernel_type == 'Periodic':
    period_length = []
elif kernel_type == 'RQ':
    alpha = []
elif kernel_type == 'Linear':
    variance = []
elif kernel_type == 'Polynomial':
    offset = []

#Conventional
covar_global_a = []
covar_trace_a = []
covar_totelements_a = []
covar_nonzeroelements_a = []
AIC_a = [] # Akaike Information Criterion
BIC_a = [] # Bayesian Information Criterion
f2_H_global_GP_a = []
f2_H_local_GP_a = []
std_sample_a = []


# initialize animation
fig = plt.figure(figsize=(24, 16))

#%% initiailize GP

# create B-NN and train

#Conventional
i_test_a = set(i_train_full_a) - set(i_train_a)
X_train_full_a = x_true[list(i_train_full_a),0:2]
y_train_full_a = y_true[list(i_train_full_a)]

# create GP and train
training_iter = 100

#Conventional
#train
X_train_GP_a = torch.from_numpy(X_train_full_a[i_train_GP_a,:])
y_train_GP_a = torch.from_numpy(y_train_full_a[i_train_GP_a])
X_train_GP_a = X_train_GP_a.float()
y_train_GP_a = y_train_GP_a.float()
#validate
x_test_GP_a = torch.from_numpy(x_true[list(i_test_a),0:2])
y_test_GP_a = torch.from_numpy(y_true[list(i_test_a)])
x_test_GP_a = x_test_GP_a.float()
y_test_GP_a = y_test_GP_a.float()

# train model with GPyTorch model, which optimizes hyperparameters

#Conventional
train_time_GP_a = []
GP_start_a = time.time()
likelihood_a, model_a, optimizer_a, output_a, loss_a = GPtrain(X_train_GP_a, y_train_GP_a, training_iter)
GP_end_a = time.time()
train_time_GP_a.append(GP_end_a - GP_start_a)
train_pred_a, lower_train_a, upper_train_a = GPeval(X_train_GP_a, model_a, likelihood_a)
rms_train_a = RMS(y_train_GP_a, train_pred_a.mean.detach().numpy())
test_pred_a, lower_global_a, upper_global_a = GPeval(x_test_GP_a, model_a, likelihood_a)
y_RMS_GP_a = []
y_RMS_GP_a.append(RMS(y_test_GP_a, test_pred_a.mean.detach().numpy()))
std_GP_a = []
std_GP_a.append(np.mean(np.array(upper_global_a-lower_global_a)))
f2_H_GP_a = []
GP_uncertainty_a = np.array(upper_global_a-lower_global_a).reshape(len(y_test_GP_a),1)
K_test_a = GP_uncertainty_a @ GP_uncertainty_a.transpose()
K_train_a = output_a._covar.detach().numpy()
y_test_GP_a = y_test_GP_a.numpy().reshape(len(y_test_GP_a),1)
n_train_a = []
n_train_a.append(len(X_train_GP_a))

#%% sample some other data and continue training GP with a small amount of iterations

#Conventional
for i_train_a in range(sample_iter):
    strategy = 1    

    #Conventional
    X_train_GP_a = X_train_full_a[i_train_GP_a,:]
    y_train_GP_a = y_train_full_a[i_train_GP_a]
    X_train_GP_a = torch.from_numpy(X_train_GP_a)
    y_train_GP_a = torch.from_numpy(y_train_GP_a)
    X_train_GP_a = X_train_GP_a.float()
    y_train_GP_a = y_train_GP_a.float()
    n_train_a.append(len(X_train_GP_a))
    i_test_GP_a = list(set(i_train_full_a) - set(i_train_GP_a))
    X_test_GP_a = X_train_full_a[i_test_GP_a,:]
    y_test_GP_a = y_train_full_a[i_test_GP_a]
    X_test_GP_a = torch.from_numpy(X_test_GP_a)
    y_test_GP_a = torch.from_numpy(y_test_GP_a)
    X_test_GP_a = X_test_GP_a.float()
    y_test_GP_a = y_test_GP_a.float()
    
    # train model with GPyTorch model, which optimizes hyperparameters
    GP_start_a = time.time()
    likelihood_a, model_a, optimizer_a, output_a, loss_a = GPtrain(X_train_GP_a, y_train_GP_a, training_iter)
    GP_end_a = time.time()
    train_time_GP_a.append(GP_end_a - GP_start_a)
    
    train_pred_a, lower_train_a, upper_train_a = GPeval(X_train_GP_a, model_a, likelihood_a)
    rms_train_a = RMS(y_train_GP_a, train_pred_a.mean.detach().numpy())
    
    test_pred_a, lower_global_a, upper_global_a = GPeval(X_test_GP_a, model_a, likelihood_a)
    y_pred_GP_a = test_pred_a.mean.detach().numpy()
    rms_test_a = RMS(y_test_GP_a, y_pred_GP_a)
    y_RMS_GP_a.append(RMS(y_test_GP_a, y_pred_GP_a))
    
    l_inf_a = np.max(np.abs(test_pred_a.mean.numpy()-y_test_GP_a.numpy()))
    
    # Test points are regularly spaced centered along the last index bounded by index displacement
    i_con_GP_a = sample_disp_con(x_true,x_true[i_train_GP_a[-1]],r_con)
    i_test_local_a = list(set(i_con_GP_a) - set(i_train_GP_a))
    i_test_global_a = list(set(i_train_full_a) - set(i_train_GP_a))
    x_test_local_GP_a = torch.from_numpy(x_true[i_con_GP_a, :])
    x_test_global_GP_a = torch.from_numpy(x_true[i_test_global_a, :])
    x_test_local_GP_a = x_test_local_GP_a.float()
    x_test_global_GP_a = x_test_global_GP_a.float()
    
    # Evaluate RMS for local
    observed_pred_local_a, lower_local_a, upper_local_a = GPeval(x_test_local_GP_a, model_a, likelihood_a)
    with torch.no_grad():
        f_preds_a = model_a(x_test_local_GP_a)
        y_preds_a = likelihood_a(model_a(x_test_local_GP_a))
        f_mean_a = f_preds_a.mean
        f_var_local_a = f_preds_a.variance # variance = np.diag(f_preds.lazy_covariance_matrix.numpy())
        f_covar_a = f_preds_a.covariance_matrix
    var_iter_local_a.append(max(f_var_local_a.numpy()))
    mse_local_true_a = sklearn.metrics.mean_squared_error(y_true[i_con_GP_a], observed_pred_local_a.mean.numpy())
    rmse_local_true_a.append(math.sqrt(mse_local_true_a))
    mse_local_obs_a = sklearn.metrics.mean_squared_error(y_true[i_con_GP_a], observed_pred_local_a.mean.numpy())
    rmse_local_obs_a.append(math.sqrt(mse_local_obs_a))
    # and global
    observed_pred_global_a, lower_global_a, upper_global_a = GPeval(x_test_global_GP_a, model_a, likelihood_a)
    with torch.no_grad():
        f_preds_a = model_a(x_test_global_GP_a)
        y_preds_a = likelihood_a(model_a(x_test_global_GP_a))
        f_mean_a = f_preds_a.mean
        f_var_global_a = f_preds_a.variance
        f_covar_a = f_preds_a.covariance_matrix
    var_iter_global_a.append(max(f_var_global_a.numpy()))
    mse_global_true_a = sklearn.metrics.mean_squared_error(y_true[i_test_GP_a], observed_pred_global_a.mean.numpy())
    rmse_global_true_a.append(math.sqrt(mse_global_true_a))
    mse_global_obs_a = sklearn.metrics.mean_squared_error(y_obs[i_test_GP_a], observed_pred_global_a.mean.numpy())
    rmse_global_obs_a.append(math.sqrt(mse_global_obs_a))
    
    # evaluate covariance properties
    covar_global_a.append(f_covar_a)
    covar_trace_a.append(np.trace(f_covar_a.detach().numpy()))
    covar_totelements_a.append(np.size(f_covar_a.detach().numpy()))
    covar_nonzeroelements_a.append(np.count_nonzero(f_covar_a.detach().numpy()))
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_a, model_a)
    
    # now evaluate information criteria
    # akaike information criterion
    AIC_sample_a = 2*np.log(covar_nonzeroelements_a[-1]) - 2*np.log(mse_global_true_a)
    AIC_a.append(AIC_sample_a)
    # BIC calculated from https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
    BIC_sample_a = np.size(i_train_a)*np.log(covar_nonzeroelements_a[-1]) - 2*np.log(mse_global_true_a)
    BIC_a.append(BIC_sample_a)
    
    GP_uncertainty_a = np.array(upper_global_a-lower_global_a).reshape(len(y_test_GP_a),1)
    std_GP_a.append(np.mean(GP_uncertainty_a))
    
    # and finally evaluate RKHS norm
    K_global_GP_a = output_a._covar.detach().numpy()
    y_global_GP_a = y_train_GP_a.numpy().reshape(len(y_train_GP_a),1)

    
    n_set_a = len(i_train_GP_a)
    n_sub_a = math.floor(n_set_a/2)
    i_sub_a = random.sample(range(1,n_set_a),n_sub_a)
    i_sub_a.sort()
    K_local_GP_a = K_global_GP_a[np.ix_(i_sub_a,i_sub_a)]
    y_local_GP_a = y_global_GP_a[i_sub_a]

    #%% calculate rover path distance for each exploration strategy and ability to find true min
    
    def find_distance(x_true,i_train):
        n_end = len(i_train)
        rover_distance = np.zeros(n_end)
        x_disp = np.zeros(n_end-1)
        for i in range(n_end-1):
            x_1 = x_true[i_train[i]]
            x_2 = x_true[i_train[i+1]]
            # x_disp = (x_2[0]-x_1[0])**2 + (x_2[1]-x_1[1])**2 + (x_2[2]-x_1[2])**2
            x_disp = np.sqrt(((x_2[0]-x_1[0])**2 + (x_2[1]-x_1[1])**2 )**2)
            rover_distance[i+1] = rover_distance[i] + x_disp  
        return rover_distance

    rover_distance_a = find_distance(x_true, i_train_GP_a)
    rover_dist_a = []
    rover_dist_a.append(rover_distance_a)
    rover_dist_a = np.array(rover_dist_a).flatten()

    # plot real surface and the observed measurements
    ax1, ax4, ax5, ax6, ax7 = plotBoth()
    plt.show()
    
    fig.tight_layout()
    fig.savefig(image_path+str(i_train_a)+'.png')
    fig.clear()

    #traditional query policy (unconstrained)
    GP_uncertainty_a = upper_global_a - lower_global_a
    GP_uncertainty_a = GP_uncertainty_a.numpy()
    GP_uncertainty_a = GP_uncertainty_a.astype(float)
    GP_uncertainty_a = GP_uncertainty_a.flatten()
    print(GP_uncertainty_a.shape)
    sorted_indices = np.argsort(GP_uncertainty_a)[::-1]

    for i_max_GP_a in sorted_indices:
        if i_max_GP_a not in i_train_GP_a:
            i_train_GP_a.append(i_max_GP_a)
            break  # Stop after finding the first unsampled point

#Conventional
observed_pred_global_a, lower_global_a, upper_global_a = GPeval(torch.from_numpy(x_true).float(), model_a, likelihood_a)
with torch.no_grad():
    f_preds_a = model_a(torch.from_numpy(x_true).float())
    f_mean_a = f_preds_a.mean.numpy()
i_min_GP_a = np.argmin(f_mean_a)
print('Conventional rover converged on min at '+str(x_true[i_min_GP_a]))
i_min_real = np.argmin(y_obs)
print('Conventional true min at '+str(x_true[i_min_real]))
x_1_a = x_true[i_min_GP_a]
x_2_a = x_true[i_min_real]
x_disp_a = np.sqrt((x_2_a[0]-x_1_a[0])**2 + (x_2_a[1]-x_1_a[1])**2)
print('Conventional min error is '+str(x_disp_a))

#%% calculate convergence value of RMS error and distance until convergence

def find_convergence(rmse_global, rover_distance, model, var_iter_global):
    v = rmse_global
    v0 = np.max(rmse_global)
    vf0 = rmse_global[-1]
    dv = v0 - vf0
    # band of noise allowable for 2% settling time convergence
    dv_2percent = 0.02*dv 
    
    # is there even enough data to confirm convergence?
    v_95thresh = v0 - 0.95*dv
    i_95thresh = np.where(v<v_95thresh)
    i_95thresh = np.array(i_95thresh[0],dtype=int)
    if len(i_95thresh)>=10:
        for i in range(len(i_95thresh)):
            v_con = v[i_95thresh[i]:-1]
            vf = np.mean(v_con)
            if np.all(v_con<= vf+dv_2percent) and np.all(v_con >= vf-dv_2percent):
                print(model + " convergence index is "+ str(i_95thresh[i])+" where the total samples is "+str(len(rover_distance)))
                print(model + " convergence rms error is "+str(rmse_global[i_95thresh[i]]))
                print(model + " convergence roving distance is "+ str(rover_distance[i_95thresh[i]]))
                print(model + " reduction of error is "+ str(max(rmse_global)/rmse_global[i_95thresh[i]]))
                # plotty plot plot converge wrt rms error and distance!
                ax1 = fig.add_subplot(1, 3, 1)
                # local_rms = ax1.plot(range(0,len(rmse_local)), rmse_local, color='blue', marker='.', label='local')
                global_rms = ax1.plot(range(0,len(rmse_global)), rmse_global, color='black', marker='*', label='global')
                ax1.plot([0,len(var_iter_global)], np.array([1,1])*(vf+dv_2percent), 'r--')
                ax1.plot([0,len(var_iter_global)], np.array([1,1])*(vf-dv_2percent), 'r--')
                ax1.plot(i_95thresh[i]*np.array([1,1]),[0,v0],'r--')
                ax1.set_xlabel('number of samples')
                ax1.set_ylabel('rmse')
                ax1.legend(['local','global','convergence bounds'], loc='upper right')
                ax1.set_title(model + ' rmse of learned model')
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.plot(range(len(rover_distance)),rover_distance,'k*-')
                ax2.plot(i_95thresh[i]*np.array([1,1]),[0,max(rover_distance)],'r--')
                ax2.plot([0,len(rover_distance)],rover_distance[i_95thresh[i]]*np.array([1,1]),'r--')
                ax2.set_xlabel('number of samples')
                ax2.set_ylabel('roving distance')
                ax2.set_title(model + ' rover distance during exploration')
                # fig.savefig(image_path+'convergence.png')
                i_con = i_95thresh[i]
                n_con = len(rover_distance)
                d_con = rover_distance[i_95thresh[i]]
                break
    else:
        print("not able to evaluate convergence")
        print("RMS error upon end is "+ str(rmse_global[-1]))
        print("reduction of error is "+ str(max(rmse_global)/rmse_global[-1]))
        i_con = -1
        n_con = -1
        d_con = -1
    return i_con, n_con, d_con

#Conventional        
fig = plt.figure(figsize=(18,6))
i_con_GP_a, n_con_GP_a, d_con_GP_a = find_convergence(y_RMS_GP_a, rover_distance_a, 'Conventional', var_iter_global_a)
print("Conventional: the final covariance trace is "+ str(covar_trace_a[-1]))
#information criteria
print("Conventional: the ending AIC is " + str(AIC_a[-1]))
print("Conventional: the ending BIC is " + str(BIC_a[-1]))
print('Conventional: total roving distance is '+str(rover_distance_a[-1]))

results['distance_a'].append(rover_distance_a)
results['rmse_a'].append(y_RMS_GP_a)
results['variance_a'].append(std_GP_a)

#make video!
video_name = image_path + 'GPAL_' + trial_name + '.avi'

images = []
int_list = []
for img in os.listdir(image_path):
    if img.endswith(".png"):
        images.append(img)
        s = re.findall(r'\d+', img)
        try:
            int_list.append(int(s[0]))
        except:
            print("whatever")

arg_list = np.argsort(int_list)

frame = cv2.imread(os.path.join(image_path, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(
    video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

for i in range(len(arg_list)):
    image = images[arg_list[i]]
    video.write(cv2.imread(os.path.join(image_path, image)))

with open(f'results_trial_{trial_number}.pkl', 'wb') as f:
    pickle.dump(results, f)

cv2.destroyAllWindows()
video.release()
