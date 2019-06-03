# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:48:34 2019

@author: mamartinod

Create:
    - the average dark for further frame processing.
    - the pdf of the zero-mean dark current

Save:
    - the average dark in *.npy file format
    - the pdf in a HDF5 file format

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glint_classes
import h5py
from timeit import default_timer as time

def saveFile(path, dic_data, date):
    '''
    Save the histogram of the dark
    ------------------------------
    path : string, path of the file to save. Must contain the name of the file
    dic_data : dictionary of data to save
    date : string, date of the acquisition of the data (YYYY-MM-DD)
    
    Return
    -------------------------------
    a HDF5 file in the specified path with the following tree:
        dark/cent_bins : centered bins of the histogram
        dark/histogram : non-normalized histogram (in case of interpolation needed)
    '''
    # Check if saved file exist
    if os.path.exists(path):
        opening_mode = 'w' # Overwright the whole existing file.
    else:
        opening_mode = 'a' # Create a new file at "path"
        
    with h5py.File(path, opening_mode) as f:
        f.attrs['date'] = date
        
        f.create_group('dark')
        for key in dic_data.keys():
            f.create_dataset('dark/%s'%(key), data=dic_data[key])

def getHistogram(data, bins):
    if len(data.shape) != 1:
        data = np.ravel(data)
        
    histo, bin_edges = np.histogram(np.ravel(data), bins=bins)
    return histo
           
''' Settings '''
save = False
monitor = False # Set True to map the average, variance of relative difference of set of dark current datacubes
nb_files = 2

''' Inputs '''
datafolder = 'simulation/'
root = "/mnt/96980F95980F72D3/glint_data/"
data_path = root+datafolder
dark_list = [data_path+f for f in os.listdir(data_path) if 'dark' in f][:nb_files]
date = '2019-04-30'

''' Output '''
output_path = '/mnt/96980F95980F72D3/glint/reduction/'+datafolder
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
''' Monitoring '''
''' Check the non-uniformities and defect pixels of the darks '''
if monitor:
    avg_dark = [] # Average dark current (in count) per frame
    var_dark = [] # Variance of dark current (in count) per frame
    diff_dark = [] # Different between a pixel value and the average dark current
    
''' Computing average dark '''
superDark = np.zeros((344,96))
superNbImg = 0.
list_hist = []
edge_min, edge_max = 0, 0

print('Setting common bin edges to all dark frames')
for f in dark_list[:]:
    dark0 = glint_classes.File(f)
    data = dark0.data - dark0.data.mean(axis=(1,2))[:,None,None]
    edge_max = max(edge_max, data.max())
    edge_min = min(edge_min, data.min())

del data, dark0
edge_min, edge_max = int(edge_min)-1, int(edge_max)+1

''' Define bounds of each track '''
y_ends = [33, 329] # row of top and bottom-most Track
sep = (y_ends[1] - y_ends[0])/(16-1)
channel_pos = np.around(np.arange(y_ends[0], y_ends[1]+sep, sep))


bin_hist, step = np.linspace(edge_min, edge_max, edge_max-edge_min, retstep=True)
bin_hist_cent = bin_hist[:-1] + step/2
hist_slices = []

for f in dark_list[:]:
    print("Process of : %s (%d / %d)" %(f, dark_list.index(f)+1, len(dark_list)))
    dark = glint_classes.Null(f)
   
    superDark = superDark + dark.data.sum(axis=0)
    superNbImg = superNbImg + dark.nbimg
    dark.data = dark.data - dark.data.mean(axis=(1,2))[:,None,None]
    
    spatial_axis = np.arange(dark.data.shape[0])
    dark.insulateTracks(channel_pos, sep, spatial_axis)
    
    histo_per_file = []
    for k in range(16):
        histo_per_file.append(getHistogram(np.ravel(dark.slices[:,:,k,:]), bin_hist))
    hist_slices.append(histo_per_file)
          
    if monitor:
        avg_dark.append(dark.data.mean(axis=(1,2)))
        var_dark.append(dark.data.var(axis=(1,2)))
        diff_dark.append(dark.data - dark.data.mean(axis=(1,2))[:,None,None])

if superNbImg != 0.:
    superDark /= superNbImg
    np.save(output_path+'superdark', superDark)


hist_slices = np.array(hist_slices)
super_hist = np.sum(hist_slices, axis=0)

if save:
    saveFile(output_path+'hist_dark.hdf5', {'histogram':super_hist, 'bins_edges':bin_hist}, date)

plt.figure()
plt.plot(bin_hist_cent, super_hist)
plt.grid()
plt.xlabel('Dark current (ADU)', size=38)
plt.ylabel('Count', size=38)
plt.xticks(size=36);plt.yticks(size=36)
plt.title('Histogram of the background noise', size=40)

plt.figure()
plt.semilogy(bin_hist_cent, super_hist, 'o')
plt.grid()
plt.xlabel('Dark current (ADU)', size=38)
plt.ylabel('Count', size=38)
plt.xticks(size=36);plt.yticks(size=36)
plt.title('Histogram of the background noise', size=40)

''' Inspecting non-uniformities '''
if monitor:
    avg_dark = np.array([selt for elt in avg_dark for selt in elt])
    var_dark = np.array([selt for elt in var_dark for selt in elt])
    diff_dark = np.array([selt for elt in diff_dark for selt in elt])
    
    hist_avg, bin_avg = np.histogram(avg_dark, bins=int(len(avg_dark)**0.5))
    hist_var, bin_var = np.histogram(var_dark, bins=int(len(var_dark)**0.5))
    hist_diff, bin_diff = np.histogram(diff_dark, bins=int(len(diff_dark)**0.5))
    
    bin_avg_cent = bin_avg[:-1] + np.diff(bin_avg)/2
    bin_var_cent = bin_var[:-1] + np.diff(bin_var)/2
    bin_diff_cent = bin_diff[:-1] + np.diff(bin_diff)/2
    
    plt.figure()
    plt.title('Histogram of average dark', size=40)
    plt.semilogy(bin_avg_cent, hist_avg, lw=3)
    plt.grid()
    plt.xlabel('Average dark', size=40)
    plt.ylabel('Count', size=40)
    plt.xticks(size=40);plt.yticks(size=40)
    
    plt.figure()
    plt.title('Histogram of variance of dark', size=40)
    plt.semilogy(bin_var_cent, hist_var, lw=3)
    plt.grid()
    plt.xlabel('Average dark', size=40)
    plt.ylabel('Count', size=40)
    plt.xticks(size=40);plt.yticks(size=40)
    
    plt.figure()
    plt.title('Histogram of difference to average dark', size=40)
    plt.semilogy(bin_diff_cent, hist_diff, lw=3)
    plt.grid()
    plt.xlabel('Difference to average dark', size=40)
    plt.ylabel('Count', size=40)
    plt.xticks(size=40);plt.yticks(size=40)