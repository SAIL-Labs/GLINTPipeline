# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:47:53 2019

@author: Marc-Antoine Martinod

Measure null depths and flux in each spectral channels.
Output file : HDF5 format containing all relevant data for model fitting
with NSC method (Hanot et al. 2011)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glint_classes
import warnings
warnings.filterwarnings(action="ignore", category=np.VisibleDeprecationWarning)
from timeit import default_timer as time
import h5py
from scipy.optimize import curve_fit

def gaussian(x, A, loc, sig):
    return A * np.exp(-(x-loc)**2/(2*sig**2))

''' Settings '''
no_noise = False
nb_img = (None, None)
debug = False
save = True
nb_files = None

''' Inputs '''
datafolder = '201907_Data/'
calibration_path = '201806_wavecal'
root = "/mnt/96980F95980F72D3/glint/"
data_path = '/mnt/96980F95980F72D3/glint_data/'+datafolder
data_list = [data_path+f for f in os.listdir(data_path) if 'p4' in f][:nb_files]

''' Output '''
output_path = root+'reduction/'+datafolder
if not os.path.exists(output_path):
    os.makedirs(output_path)

dark = np.load(output_path+'superdark.npy')
dark_per_channel = np.load(output_path+'superdarkchannel.npy')
if no_noise:
    dark_per_channel[:] = 0.

''' Set processing configuration and load instrumental calibration data '''
nb_tracks = 16 # Number of tracks
which_tracks = np.arange(16) # Tracks to process
coeff_pos = np.load(output_path+'coeff_position_poly.npy')
coeff_width = np.load(output_path+'coeff_width_poly.npy')
position_poly = [np.poly1d(coeff_pos[i]) for i in range(nb_tracks)]
width_poly = [np.poly1d(coeff_width[i]) for i in range(nb_tracks)]
wl_to_px_coeff = np.load(root+'reduction/'+calibration_path+'/wl_to_px.npy')
px_to_wl_coeff = np.load(root+'reduction/'+calibration_path+'/px_to_wl.npy')


spatial_axis = np.arange(dark.shape[0])
spectral_axis = np.arange(dark.shape[1])

''' Define bounds of each track '''
y_ends = [33, 329] # row of top and bottom-most Track
sep = (y_ends[1] - y_ends[0])/(nb_tracks-1)
channel_pos = np.around(np.arange(y_ends[0], y_ends[1]+sep, sep))

''' Output lists for different stages of the processing.
Include data from all processed files '''
amplitude = []
amplitude_fit = []
integ_raw = []
integ_model = []
integ_windowed = []
residuals_reg = []
residuals_fit = []
cov = []
bg_noise = []
fluxes = np.zeros((1,4))
null = []
null_err = []
p1 = []
p2 = []
p3 = []
p4 = []

''' Start the data processing '''
nb_frames = 0
for f in data_list:
    start = time()
    print("Process of : %s (%d / %d)" %(f, data_list.index(f)+1, len(data_list)))
    img = glint_classes.Null(f, nbimg=nb_img)
    
    ''' Process frames '''
    img.cosmeticsFrames(np.zeros(dark.shape), no_noise)
    
    ''' Insulating each track '''
    print('Getting channels')
    img.insulateTracks(channel_pos, sep, spatial_axis, dark=dark_per_channel)
    
    ''' Map the spectral channels between every chosen tracks before computing 
    the null depth'''
    img.matchSpectralChannels(wl_to_px_coeff, px_to_wl_coeff, which_tracks)
    
    ''' Measurement of flux per frame, per spectral channel, per track '''
    list_channels = np.arange(16) #[1,3,4,5,6,7,8,9,10,11,12,14]
    img.getSpectralFlux(list_channels, spectral_axis, position_poly, width_poly, debug=debug)
#    img.getPhotoFluctuations(spectral_axis, position_poly, width_poly)
    
    ''' Compute null depth '''
    print('Computing null depths')
    img.computeNullDepth()
    null_depths = np.array([img.null1, img.null2, img.null3, img.null4, img.null5, img.null6])
    null_depths_err = np.array([img.null1_err, img.null2_err, img.null3_err, img.null4_err, img.null5_err, img.null6_err])
    
    ''' Measure flux in photometric channels '''
    img.getIntensities()
    p1.append(img.p1)
    p2.append(img.p2)
    p3.append(img.p3)
    p4.append(img.p4)
    
    ''' Output file'''
    if save:
        img.save(output_path+os.path.basename(f)[:-4]+'.hdf5', '2019-04-30', 'amplitude')

    null.append(np.transpose(null_depths, axes=(1,0,2)))
    null_err.append(np.transpose(null_depths_err, axes=(1,0,2)))
    if debug:
        amplitude.append(img.amplitude)
        integ_model.append(img.integ_model)
        integ_windowed.append(img.integ_windowed)
        residuals_reg.append(img.residuals_reg)
        bg_noise.append(img.bg_std)
        integ_raw.append(img.raw)

        
        try: # if debug mode TRUE in getSpectralFlux method
            residuals_fit.append(img.residuals_fit)
            cov.append(img.cov)
            amplitude_fit.append(img.amplitude_fit)
        except AttributeError:
            pass
    
    ''' For following the evolution of flux in every tracks '''
    print('Getting total flux')
    img.getTotalFlux()
    fluxes = np.vstack((fluxes, img.fluxes.T))
    nb_frames += img.nbimg
    stop = time()
    print('Last: %.3f'%(stop-start))
    
amplitude = np.array([selt for elt in amplitude for selt in elt])
amplitude = np.array([[elt[i][img.px_scale[i]] for i in range(16)] for elt in amplitude])
amplitude_fit = np.array([selt for elt in amplitude_fit for selt in elt])
amplitude_fit = np.array([[elt[i][img.px_scale[i]] for i in range(16)] for elt in amplitude_fit])
integ_raw = np.array([selt for elt in integ_raw for selt in elt])
integ_raw = np.array([[elt[i][img.px_scale[i]] for i in range(16)] for elt in integ_raw])
integ_model = np.array([selt for elt in integ_model for selt in elt])
integ_model = np.array([[elt[i][img.px_scale[i]] for i in range(16)] for elt in integ_model])
integ_windowed = np.array([selt for elt in integ_windowed for selt in elt])
integ_windowed = np.array([[elt[i][img.px_scale[i]] for i in range(16)] for elt in integ_windowed])
residuals_reg = np.array([selt for elt in residuals_reg for selt in elt])
residuals_reg = np.array([[elt[i][img.px_scale[i]] for i in range(16)] for elt in residuals_reg])
residuals_fit = np.array([selt for elt in residuals_fit for selt in elt])
residuals_fit = np.array([[elt[i][img.px_scale[i]] for i in range(16)] for elt in residuals_fit])
cov = np.array([selt for elt in cov for selt in elt])
cov = np.array([[elt[i][img.px_scale[i]] for i in range(16)] for elt in cov])
bg_noise = np.array([selt for elt in bg_noise for selt in elt])
null = np.array([selt for elt in null for selt in elt])
null_err = np.array([selt for elt in null_err for selt in elt])
p1 = np.array([selt for elt in p1 for selt in elt])
p2 = np.array([selt for elt in p2 for selt in elt])
p3 = np.array([selt for elt in p3 for selt in elt])
p4 = np.array([selt for elt in p4 for selt in elt])
fluxes = fluxes[1:]

''' Miscellaneous '''
photometries = [p1[:,56], p2[:,56], p3[:,56], p4[:,56]]
photometries_label = ['p1' ,'p2', 'p3', 'p4']
for k in range(len(photometries)):
    photo = photometries[k]
    histo, bin_edges = np.histogram(photo, int(photo.size**0.5))
    binning = bin_edges[:-1] + np.diff(bin_edges)/2
    histo = histo / np.sum(histo)
#    dk = h5py.File('/mnt/96980F95980F72D3/glint/reduction/201806_alfBoo/hist_dark_slices.hdf5') 
#    dk = np.array(dk[plop_label[k]]).T
    popt, pcov = curve_fit(gaussian, binning, histo, p0=[max(histo), photo.mean(), photo.std()])
    y = gaussian(binning, *popt)
#    popt2, pcov2 = curve_fit(gaussian, dk[1], dk[0], p0=[max(dk[0]), 0, 90])
#    y2 = gaussian(dk[1], *popt2)
    
    fig = plt.figure(figsize=(19.20, 10.80))
    ax = fig.add_subplot(111)
#    plt.plot(dk[1], dk[0]/dk[0].max(), 'o', label='Dark current')
    plt.plot(binning, histo/histo.max(), 'o-', label='P%s'%(k+1))
    plt.plot(binning, y/y.max(), '--', lw=4, label='Fit of P%s'%(k+1))
#    plt.plot(dk[1], y2/y2.max(), '--', lw=4, label='Fit of dark')
    plt.grid()
    plt.legend(loc='best', fontsize=36)
    plt.xticks(size=36);plt.yticks(size=36)
    plt.xlabel('Bins', size=40)
    plt.ylabel('Counts (normalised)', size=40)
    txt = r'$\mu_{p%s} = %.3f$'%(k+1, popt[1]) + '\n' + r'$\sigma_{p%s} = %.3f$'%(k+1,popt[2])
    plt.text(0.05,0.3, txt, va='center', fontsize=30, transform = ax.transAxes, bbox=dict(boxstyle="square", facecolor='white'))

for k in range(len(photometries)):
    plt.figure(figsize=(19.20, 10.80))
    plt.plot(np.arange(photometries[k].size)[::100], photometries[k][::100])
    plt.grid()
    plt.xlabel('Frame/100', size=30)
    plt.ylabel('Fitted amplitude', size=30)
    plt.xticks(size=30);plt.yticks(size=30)
    
if debug:
    for k in range(1):
        plt.figure()
        plt.suptitle('Amplitude respect to different methods')
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.title('Track '+str(i+1))
            plt.plot(amplitude[k,i,:], '^')
            plt.plot(integ_raw[k,i,:], 'o')
            plt.plot(integ_windowed[k,i,:], 'd')
            plt.plot(integ_model[k,i,:], '+')
            plt.grid()

    plt.figure()
    plt.suptitle('Amplitude')
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.title('Track '+str(i+1))
        plt.plot(img.wl_scale[i], amplitude[0,i,:], 'o')
        plt.ylim(-50, 800)
        plt.grid()

for i in range(1):
    plt.figure()
    plt.suptitle('Null')
    plt.subplot(321)
    plt.plot(img.wl_scale[0], null[i][0], '-')
    plt.grid()
    plt.ylim(-2,3)
    plt.subplot(322)
    plt.plot(img.wl_scale[0], null[i][1], '-')
    plt.grid()
    plt.ylim(-2,3)
    plt.subplot(323)
    plt.plot(img.wl_scale[0], null[i][2], '-')
    plt.grid()
    plt.ylim(-2,3)
    plt.subplot(324)
    plt.plot(img.wl_scale[0], null[i][3], '-')
    plt.grid()
    plt.ylim(-2,3)
    plt.subplot(325)
    plt.plot(img.wl_scale[0], null[i][4], '-')
    plt.grid()
    plt.ylim(-2,3)
    plt.subplot(326)
    plt.plot(img.wl_scale[0], null[i][5], '-')
    plt.grid()
    plt.ylim(-2,3)
    
#amplitude0 = np.load('amplitude.npy')
#integ_raw0 = np.load('integ_raw.npy')
#integ_model0 = np.load('integ_model.npy')
#integ_windowed0 = np.load('integ_windowed.npy')
#
#for k in range(1):
#    plt.figure()
#    for i in range(16):
#        plt.subplot(4,4,i+1)
#        plt.title('Track '+str(i+1))
#        plt.plot(integ_raw[0,k,:,i], 'o')
#        plt.plot(integ_raw0[0,k,:,i], 'd')
#        plt.grid()

#for k in range(1):
#    plt.figure()
#    for i in range(16):
#        plt.subplot(4,4,i+1)
#        plt.title('Track '+str(i+1))
#        plt.plot(integ_model[0,k,:,i], 'o')
#        plt.plot(integ_model0[0,k,:,i], 'd')
#        plt.grid()
#for k in range(1):
#    plt.figure()
#    for i in range(16):
#        plt.subplot(4,4,i+1)
#        plt.title('Track '+str(i+1))
#        plt.plot(integ_windowed[0,k,:,i], 'o')
#        plt.plot(integ_windowed0[0,k,:,i], 'd')
#        plt.grid()

#results = []        
#for i in range(2,3):
#    y = img.slices[0,i,0]
#    x = img.slices_axes[0]
#    A = np.vstack((x, np.zeros(x.shape)))
#    A = np.transpose(A)
#    p = np.linalg.lstsq(A, y)[0][0]
#    results.append(p)

#for k in range(1):
#    plt.figure()
#    for i in range(16):
#        plt.subplot(4,4,i+1)
#        plt.title('Track '+str(i+1))
#        plt.plot(cov[0,k,i]**0.5, 'o')
#        plt.ylabel('Std of curve_fit')
#        plt.grid()

