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

def gaussian(x, A, loc, sig):
    return A * np.exp(-(x-loc)**2/(2*sig**2))

''' Inputs '''
datafolder = '201806_alfBoo/'
root = "C:/glint/"
data_path = 'C:/glint_data/'+datafolder
data_list = [data_path+f for f in os.listdir(data_path) if not 'dark' in f]

''' Output '''
output_path = root+'reduction/'+datafolder
if not os.path.exists(output_path):
    os.makedirs(output_path)

dark = np.load(output_path+'superdark.npy')

''' Set processing configuration and load instrumental calibration data '''
nb_tracks = 16 # Number of tracks
which_tracks = np.arange(16) # Tracks to process
coeff_pos = np.load(output_path+'coeff_position_poly.npy')
coeff_width = np.load(output_path+'coeff_width_poly.npy')
position_poly = [np.poly1d(coeff_pos[i]) for i in range(nb_tracks)]
width_poly = [np.poly1d(coeff_width[i]) for i in range(nb_tracks)]
wl_to_px_coeff = np.load(root+'reduction/201806_wavecal/wl_to_px.npy')
px_to_wl_coeff = np.load(root+'reduction/201806_wavecal/px_to_wl.npy')


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
fluxes = np.zeros((1,16))
null = []
null_err = []

''' Start the data processing '''
nb_frames = 0.
for f in data_list[:]:
    print("Process of : %s (%d / %d)" %(f, data_list.index(f)+1, len(data_list)))
    img = glint_classes.Null(f)
    
    ''' Process frames '''
    img.cosmeticsFrames(dark)
    
    ''' Insulating each track '''
    img.insulateTracks(channel_pos, sep, spatial_axis)

    ''' Measurement of flux per frame, per spectral channel, per track '''
    img.getSpectralFlux(which_tracks, spectral_axis, position_poly, width_poly, debug=0)
               
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
    
    ''' Map the spectral channels between every chosen tracks before computing 
    the null depth'''
    img.matchSpectralChannels(wl_to_px_coeff, px_to_wl_coeff, which_tracks)
    
    ''' Compute null depth '''
    img.computeNullDepth()
    null.append([img.null1, img.null2, img.null3, img.null4])
    null_err.append([img.null1_err, img.null2_err, img.null3_err, img.null4_err])
    
    ''' Measure flux in photometric channels '''
    img.getPhotometry()
    
    ''' Output file'''
    img.save(output_path+os.path.basename(f)[:-4]+'.hdf5', '2019-03-19', 'amplitude')
    
    ''' For following the evolution of flux in every tracks '''
#    img.getTotalFlux()
#    fluxes = np.vstack((fluxes, img.fluxes))
#    nb_frames += img.nbimg
    
#amplitude = np.array(amplitude)
#amplitude_fit = np.array(amplitude_fit)
#integ_raw = np.array(integ_raw)
#integ_model = np.array(integ_model)
#integ_windowed = np.array(integ_windowed)
#residuals_reg = np.array(residuals_reg)
#residuals_fit = np.array(residuals_fit)
#cov = np.array(cov)
#bg_noise = np.array(bg_noise)
#null = np.array(null)
#null_err = np.array(null_err)


#''' Miscellaneous ''' 
#plt.figure()
#plt.imshow(img.data[0], interpolation='none', aspect='auto')
#plt.colorbar()

#for k in range(1):
#    plt.figure()
#    for i in range(16):
#        plt.subplot(4,4,i+1)
#        plt.title('Track '+str(i+1))
#        plt.plot(integ_raw[0,k,:,i], 'o')
#        plt.plot(integ_windowed[0,k,i], 'd')
#        plt.plot(integ_model[0,k,i], '+')
#        plt.grid()

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
##    plt.figure()
#    for i in range(16):
#        plt.figure()
##        plt.subplot(4,4,i+1)
#        plt.title('Track '+str(i+1))
#        plt.plot(amplitude[k,0,i], 'o')
#        try:
#            plt.plot(amplitude_fit[k,0,i], '+')
#        except IndexError:
#            pass
#        plt.ylim(-50, 800)
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

#for i in range(2):
#    plt.figure()
#    plt.subplot(221)
#    plt.errorbar(np.arange(94), null[0][0][i], yerr=null_err[0][0], fmt='o')
#    plt.grid()
#    plt.ylim(-5,5)
#    plt.subplot(222)
#    plt.errorbar(np.arange(94), null[0][1][i], yerr=null_err[0][1], fmt='d')
#    plt.grid()
#    plt.ylim(-5,5)
#    plt.subplot(223)
#    plt.errorbar(np.arange(94), null[0][2][i], yerr=null_err[0][3], fmt='s')
#    plt.grid()
#    plt.ylim(-5,5)
#    plt.subplot(224)
#    plt.errorbar(np.arange(94), null[0][3][i], yerr=null_err[0][3], fmt='+')
#    plt.grid()
#    plt.ylim(-5,5)