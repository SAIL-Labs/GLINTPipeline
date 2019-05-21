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

''' Settings '''
no_noise = False
nb_img = None
debug = False
save = False
nb_files = 100
idx_edges = (40,45)
idx_edges2 = (50,55)

''' Inputs '''
datafolder = '201806_alfBoo/'
root = "/mnt/96980F95980F72D3/glint/"
data_path = '/mnt/96980F95980F72D3/glint_data/'+datafolder
data_list = [data_path+f for f in os.listdir(data_path) if not 'dark' in f][:nb_files]

''' Output '''
output_path = root+'reduction/'+datafolder
if not os.path.exists(output_path):
    os.makedirs(output_path)

dark = np.load(output_path+'superdark.npy')

''' Set processing configuration and load instrumental calibration data '''
calibration_path = 'simulation'
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
fluxes = np.zeros((1,16))
fluxes2 = np.zeros((1,16))
null = []
null_err = []

''' Start the data processing '''
nb_frames = 0
for f in data_list:
    print("Process of : %s (%d / %d)" %(f, data_list.index(f)+1, len(data_list)))
    img = glint_classes.Null(f, nbimg=nb_img, transpose=True)
    
    ''' Process frames '''
    img.cosmeticsFrames(dark, no_noise)
    
    ''' Insulating each track '''
    img.insulateTracks(channel_pos, sep, spatial_axis)

    ''' Measurement of flux per frame, per spectral channel, per track '''
    img.getSpectralFlux(which_tracks, spectral_axis, position_poly, width_poly, debug=debug)
    
    ''' Map the spectral channels between every chosen tracks before computing 
    the null depth'''
    img.matchSpectralChannels(wl_to_px_coeff, px_to_wl_coeff, which_tracks)
    
    ''' Compute null depth '''
    img.computeNullDepth()
    null_depths = np.array([img.null1, img.null2, img.null3, img.null4])
    null_depths_err = np.array([img.null1_err, img.null2_err, img.null3_err, img.null4_err])
    
    ''' Measure flux in photometric channels '''
    img.getPhotometry()
    
    ''' Output file'''
    if save:
        img.save(output_path+os.path.basename(f)[:-4]+'.hdf5', '2019-04-30', 'amplitude')

    if debug:
        amplitude.append(img.amplitude)
        integ_model.append(img.integ_model)
        integ_windowed.append(img.integ_windowed)
        residuals_reg.append(img.residuals_reg)
        bg_noise.append(img.bg_std)
        integ_raw.append(img.raw)
        null.append(null_depths)
        null_err.append(null_depths_err)
        
        try: # if debug mode TRUE in getSpectralFlux method
            residuals_fit.append(img.residuals_fit)
            cov.append(img.cov)
            amplitude_fit.append(img.amplitude_fit)
        except AttributeError:
            pass
    
    ''' For following the evolution of flux in every tracks '''
    img.getTotalFlux(idx_edges)
    fluxes = np.vstack((fluxes, img.fluxes))
    img.getTotalFlux(idx_edges2)
    fluxes2 = np.vstack((fluxes, img.fluxes))
    nb_frames += img.nbimg
    
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
fluxes = fluxes[1:]
fluxes2 = fluxes2[1:]

''' Miscellaneous ''' 
fluxes = np.array([fluxes[:,15], fluxes[:,13], fluxes[:,2], fluxes[:,0]])/nb_frames
fluxes -= np.mean(fluxes, axis=-1)[:,None]

flux_histo = [np.histogram(elt, bins=int(np.size(elt)**0.5)) for elt in fluxes]
flux_edges = np.array([elt[1] for elt in flux_histo])
flux_histo = np.array([elt[0] for elt in flux_histo])
flux_histo = flux_histo / np.sum(flux_histo, axis=-1)[:,None]

fluxes2 = np.array([fluxes2[:,15], fluxes2[:,13], fluxes2[:,2], fluxes2[:,0]])/nb_frames
fluxes2 -= np.mean(fluxes2, axis=-1)[:,None]

flux2_histo = [np.histogram(elt, bins=int(np.size(elt)**0.5)) for elt in fluxes2]
flux2_edges = np.array([elt[1] for elt in flux2_histo])
flux2_histo = np.array([elt[0] for elt in flux2_histo])
flux2_histo = flux2_histo / np.sum(flux2_histo, axis=-1)[:,None]

fluxes_dk = np.load('fluxes_dark.npy')
fluxes_dk_histo = [np.histogram(elt, bins=int(np.size(elt)**0.5)) for elt in fluxes_dk]
fluxes_dk_edges = np.array([elt[1] for elt in fluxes_dk_histo])
fluxes_dk_histo = np.array([elt[0] for elt in fluxes_dk_histo])
fluxes_dk_histo = fluxes_dk_histo / np.sum(fluxes_dk_histo, axis=-1)[:,None]

plt.figure()
plt.suptitle('Histogram of fluxes')
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title('P%s'%(i+1))
    plt.plot(flux_edges[i,:-1], flux_histo[i], 'o')
    plt.plot(flux2_edges[i,:-1], flux2_histo[i], 'o')
    plt.plot(fluxes_dk_edges[i,:-1], fluxes_dk_histo[i], 'o')    
    plt.grid()


plt.figure()
plt.imshow(img.data.mean(axis=0), interpolation='none', aspect='auto')
plt.colorbar()
plt.title('Avg frame')

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
plt.figure()
plt.suptitle('Amplitude')
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.title('Track '+str(i+1))
    plt.plot(img.wl_scale[i], amplitude[0,i,:], 'o')
    plt.ylim(-50, 800)
    plt.grid()
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

for i in range(1):
    plt.figure()
    plt.suptitle('Null')
    plt.subplot(221)
    plt.errorbar(img.wl_scale[0], null[0][i], yerr=null_err[0][i], fmt='o')
    plt.grid()
    plt.ylim(-5,5)
    plt.subplot(222)
    plt.errorbar(img.wl_scale[0], null[1][i], yerr=null_err[1][i], fmt='d')
    plt.grid()
    plt.ylim(-5,5)
    plt.subplot(223)
    plt.errorbar(img.wl_scale[0], null[2][i], yerr=null_err[3][i], fmt='s')
    plt.grid()
    plt.ylim(-5,5)
    plt.subplot(224)
    plt.errorbar(img.wl_scale[0], null[3][i], yerr=null_err[3][i], fmt='+')
    plt.grid()
    plt.ylim(-5,5)