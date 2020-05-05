#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script determines the position and width of the outputs per spectral channel, 
assuming a Gaussian profile.
This script requires the library :doc:`glint_classes` to work.

The inputs are **averaged dark** and **datacube with no fringe**.
To get them, either you can try to be out of the coherent envelop for all baselines
or having a large time-varying phase.
In the last case, the average frame of the datacube blurs the fringe.

The outputs products are:
        * The polynomial coefficients determining the location of 16 outputs respect to spectral axis
        * The polynomial coefficients determining the width of 16 outputs respect to spectral axis
        
The outputs are saved into numpy-format file (.npy).

This script is used in 3 steps.

First step: simply change the value of the variables in the **Settings** section:
    * **save**: boolean, ``True`` for saving products and monitoring data, ``False`` otherwise
    * **monitoring**: boolean, ``True`` for displaying the results of the model fitting and the residuals for both location and width for all outputs
    
Second step: change the value of the variables in the **Inputs** and **Outputs** sections:
    * **datafolder**: folder containing the datacube to use.
    * **root**: path to **datafolder**.
    * **data_list**: list of files in **datafolder** to open.
    * **output_path**: path to the folder where the products are saved.
    
Third step: start the script and let it run.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glint_classes
from scipy.optimize import curve_fit
#import skimage.measure as sk

def gaussian(x, A, x0, sig):
    """
    Computes a Gaussian curve
    
    :Parameters:
        
        **x**: values where the curve is estimated.
        
        **a**: amplitude of the Gaussian.
        
        **x0**: location of the Gaussian.
        
        **sig**: scale of the Gaussian.
        
    :Returns:
        
        Gaussian curve.
    """    
    return A * np.exp(-(x-x0)**2/(2*sig**2))# + offset

def polynom(x, *args):
    """
    Computes a polynom from the coefficients in ``*args`` and returns the evaluation for ``x``.
    
    :Parameters:
        
        **x**: values where the curve is estimated.
        
        **a**: amplitude of the Gaussian.
        
        **x0**: location of the Gaussian.
        
        **sig**: scale of the Gaussian.
        
    :Returns:
        
        Gaussian curve.
    """    
    p = np.poly1d([*args])
    return p(x)

if __name__ == '__main__':
    # =============================================================================
    # Get the shape (position and width) of all tracks
    # =============================================================================
    ''' Settings '''
    save = False
    
    print("Getting the shape (position and width) of all tracks")
    ''' Inputs '''
    datafolder = '20200201/AlfBoo/'
    root = "/mnt/96980F95980F72D3/glint/"
    data_path = '/mnt/96980F95980F72D3/glint_data/'+datafolder
    data_list = [data_path+f for f in os.listdir(data_path) if 'AlfBoo' in f in f]
    if len(data_list) == 0:
        raise IndexError('Data list is empty')
        
    ''' Output '''
    output_path = root+'GLINTprocessed/'+datafolder
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    ''' Remove dark from the frames and average them to increase SNR '''
    dark = np.load(output_path+'superdark.npy')
    dark_per_channel = np.load(output_path+'superdarkchannel.npy')
    super_img = np.zeros(dark.shape)
    superNbImg = 0.
     
    spatial_axis = np.arange(dark.shape[0])
    spectral_axis = np.arange(dark.shape[1])
    slices = np.zeros_like(dark_per_channel)
    
    ''' Define bounds of each track '''
    y_ends = [33, 329] # row of top and bottom-most Track
    nb_tracks = 16
    sep = (y_ends[1] - y_ends[0])/(nb_tracks-1)
    channel_pos = np.around(np.arange(y_ends[0], y_ends[1]+sep, sep))
    
    for f in data_list[:]:
        img = glint_classes.Null(f)
        img.cosmeticsFrames(np.zeros(dark.shape))
        img.getChannels(channel_pos, sep, spatial_axis, dark=dark_per_channel)
        super_img = super_img + img.data.sum(axis=0)
        slices = slices + np.sum(img.slices, axis=0)
        superNbImg = superNbImg + img.nbimg
    
    slices = slices / superNbImg
    super_img = super_img / superNbImg
   
    plt.figure(0)
    plt.clf()
    plt.imshow(super_img-dark, interpolation='none', vmin=super_img.min(), vmax=5000)
    plt.colorbar()
    
    ''' Fit a gaussian on every track and spectral channel
        to get their positions, widths and amplitude '''
    
    img = glint_classes.Null(data=None, nbimg=(0,1))
    img.cosmeticsFrames(np.zeros(dark.shape))
    img.getChannels(channel_pos, sep, spatial_axis)
    
    img.slices = slices
    slices_axes = img.slices_axes
    params = []
    cov = []
    residuals = []
    
    for i in range(slices.shape[0]): # Loop over columns of pixel
        for j in range(slices.shape[1]): # Loop over tracks
            p_init2 = np.array([100, channel_pos[j], 1.])
            try:
                popt, pcov = curve_fit(gaussian, slices_axes[j], slices[i,j], p0=p_init2)
                params.append(popt)
                cov.append(np.diag(pcov))
                residuals.append(slices[i,j] - gaussian(slices_axes[j], *popt))
                if i == 50 and j == 0:
                    plt.figure();plt.plot(slices_axes[j], slices[i,j]);plt.plot(slices_axes[j], gaussian(slices_axes[j], *popt));plt.title('plop')
            except RuntimeError:
                params.append(np.zeros(p_init2.shape))
                cov.append(np.zeros(p_init2.shape))
                residuals.append(np.zeros(slices_axes[j].shape))
                print("Error fit at spectral channel "+str(i)+" of track "+str(j))
    
    params = np.array(params).reshape((dark.shape[1],nb_tracks,-1))
    cov = np.array(cov).reshape((dark.shape[1],nb_tracks,-1))
    residuals = np.array(residuals).reshape((dark.shape[1],nb_tracks,-1))
    
    params[:,:,2] = abs(params[:,:,2]) # convert negative widths into positive ones
    
    ''' Fit functions on positions and widths vector to extrapolate on any point 
        along the spectral axis '''
        
    bounds = [33, 96] # Cut the noisy part of the spectral
    pos = params[bounds[0]:bounds[1],:,1]
    wi = params[bounds[0]:bounds[1],:,2]
    
    weight_pos = np.ones(pos.shape)
    weight_width = np.ones(wi.shape)
    
    # weight_pos[80-bounds[0]:,1] = 1.e-36
    # weight_pos[80-bounds[0]:,2] = 1.e-36
    # weight_pos[85-bounds[0]:,7] = 1.e-36
    # weight_pos[84-bounds[0]:,13] = 1.e-36
    # weight_pos[78-bounds[0],15] = 1.e-36
    
    # weight_width[36-bounds[0]:80-bounds[0],:] = 1000
    # weight_width[36-bounds[0]:,3] = 1000
    # weight_width[36-bounds[0]:,4] = 1000
    # weight_width[36-bounds[0]:40-bounds[0],5] = 1
    # weight_width[36-bounds[0]:,6] = 1000
    # weight_width[36-bounds[0]:,7] = 1000
    # weight_width[36-bounds[0]:,8] = 1000
    # weight_width[36-bounds[0]:,9] = 1000
    # weight_width[36-bounds[0]:,10] = 1000
    # weight_width[36-bounds[0]:40-bounds[0],11] = 1
    # weight_width[36-bounds[0]:40-bounds[0],12] = 1
    # weight_width[80-bounds[0]:89-bounds[0],12] = 1000
    # weight_width[77-bounds[0],12] = 1
    # weight_width[36-bounds[0]:40-bounds[0],13] = 1
    # weight_width[36-bounds[0]:40-bounds[0],14] = 1
    # weight_width[36-bounds[0]:40-bounds[0],15] = 1
    # weight_width[78-bounds[0],15] = 1

    
    
    coeff_position_poly = np.array([np.polyfit(spectral_axis[bounds[0]:bounds[1]], pos[:,i], deg=4, w=weight_pos[:,i]) for i in range(nb_tracks)])
    coeff_width_poly = np.array([np.polyfit(spectral_axis[bounds[0]:bounds[1]], wi[:,i], deg=4, w=weight_width[:,i]) for i in range(nb_tracks)])
    position_poly = [np.poly1d(coeff_position_poly[i]) for i in range(nb_tracks)]
    width_poly = [np.poly1d(coeff_width_poly[i]) for i in range(nb_tracks)]
    
    for i in range(nb_tracks)[:]:
        fitted_pos = position_poly[i](spectral_axis[:])
        plt.figure()
        ax = plt.gca()
        color = next(ax._get_lines.prop_cycler)['color']
        marker, marker2 = '+-', '.'
        plt.subplot(221)
        plt.plot(spectral_axis[:], params[:,i,1], marker, lw=3, label='Position')
        plt.plot(spectral_axis[:], fitted_pos, marker2, label='Fit')
        plt.grid()
        plt.legend(loc='best', ncol=4)
        plt.ylim(fitted_pos.min()*0.999, fitted_pos.max()*1.001)
        plt.title('x0 / Track '+str(i+1))
        plt.ylabel('x0')
        plt.xlabel('Wavelength')
        plt.subplot(223)
        plt.plot(spectral_axis[:], (params[:,i,1]-position_poly[i](spectral_axis[:]))/params[:,i,1]*100)
        plt.grid()
        plt.xlabel('Wavelength')
        plt.ylabel('Residual (%)')
        plt.ylim(-1, 1)
        plt.subplot(222)
        plt.plot(spectral_axis, params[:,i,2], marker, lw=3, label='Width')
        plt.plot(spectral_axis, width_poly[i](spectral_axis), marker2, label='Fit')
        plt.grid()
        plt.legend(loc='best', ncol=4)
        plt.title('sig / Track'+str(i+1))
        plt.ylabel('Sig')
        plt.xlabel('Wavelength')
        plt.xlim(0)
        plt.ylim(0,2)
        plt.subplot(224)
        plt.plot(spectral_axis, (params[:,i,2]-width_poly[i](spectral_axis))/params[:,i,2]*100)
        plt.grid()
        plt.xlabel('Wavelength')
        plt.ylabel('Residual (%)')
        plt.ylim(-10,10)
    
    if save:
        np.save(output_path+'coeff_position_poly', coeff_position_poly)
        np.save(output_path+'coeff_width_poly', coeff_width_poly)
