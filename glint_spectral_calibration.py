#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script makes the spectral calibration of the 16 outputs.
This script requires the library :doc:`glint_classes` to work.

The inputs are **averaged dark**, **datacubes with spectral bands** and the **data from the geometric calibration** (cf :doc:`glint_geometric_calibration`).
It is assumed one data file contains only one spectral band with its wavelength in the name.
The script successively loads the data files related to one wavelength and extracts the 16 outputs.
For each of them, we assume the spectral band is shaped as a Gaussian.
A model fitting with this shape is performed to get the position and width (in pixel).
Once all files of all wavelength are processed, a polynomial fit is performed to map the wavelength to the column of pixels
for each output.

The outputs products are:
        * The polynomial coefficients mapping the wavelength respect to the column of pixels.
        * The polynomial coefficients mapping he column of pixels respect to the wavelengths.
        * The spectral psf, giving the spectral resolution.
        
The outputs are saved into numpy-format file (.npy).

This script is used in 3 steps.

First step: simply change the value of the variables in the section **Settings**:
    * **save**: boolean, ``True`` for saving products and monitoring data, ``False`` otherwise
    
Second step: change the value of the variables in the sections **Inputs**, **Outputs** and **Iterate on wavelength**:
    * **datafolder**: folder containing the datacube to use.
    * **data_list**: list of files in **datafolder** to open.
    * **output_path**: path to the folder where the products are saved.
    * **calibration_path**: path to the calibration files used to process the file (location, width of the outputs, etc.).
    * **wavelength**: list, list of wavelengths used to acquire the data
    
Third step: start the script and let it run.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glint_classes
from scipy.optimize import curve_fit
#import skimage.measure as sk

def gaussian(x, A, x0, sig, offset):
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
    return A * np.exp(-(x-x0)**2/(2*sig**2)) + offset

if __name__ == '__main__':
    ''' Settings '''
    save = False
    
    ''' Inputs '''
    print("-----------------------------\nSpectral calibration")
    datafolder = 'data202009/20200906/wavecal/'
    data_path = '//tintagel.physics.usyd.edu.au/snert/GLINTData/'+datafolder
    output_path = '//tintagel.physics.usyd.edu.au/snert/GLINTprocessed/'+datafolder

    ''' Output '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    ''' Iterate on wavelength '''
    wavelength = [1400, 1450, 1500, 1550, 1600][:]
    data_list0 = [[data_path+f for f in os.listdir(data_path) if '1400_' in f],
                 [data_path+f for f in os.listdir(data_path) if '1450_' in f],
                 [data_path+f for f in os.listdir(data_path) if '1500_' in f],
                 [data_path+f for f in os.listdir(data_path) if '1550_' in f],
                 [data_path+f for f in os.listdir(data_path) if '1600_' in f]][:]
    
    ''' Remove dark from the frames and average them to increase SNR '''
    dark = np.load(output_path+'superdark.npy')
    dark_per_channel = np.load(output_path+'superdarkchannel.npy')
    super_img = np.zeros(dark.shape)
    superNbImg = 0.
    
    ''' Run '''
    spatial_axis = np.arange(dark.shape[0])
    spectral_axis = np.arange(dark.shape[1])
    slices = np.zeros_like(dark_per_channel)
    
    ''' Define bounds of each track '''
    y_ends = [33, 329] # row of top and bottom-most Track
    nb_tracks = 16
    sep = (y_ends[1] - y_ends[0])/(nb_tracks-1)
    channel_pos = np.around(np.arange(y_ends[0], y_ends[1]+sep, sep))
    
    calib_pos = []
    for data_list in data_list0:
        print('Processing wavelength %s'%(wavelength[data_list0.index(data_list)]))
        print('Averaging frames')
        for f in data_list[:]:
            img = glint_classes.Null(f)
            img.cosmeticsFrames(np.zeros(dark.shape))
            img.getChannels(channel_pos, sep, spatial_axis, dark=dark_per_channel)
            super_img = super_img + img.data.sum(axis=0)
            slices = slices + np.sum(img.slices, axis=0)
            superNbImg = superNbImg + img.nbimg
                
        slices = slices / superNbImg
        super_img = super_img / superNbImg
    
        labels = ['P4', 'N3', 'P3', 'N2', 'AN4', 'N5', 'N4', 'AN5', 'N6', 'AN1', 'AN6', 'N1', 'AN2', 'P2', 'AN3', 'P1']
    
        ''' Fit a gaussian on every track and spectral channel
            to get their positions, widths and amplitude '''
        print('Determine position and width of the outputs')
        img = glint_classes.Null(data=None, nbimg=(0,1))
        img.cosmeticsFrames(np.zeros(dark.shape))
        img.getChannels(channel_pos, sep, spatial_axis)
        
        img.slices = slices
        slices_axes = img.slices_axes
        tracks = img.slices[:,:,10-4:10+5].mean(axis=2) # Average along spatial axis
        tracks = np.transpose(tracks)
        
        wl_pos = []
        for i in range(nb_tracks):
            popt, pcov = curve_fit(gaussian, spectral_axis, tracks[i], p0=[tracks[i].max(), spectral_axis[np.argmax(tracks[i])], 1., 0])
            wl_pos.append(popt[1:-1])
            plt.figure(10+data_list0.index(data_list), figsize=(19.2, 10.8))
            plt.subplot(4,4,i+1)
            plt.plot(spectral_axis, tracks[i], '.')
            plt.plot(spectral_axis, gaussian(spectral_axis, *popt))
            plt.grid('on')
            plt.title(labels[i])
            plt.suptitle(str(wavelength[data_list0.index(data_list)]))
            plt.tight_layout()
            if save: plt.savefig(output_path+'fitting_%s'%(wavelength[data_list0.index(data_list)]))
        calib_pos.append(wl_pos)
    
    calib_pos = np.array(calib_pos)
    coeff_poly_wl_to_px = np.array([np.polyfit(wavelength, calib_pos[:,i,0], deg=1) for i in range(nb_tracks)]) # detector resolution is around 5 nm/px
    coeff_poly_px_to_wl = np.array([np.polyfit(calib_pos[:,i,0], wavelength, deg=1) for i in range(nb_tracks)])
    poly_wl = [np.poly1d(coeff_poly_wl_to_px[i]) for i in range(nb_tracks)]
    poly_px = [np.poly1d(coeff_poly_px_to_wl[i]) for i in range(nb_tracks)]
    
    spectral_psf_pos = np.array([poly_px[i](calib_pos[:,i,0]) for i in range(16)]).T
    spectral_psf_sig = calib_pos[:,:,1] * abs(coeff_poly_px_to_wl[None,:,0])
    spectral_psf = np.stack([spectral_psf_pos, spectral_psf_sig], axis=2)
    
    if save:
        np.save(output_path+'wl_to_px', coeff_poly_wl_to_px)
        np.save(output_path+'px_to_wl', coeff_poly_px_to_wl)
        np.save(output_path+'spectral_psf', spectral_psf)
    
    fwhm = 2 * np.sqrt(2*np.log(2)) * calib_pos[:,:,1] * abs(coeff_poly_px_to_wl[None,:,0])
    print('Spectral resolution for')
    for wl in wavelength:
        print(str(wl)+' nm -> '+str(wl/fwhm.mean(axis=1)[wavelength.index(wl)]))
        
    if save:
        with open(output_path+'spectral_resolution.txt', 'a') as sr:
            sr.write('Spectral resolution for:\n')
            for wl in wavelength:
                sr.write(str(wl)+' nm -> \t'+str(wl/fwhm.mean(axis=1)[wavelength.index(wl)])+'\n')
            sr.write('\n')
            
    plt.figure(figsize=(19.2, 10.8))
    for i in range(nb_tracks):
        plt.subplot(4,4,i+1)
        plt.plot(wavelength, calib_pos[:,i,0], 'o')
        plt.plot(wavelength, poly_wl[i](wavelength))
        plt.grid()
        plt.title('Track %s'%(i+1))
    plt.tight_layout()
    if save: plt.savefig(output_path+'wl2px_%s'%(wavelength[data_list0.index(data_list)]))

    print('Deconvolution from CHARIS tunable laser')
    fwhm2 = fwhm.mean(axis=1)
    measured_sigma = fwhm2 / (2 * np.sqrt(2*np.log(2)))
    measured_sigma2 = calib_pos[:,:,1] * abs(coeff_poly_px_to_wl[None,:,0])

    x = np.array([400, 1000])
    y = np.array([1, 2])
    x2 = np.array([1000, 2300])
    y2 = np.array([2, 5])
    coeff = np.polyfit(x, y,1)
    coeff2 = np.polyfit(x2, y2, 1)
    p = np.poly1d(coeff)
    p2 = np.poly1d(coeff2)
    laser_sig = p2(wavelength) / (2 * np.sqrt(2*np.log(2)))
    deconv_sig = (measured_sigma**2 - laser_sig**2)**0.5
    deconv_sig2 = (measured_sigma2**2 - laser_sig[:,None]**2)**0.5
    deconv_fwhm = deconv_sig * 2 * np.sqrt(2*np.log(2))
    deconv_fwhm2 = deconv_sig2 * 2 * np.sqrt(2*np.log(2))
    
    for wl in wavelength:
        print(str(wl)+' nm -> '+str(wl/deconv_fwhm[wavelength.index(wl)]))
        
    if save:
        with open(output_path+'spectral_resolution.txt', 'a') as sr:
            sr.write('Spectral resolution after deconvolution for:\n')
            for wl in wavelength:
                sr.write(str(wl)+' nm -> \t'+str(wl/deconv_fwhm[wavelength.index(wl)])+'\n')
            sr.write('\n')        