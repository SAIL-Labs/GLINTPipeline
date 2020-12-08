# -*- coding: utf-8 -*-
"""
This script measures the intensity in every output, per spectral channel and 
computes the null depths.
It relies on the library :doc:`glint_classes` to work.

The inputs are the datacubes (target or dark).
The datacube of **dark** needs to be processed as it gives the distribution of dark currents in the different outputs used in the model fitting later.
The datacube of **target** gives the **null depth** and the intensities are used to monitor any suspicious behavior if needed.

The products are HDF5 files structured as a dictionary (see glint_classes documentation).
One HDF5 produces per datacube.

It contains:
    * 1d-arrays for the null depth, named **nullX** (with X=1..6). The elements of the array are the null depths per spectral channel.
    * 1d-array for the photometry, named **pX** (with X=1..4). It has the same structure as ``nullX``.
    * 1d-array for the intensity in the null outputs, named **IminusX** (with X=1..6).
    * 1d-array for the intensity in the anti-null outputs, named **IplusX** (with X=1..6).
    * 1d-array containing the common spectral channels for all the outputs (as each output is slightly shifted from the others)
    
Some monitoring data can be created (but not saved):
    * Histograms of intensities of the photometries
    * Optimal parameters from a Gaussian fitting of the intensity profile in one output for one spectral channel
    * Evolution of the null depths along the frame axis for 10 spectral channels
    * Evolution of the measured intensities along the frame axis of every outputs according to different estimators

The monitoring is activated by setting the boolean variable **debug = True**.

In that case, it is strongly advised deactivate the save of the results and 
to process one frame of one datacube to avoid extremely long data processing.


In order to increase the SNR in the extraction of the flux in the photometric output, ones can create the spectra in them by averaging the frames.
The spectra are then normalized so that their integral in the bandwidth is equal to 1.
Therefore, the extraction of the photometries on a frame basis first estimates the total flux in bandwidth then the spectral flux is given by the product of this total flux with the spectra.
However, the gain of SNR is barely significant so this mode should not be used.


This script is used in 3 steps.

First step: simply change the value of the variables in the **Settings** section:
    * **save**: boolean, ``True`` for saving products and monitoring data, ``False`` otherwise
    * **no_noise**: boolean, ``True`` for noise-free (simulated) data
    * **nbfiles**: 2-tuple of int, set the bounds between which the data files are selected. ``None`` is equivalent to 0 if it is the lower bound or -1 included or it is the upper one.
    * **nb_img**: 2-tuple of int, set the bounds between which the frame are selected, into a data file.
    * **nulls_to_invert**: list of null outputs to invert. Fill with ``nullX`` (X=1..6) or leave empty if no null is to invert (deprecated)
    * **bin_frames**: boolean, set True to bin frames
    * **nb_frames_to_bin**: number of frames to bin (average) together. If ``None``, the whole stack is average into one frame. If the total number of frames is not a multiple of the binning value, the remaining frames are lost.
    * **spectral_binning**: bool, set to ``True`` to spectrally bins the outputs
    * **wl_bin_min**: scalar, lower bounds (in nm) of the bandwidth to bin, possibly in several chunks
    * **wl_bin_max**: scalar, upper bounds (in nm) of the bandwidth to bin, possibly in several chunks
    * **bandwidth_binning**: scalar, width of the chunks of spectrum to bin between the lower and upper bounds
    * **mode_flux**: string, choose the method to estimate the spectral flux in the outputs among:
        * ``amplitude`` uses patterns determined in the script ``glint_geometric_calibration`` and a linear least square is performed to get the amplitude of the pattern
        * ``model`` proceeds like ``amplitude`` but the integral of the flux is returned
        * ``windowed`` returns a weighted mean as flux of the spectral channel. The weights is the same pattern as the other modes above
        * ``raw`` returns the mean of the flux along the spatial axis over the whole width of the output        
    * **activate_estimate_spectrum**, boolean, if ``True``, the spectrum of the source in the photometric output is created.
    * **nb_files_spectrum**: tuple, range of files to read to get the spectra.
    * **wavelength_bounds**: tuple, bounds of the bandwidth one wants to keep after the extraction. Used in the method ``getIntensities``. It works independantly of **wl_bin_min** and **wl_bin_max**.
    * **suffix**: str, suffix to distinguish plots respect to data present in the datafolder (e.g. dark, baselines, stars...)

Second step: change the value of the variables in the **Inputs** and **Outputs** sections:
    * **datafolder**: folder containing the datacube to use.
    * **root**: path to **datafolder**.
    * **data_list**: list of files in **datafolder** to open.
    * **spectral_calibration_path**: path to the spectral calibration files used to process the file
    * **geometric_calibration_path**: path to the geometric calibration files used to process the file (location and width of the outputs per spectral channel)
    
Third step: start the script and let it run.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glint_classes
import warnings
from timeit import default_timer as time
import h5py
from scipy.optimize import curve_fit

def gaussian(x, A, loc, sig):
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
    return A * np.exp(-(x-loc)**2/(2*sig**2))

if __name__ == '__main__':
    warnings.filterwarnings(action="ignore", category=np.VisibleDeprecationWarning) # Ignore deprecation warning
    ''' Settings '''
    no_noise = False
    nb_img = (0, None)
    debug = False
    save = True
    nb_files = (2000, None)
    bin_frames = False
    nb_frames_to_bin = 50
    spectral_binning = True
    wl_bin_min, wl_bin_max = 1525, 1575# In nm
    bandwidth_binning = 50 # In nm
    mode_flux = 'raw'
    activate_estimate_spectrum = False
    nb_files_spectrum = (5000,10000)
    wavelength_bounds = (1400, 1700)
    suffix = 'n5n6'
#    ron = 0
    
    mode_flux_list = ['raw', 'fit']
    if not mode_flux in mode_flux_list:
        raise Exception('Select mode of flux measurement among:', mode_flux_list)
        
    ''' Inputs '''
    datafolder = 'data202006/AlfBoo/'
#    root = "C:/Users/marc-antoine/glint/"
    # root = "/mnt/96980F95980F72D3/glint/"
    root = "//tintagel.physics.usyd.edu.au/snert/"
    output_path = root+'GLINTprocessed/'+datafolder
    spectral_calibration_path = output_path
    geometric_calibration_path = output_path
    data_path = '//tintagel.physics.usyd.edu.au/snert/GLINTData/'+datafolder
    # data_path = 'C:/Users/marc-antoine/glint//GLINTData/'+datafolder
    # data_path = '/mnt/96980F95980F72D3/glint_data/'+datafolder
    data_list = sorted([data_path+f for f in os.listdir(data_path) if suffix in f])
    plot_name = datafolder.split('/')[-2]
    if len(data_list) == 0:
        raise IndexError('Data list is empty')

    
    if no_noise:
        dark = np.zeros((344,96))
        dark_per_channel = np.zeros((96,16,20))
    else:
        dark = np.load(output_path+'superdark.npy')
        dark_per_channel = np.load(output_path+'superdarkchannel.npy')
    
    ''' Set processing configuration and load instrumental calibration data '''
    nb_tracks = 16 # Number of tracks
    # coeff_pos = np.load(geometric_calibration_path+'coeff_position_poly.npy')
    # coeff_width = np.load(geometric_calibration_path+'coeff_width_poly.npy')
    # position_poly = [np.poly1d(coeff_pos[i]) for i in range(nb_tracks)]
    # width_poly = [np.poly1d(coeff_width[i]) for i in range(nb_tracks)]
    pattern_coeff = np.load(geometric_calibration_path+'pattern_coeff.npy')
    position_outputs = pattern_coeff[:,:,1].T
    width_outputs = pattern_coeff[:,:,2].T
    wl_to_px_coeff = np.load(spectral_calibration_path+'20200601_wl_to_px.npy')
    px_to_wl_coeff = np.load(spectral_calibration_path+'20200601_px_to_wl.npy')
    
    
    spatial_axis = np.arange(dark.shape[0])
    spectral_axis = np.arange(dark.shape[1])
    
    ''' Define bounds of each track '''
    y_ends = [33, 329] # row of top and bottom-most Track
    sep =  (y_ends[1] - y_ends[0])/(nb_tracks-1)
    channel_pos = np.around(np.arange(y_ends[0], y_ends[1]+sep, sep))
    
    ''' Get the spectrum of photometric channels '''
    nb_frames = 0
    slices_spectrum = np.zeros_like(dark_per_channel)

    if not 'dark' in data_list[0] and not os.path.exists(output_path+'spectra.npy') and activate_estimate_spectrum:
        print('Determining spectrum\n')
        for f in data_list[nb_files_spectrum[0]:nb_files_spectrum[1]]:
            start = time()
            print("Process of : %s (%d / %d)" %(f, data_list.index(f)+1, len(data_list[nb_files_spectrum[0]:nb_files_spectrum[1]])))
            img_spectrum = glint_classes.Null(f, nbimg=nb_img)
            
            ''' Process frames '''
            img_spectrum.cosmeticsFrames(np.zeros(dark.shape), no_noise)
            
            ''' Insulating each track '''
            img_spectrum.getChannels(channel_pos, sep, spatial_axis, dark=dark_per_channel)
    
            img_spectrum.matchSpectralChannels(wl_to_px_coeff, px_to_wl_coeff)
            img_spectrum.getSpectralFlux(spectral_axis, position_outputs, width_outputs, mode_flux)
            
            img_spectrum.getIntensities(mode=mode_flux, wl_bounds=wavelength_bounds)
            
            slices_spectrum = slices_spectrum + np.sum(img_spectrum.slices, axis=0)
            nb_frames = nb_frames + img_spectrum.nbimg
            stop = time()
            print('Spectrum time:', stop-start)
            
        slices_spectrum = slices_spectrum / nb_frames
        spectrum = glint_classes.Null(data=None, nbimg=(0,1))
        spectrum.cosmeticsFrames(np.zeros(dark.shape), no_noise)
        spectrum.getChannels(channel_pos, sep, spatial_axis)
        spectrum.slices = np.reshape(slices_spectrum, (1,slices_spectrum.shape[0], slices_spectrum.shape[1], slices_spectrum.shape[2]))
        spectrum.matchSpectralChannels(wl_to_px_coeff, px_to_wl_coeff)
        spectrum.getSpectralFlux(spectral_axis, position_outputs, width_outputs, mode_flux)
        
        spectrum.getIntensities(mode=mode_flux, wl_bounds=wavelength_bounds)
        spectrum.p1 = spectrum.p1[0] / spectrum.p1[0].sum()
        spectrum.p2 = spectrum.p2[0] / spectrum.p2[0].sum()
        spectrum.p3 = spectrum.p3[0] / spectrum.p3[0].sum()
        spectrum.p4 = spectrum.p4[0] / spectrum.p4[0].sum()
        spectra = np.array([spectrum.p1, spectrum.p2, spectrum.p3, spectrum.p4])
        del spectrum, img_spectrum
        np.save(output_path+'spectra', spectra)
        # plt.figure()
        # plt.subplot(221)
        # plt.plot(spectra[0])
        # plt.subplot(222)
        # plt.plot(spectra[1])
        # plt.subplot(223)
        # plt.plot(spectra[2])
        # plt.subplot(224)
        # plt.plot(spectra[3])
         

    ''' Output lists for different stages of the processing.
    Include data from all processed files '''
    amplitude = []
    amplitude_fit = []
    integ_raw = []
    integ_raw_err = []
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
    
    if os.path.exists(output_path+'spectra.npy') and activate_estimate_spectrum:
        spectra = np.load(output_path+'spectra.npy')
    
    ''' Start the data processing '''
    nb_frames = 0
    for f in data_list[nb_files[0]:nb_files[1]]:
        start = time()
        print("Process of : %s (%d / %d)" %(f, data_list.index(f)+1, len(data_list[nb_files[0]:nb_files[1]])))
        img = glint_classes.Null(f, nbimg=nb_img)
        
        ''' Process frames '''
        if bin_frames:
            img.data = img.binning(img.data, nb_frames_to_bin, axis=0, avg=True)
            img.nbimg = img.data.shape[0]
        img.cosmeticsFrames(np.zeros(dark.shape), no_noise)
        
        ''' Insulating each track '''
        print('Getting channels')
        img.getChannels(channel_pos, sep, spatial_axis, dark=dark_per_channel)
#        img.slices = img.slices + np.random.normal(0, ron, img.slices.shape)
        
        ''' Map the spectral channels between every chosen tracks before computing 
        the null depth'''
        img.matchSpectralChannels(wl_to_px_coeff, px_to_wl_coeff)
        
        ''' Measurement of flux per frame, per spectral channel, per track '''
        img.getSpectralFlux(spectral_axis, position_outputs, width_outputs, mode_flux, debug=debug)
        
        ''' Reconstruct flux in photometric channels '''
        img.getIntensities(mode=mode_flux, wl_bounds=wavelength_bounds)
        
        if activate_estimate_spectrum:
            integ = np.array([np.sum(img.p1, axis=1), np.sum(img.p2, axis=1), np.sum(img.p3, axis=1), np.sum(img.p4, axis=1)])
            new_photo = integ[:,:,None] * spectra[:,None,:]
    
            
            # plt.figure()
            # plt.plot(img.wl_scale[15], img.p1[0])
            # plt.plot(img.wl_scale[15], new_photo[0][0])
            # plt.plot(img.wl_scale[15], img.p1.mean(axis=0))
            # plt.figure()
            # plt.plot(img.wl_scale[13], img.p2[0])
            # plt.plot(img.wl_scale[13], new_photo[1][0])
            # plt.plot(img.wl_scale[15], img.p2.mean(axis=0))
            # plt.figure()
            # plt.plot(img.wl_scale[2], img.p3[0])
            # plt.plot(img.wl_scale[2], new_photo[2][0])
            # plt.plot(img.wl_scale[15], img.p3.mean(axis=0))
            # plt.figure()
            # plt.plot(img.wl_scale[0], img.p4[0])
            # plt.plot(img.wl_scale[0], new_photo[3][0])
            # plt.plot(img.wl_scale[15], img.p4.mean(axis=0))
            
            img.p1, img.p2, img.p3, img.p4 = new_photo

        if spectral_binning:
            img.spectralBinning(wl_bin_min, wl_bin_max, bandwidth_binning, wl_to_px_coeff)
            
        p1.append(img.p1)
        p2.append(img.p2)
        p3.append(img.p3)
        p4.append(img.p4)
        
        ''' Compute null depth '''
        print('Computing null depths')
        img.computeNullDepth()
        null_depths = np.array([img.null1, img.null2, img.null3, img.null4, img.null5, img.null6])
        null_depths_err = np.array([img.null1_err, img.null2_err, img.null3_err, img.null4_err, img.null5_err, img.null6_err])
        
        ''' Output file'''
        if save:
            img.save(output_path+os.path.basename(f)[:-4]+'.hdf5', '2019-04-30')
            print('Saved')
    
        null.append(np.transpose(null_depths, axes=(1,0,2)))
        null_err.append(np.transpose(null_depths_err, axes=(1,0,2)))
        if debug:
            amplitude.append(img.amplitude)
            # integ_model.append(img.integ_model)
            # integ_windowed.append(img.integ_windowed)
            residuals_reg.append(img.amplitude_error)
            bg_noise.append(img.bg_std)
            integ_raw.append(img.raw)
            integ_raw_err.append(img.raw_err)
    
            
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
        
    '''
    Store quantities for monitoring purpose
    '''
    amplitude = np.array([selt for elt in amplitude for selt in elt])
    amplitude = np.array([[elt[i][img.px_scale[i]] for i in range(16)] for elt in amplitude])
    amplitude_fit = np.array([selt for elt in amplitude_fit for selt in elt])
    amplitude_fit = np.array([[elt[i][img.px_scale[i]] for i in range(16)] for elt in amplitude_fit])
    integ_raw = np.array([selt for elt in integ_raw for selt in elt])
    integ_raw = np.array([[elt[i][img.px_scale[i]] for i in range(16)] for elt in integ_raw])
    integ_raw_err = np.array([selt for elt in integ_raw_err for selt in elt])
    integ_raw_err = np.array([[elt[i][img.px_scale[i]] for i in range(16)] for elt in integ_raw_err])    
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
    
    # =============================================================================
    # Miscellaneous
    # =============================================================================
#    try:
#        photometries = [p1[:,56], p2[:,56], p3[:,56], p4[:,56]]
#    except IndexError:
#        photometries = [p1[:,0], p2[:,0], p3[:,0], p4[:,0]]
    photometries = [p1.mean(axis=1), p2.mean(axis=1), p3.mean(axis=1), p4.mean(axis=1)]
        
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
        if save: plt.savefig(output_path+plot_name+'_'+suffix+'_plot_histo_p%s.png'%(k+1))
    
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
    
    for i in range(5):
        plt.figure()
        plt.suptitle('Null')
        plt.subplot(321)
        plt.plot(img.wl_scale[0], null[i][0], '.')
        plt.grid()
        plt.subplot(322)
        plt.plot(img.wl_scale[0], null[i][1], '.')
        plt.grid()
        plt.subplot(323)
        plt.plot(img.wl_scale[0], null[i][2], '.')
        plt.grid()
        plt.subplot(324)
        plt.plot(img.wl_scale[0], null[i][3], '.')
        plt.grid()
        plt.subplot(325)
        plt.plot(img.wl_scale[0], null[i][4], '.')
        plt.grid()
        plt.subplot(326)
        plt.plot(img.wl_scale[0], null[i][5], '.')
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

# import cupy as cp
# positions = position_outputs
# widths = width_outputs
# gslices_axes = cp.asarray(img.slices_axes, dtype=cp.float32)
# gpositions = cp.asarray(positions, dtype=cp.float32)
# gwidths = cp.asarray(widths, dtype=cp.float32)
# std = 1/img.slices[:,:,:,:10-5].std()
# std = cp.array([std], dtype=cp.float32)

# simple_gaus0 = cp.exp(-(gslices_axes[:,None,:]-gpositions[:,:,None])**2/(2*gwidths[:,:,None]**2))
# simple_gaus = simple_gaus0 / cp.sum(simple_gaus0, axis=(-1))[:,:,None]
# simple_gaus[cp.isnan(simple_gaus)] = 0.

# weights0 = cp.asnumpy(simple_gaus + std)
# weights = np.zeros((*weights0.shape, weights0.shape[-1]))
# idx = np.diag_indices(weights.shape[-1])
# weights[:,:,idx[0],idx[1]]=weights0
# weights = cp.asarray(weights, dtype=cp.float32)

# gslices_axes2 = cp.repeat(gslices_axes[:,None,:], simple_gaus.shape[-2], axis=-2)
# A = cp.vstack((simple_gaus[None,:], cp.ones_like(simple_gaus)[None,:], gslices_axes2[None,:]))
# A = cp.transpose(A, axes=(1,2,0,3))
# Aw = A#cp.matmul(A, weights)
# Aw2 = cp.matmul(Aw, cp.transpose(Aw, (0,1,3,2)))

# data = cp.asarray(img.slices, dtype=cp.float32)
# data = cp.transpose(data, (0,2,1,3))
# dataw = data#cp.matmul(data[:,:,:,None,:], weights)
# # dataw = cp.reshape(dataw, data.shape)
# b = cp.matmul(Aw,dataw[:,:,:,:,None])
# a = cp.repeat(Aw2[None,:], b.shape[0], axis=0)

# popt2 = cp.linalg.solve(a, b)

# popt3 = cp.asnumpy(popt2)
# print(np.allclose(img.amplitude2, popt3[:,:,:,0,0]))

# offset = 20
# gp = positions
# gw = widths
# simple_gaus0 = np.exp(-(img.slices_axes[:,None,:]-gp[:,:,None])**2/(2*gw[:,:,None]**2))
# simple_gaus = simple_gaus0 / np.sum(simple_gaus0, axis=(-1))[:,:,None]
# simple_gaus[np.isnan(simple_gaus)] = 0.
# simple_gaus = simple_gaus[:,offset:]
# gslices_axes2 = np.repeat(img.slices_axes[:,None,:], simple_gaus.shape[-2], axis=-2)
# A = np.vstack((simple_gaus[None,:], np.ones_like(simple_gaus)[None,:], gslices_axes2[None,:]))
# A = np.transpose(A, axes=(1,2,0,3))
# B = np.matmul(A, np.transpose(A, (0,1,3,2)))
# data = np.asarray(img.slices[0,offset:])#, dtype=np.float32)
# data = np.transpose(data, (1,0,2))
# b = np.matmul(A,data[:,:,:,None])
# popt = cp.linalg.solve(cp.asarray(B), cp.asarray(b))
# poptbis = cp.asnumpy(popt)


# # p = position_outputs
# # w = width_outputs
# # simple_gaus02 = np.exp(-(img.slices_axes[:,None,:]-p[:,:,None])**2/(2*w[:,:,None]**2))
# # simple_gaus2 = simple_gaus02 / np.sum(simple_gaus02, axis=(-1))[:,:,None]
# # simple_gaus2[np.isnan(simple_gaus2)] = 0.
# # slices_axes2 = np.repeat(img.slices_axes[:,None,:], simple_gaus2.shape[-2], axis=-2)
# # A2 = np.vstack((simple_gaus2[None,:], np.ones_like(simple_gaus2)[None,:], slices_axes2[None,:]))
# # A2 = np.transpose(A2, axes=(1,2,0,3))
# # B2 = np.matmul(A2, np.transpose(A2, axes=(0,1,3,2)))
# # data2 = img.slices[0].copy()
# # data2 = np.transpose(data2, (1,0,2))
# # b2 = np.matmul(A2,data2[:,:,:,None])
# # popt2 = np.linalg.solve(B2, b2)
# # popt2bis = np.array(popt2, dtype=np.float32)