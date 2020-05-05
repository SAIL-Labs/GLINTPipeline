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
    save = False
    nb_files = (0, None)
    bin_frames = False
    nb_frames_to_bin = 50
    spectral_binning = False
    wl_bin_min, wl_bin_max = 1525, 1575# In nm
    bandwidth_binning = 50 # In nm
    mode_flux = 'amplitude'
    nb_files_spectrum = (0,10)
    wavelength_bounds = (1400, 1700)
    ron = 0
    
    ''' Inputs '''
    datafolder = 'NullerData_SubaruJuly2019/20190718/20190718_turbulence1/'
#    root = "C:/Users/marc-antoine/glint/"
    root = "/mnt/96980F95980F72D3/glint/"
    spectral_calibration_path = root+'GLINTprocessed/'+'calibration_params/'
    geometric_calibration_path = root+'GLINTprocessed/'+'calibration_params/'
#    data_path = '//silo.physics.usyd.edu.au/silo4/snert/GLINTData/'+datafolder
    data_path = '/mnt/96980F95980F72D3/glint_data/'+datafolder
    data_list = sorted([data_path+f for f in os.listdir(data_path) if 'n1n4' in f])
    if len(data_list) == 0:
        raise IndexError('Data list is empty')
    
    ''' Output '''
    output_path = root+'GLINTprocessed/'+datafolder
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if no_noise:
        dark = np.zeros((344,96))
        dark_per_channel = np.zeros((96,16,20))
    else:
        dark = np.load(output_path+'superdark.npy')
        dark_per_channel = np.load(output_path+'superdarkchannel.npy')
    
    ''' Set processing configuration and load instrumental calibration data '''
    nb_tracks = 16 # Number of tracks
    coeff_pos = np.load(geometric_calibration_path+'coeff_position_poly.npy')
    coeff_width = np.load(geometric_calibration_path+'coeff_width_poly.npy')
    position_poly = [np.poly1d(coeff_pos[i]) for i in range(nb_tracks)]
    width_poly = [np.poly1d(coeff_width[i]) for i in range(nb_tracks)]
    wl_to_px_coeff = np.load(spectral_calibration_path+'wl_to_px.npy')
    px_to_wl_coeff = np.load(spectral_calibration_path+'px_to_wl.npy')
    
    
    spatial_axis = np.arange(dark.shape[0])
    spectral_axis = np.arange(dark.shape[1])
    
    ''' Define bounds of each track '''
    y_ends = [33, 329] # row of top and bottom-most Track
    sep = (y_ends[1] - y_ends[0])/(nb_tracks-1)
    channel_pos = np.around(np.arange(y_ends[0], y_ends[1]+sep, sep))
    
    ''' Get the spectrum of photometric channels '''
    nb_frames = 0
    slices_spectrum = np.zeros_like(dark_per_channel)

    if not 'dark' in data_list[0]:
        for f in data_list[nb_files_spectrum[0]:nb_files_spectrum[1]]:
            start = time()
            print("Process of : %s (%d / %d)" %(f, data_list.index(f)+1, len(data_list[nb_files_spectrum[0]:nb_files_spectrum[1]])))
            img_spectrum = glint_classes.Null(f, nbimg=nb_img)
            
            ''' Process frames '''
            img_spectrum.cosmeticsFrames(np.zeros(dark.shape), no_noise)
            
            ''' Insulating each track '''
            img_spectrum.getChannels(channel_pos, sep, spatial_axis, dark=dark_per_channel)
    
            img_spectrum.matchSpectralChannels(wl_to_px_coeff, px_to_wl_coeff)
            list_channels = np.arange(16) #[1,3,4,5,6,7,8,9,10,11,12,14]
            img_spectrum.getSpectralFlux(list_channels, spectral_axis, position_poly, width_poly)
            
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
        list_channels = np.arange(16) #[1,3,4,5,6,7,8,9,10,11,12,14]
        spectrum.getSpectralFlux(list_channels, spectral_axis, position_poly, width_poly)
        
        spectrum.getIntensities(mode=mode_flux, wl_bounds=wavelength_bounds)
        spectrum.p1 = spectrum.p1[0] / spectrum.p1[0].sum()
        spectrum.p2 = spectrum.p2[0] / spectrum.p2[0].sum()
        spectrum.p3 = spectrum.p3[0] / spectrum.p3[0].sum()
        spectrum.p4 = spectrum.p4[0] / spectrum.p4[0].sum()
        spectra = np.array([spectrum.p1, spectrum.p2, spectrum.p3, spectrum.p4])
        del spectrum, img_spectrum
        # plt.figure()
        # plt.subplot(221)
        # plt.plot(spectra[0])
        # plt.subplot(222)
        # plt.plot(spectra[1])
        # plt.subplot(223)
        # plt.plot(spectra[2])
        # plt.subplot(224)
        # plt.plot(spectra[3])
        # ppp

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
        img.slices = img.slices + np.random.normal(0, ron, img.slices.shape)
        
        ''' Map the spectral channels between every chosen tracks before computing 
        the null depth'''
        img.matchSpectralChannels(wl_to_px_coeff, px_to_wl_coeff)
        
        ''' Measurement of flux per frame, per spectral channel, per track '''
        list_channels = np.arange(16) #[1,3,4,5,6,7,8,9,10,11,12,14]
        positions_tracks, width_tracks = img.getSpectralFlux(list_channels, spectral_axis, position_poly, width_poly, debug=debug)
        
        ''' Reconstruct flux in photometric channels '''
        img.getIntensities(mode=mode_flux, wl_bounds=wavelength_bounds)
        
        if not 'dark' in data_list[0]:
            integ = np.array([np.sum(img.p1, axis=1), np.sum(img.p2, axis=1), np.sum(img.p3, axis=1), np.sum(img.p4, axis=1)])
            new_photo = integ[:,:,None] * spectra[:,None,:]
    
            stack = np.transpose(img.slices[:,:,15,:], (1,0,2))
            stack = stack[img.px_scale[15]]
            stack = stack.mean(axis=0) # Integer along spectral axis
            stack_err = stack[:,(np.arange(20)<=5)|(np.arange(20)>=15)].std() * np.ones_like(stack)
            x = img.slices_axes[15]
            
            def model2(x, amp, x0, sigma):
                expo = np.exp(-(x-x0)**2/(2*sigma**2))
                return amp * expo / expo.sum()
            popt, pcov = curve_fit(model2, x, stack[0], [stack[0].max(), x.mean(), 2], stack_err[0], True)

            plt.figure()
            plt.plot(img.wl_scale[15], img.p1[0])
            plt.plot(img.wl_scale[15], new_photo[0][0])
            plt.plot(img.wl_scale[15], img.p1.mean(axis=0))
            plt.plot(img.wl_scale[15], spectra[0]*popt[0]*img.wl_scale[15].size)
            
            plt.figure()
            plt.plot(abs(new_photo[0][0]-img.p1[0])/img.p1[0])
            plt.plot(abs(img.p1.mean(axis=0)-img.p1[0])/img.p1[0])
            plt.plot(abs(spectra[0]*popt[0]*img.wl_scale[15].size-img.p1[0])/img.p1[0])
            
            print(np.mean(abs(new_photo[0][0]-img.p1[0])/img.p1[0]))
            print(np.mean(abs(img.p1.mean(axis=0)-img.p1[0])/img.p1[0]))
            print(np.mean(abs(spectra[0]*popt[0]*img.wl_scale[15].size-img.p1[0])/img.p1[0]))
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
            ppp
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
        
    '''
    Store quantities for monitoring purpose
    '''
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
    
    # =============================================================================
    # Miscellaneous
    # =============================================================================
    try:
        photometries = [p1[:,56], p2[:,56], p3[:,56], p4[:,56]]
    except IndexError:
        photometries = [p1[:,0], p2[:,0], p3[:,0], p4[:,0]]
        
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
        plt.savefig(output_path+'plot_histo_p%s.png'%(k+1))
    
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

#histo_amplitude = np.histogram(null_amplitude[:,3,0], bins=int(934**0.5), density=True)
#histo_raw = np.histogram(null_raw[:,3,0], bins=int(934**0.5), density=True)
#histo_pixel = np.histogram(null_pixel[:,3,0], bins=int(934**0.5), density=True)
#histo_interp = np.histogram(null[:,3,0], bins=int(934**0.5), density=True)
#
#plt.figure()
#plt.plot(histo_amplitude[1][:-1], histo_amplitude[0], lw=2)
#plt.plot(histo_raw[1][:-1], histo_raw[0], lw=2)
#plt.plot(histo_pixel[1][:-1], histo_pixel[0], lw=2)
#plt.plot(histo_interp[1][:-1], histo_interp[0], lw=2)
#plt.grid()
#plt.legend(['fit', 'raw', 'pixel', 'interp'], loc='best', fontsize=30)
#plt.xticks(size=30);plt.yticks(size=30)
#plt.ylabel('Count (normalized)', size=35)
#plt.xlabel('Null depth', size=35)
#plt.title('Histogram of N4', size=40)

# from scipy.optimize import curve_fit
# def model1(x, amp):
#     global x0, sigma
#     return amp * np.exp(-(x-x0)**2/(2*sigma**2))

# popt_list = []
# pcov_list = []
# for k in range(96):
#     x0 = positions_tracks[15,k]
#     sigma = width_tracks[15, k]
#     y = img.slices[-1,k,15,:]
#     yerr = img.slices[:,:10,15,:].std() * np.ones_like(y)
#     yerr = y[(np.arange(20)<=5)|(np.arange(20)>=15)].std() * np.ones_like(y)
#     x = img.slices_axes[15]
#     popt, pcov = curve_fit(model1, x, y, [1], yerr, True)
#     popt_list.append(popt)
#     pcov_list.append(pcov)
    
# popt_list = np.array(popt_list)
# pcov_list = np.array(pcov_list)

# popt_list = popt_list[img.px_scale[15]]
# pcov_list = pcov_list[img.px_scale[15]]
# integ = popt_list.mean(axis=0)
# var_integ = pcov_list.sum(axis=0) / popt_list.size**2
# snr = integ / np.diag(var_integ)**0.5

# stack = img.slices[-1,:,15,:]
# stack = stack[img.px_scale[15]]
# stack = stack.mean(axis=0)
# stack_err = img.slices[:,:10,15,:].std() * img.px_scale[15].size**0.5 * np.ones_like(stack)
# stack_err = stack[(np.arange(20)<=5)|(np.arange(20)>=15)].std() * np.ones_like(stack)
# x = img.slices_axes[15]

# def model2(x, amp):
#     x0 = positions_tracks[15].mean()
#     sigma = width_tracks[15].mean()
#     return amp * np.exp(-(x-x0)**2/(2*sigma**2))

# popt, pcov = curve_fit(model2, x, stack, [stack.max()], stack_err, True)
# snr2 = popt/np.diag(pcov)**0.5

# print(snr)
# print(snr2)
# print(integ, np.diag(var_integ)**0.5)
# print(popt, np.diag(pcov)**0.5)
