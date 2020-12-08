#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 19:48:06 2020

@author: mam
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from timeit import default_timer as time
from scipy.optimize import curve_fit
from scipy.io import loadmat
import glint_classes

def envelope(x, opd0, bw, amp):
    global wl
    # bw = 0.2
    lc = wl**2/bw    
    y = abs(np.sinc((x-opd0)/lc)) * amp
    
    return y
   
def lowpass_filter(freq, fc):
    lowpass = np.zeros_like(freq)
    lowpass[(freq>=-fc)&(freq<=fc)] = 1.
    return lowpass
    
def processSimpleScan(data_list, pos_seg2):  
    global scanrange
    print("Process of : %s" %(data_list))
    fichier = loadmat(data_list[0])
    rtscan = fichier['curVals'][:,0]
    rtscan /= rtscan.mean()
    scanrange = fichier['memsScanRange'][0]
    opd = (scanrange - pos_seg2)*1000*2

    return rtscan, opd

def modelsinminus(x, opd0, phi, *args):
    global wl, bandwidth_binning, fit_wavelength
    
#    amp = 1.
#    phi = 0
    bw = bandwidth_binning #5.
#    lamb0 = 1.628
    
    fringes = []
    for i in range(wl.size):
        if fit_wavelength:
            wavel = wl[i]+args[2*wl.size+i]
        else:
            wavel = wl[i]
#        fringe = (args[i] - args[wl.size+i] * np.sin(2*np.pi/wl[i] * (x-opd0) + phi) * np.sinc((x-opd0)/wl[i]**2 * bw))
        fringe = (args[i] - args[wl.size+i] * np.sin(2*np.pi/wavel * (x-opd0) + phi) * np.sinc((x-opd0)/wavel**2 * bw))
        fringes.append(fringe)
    
    fringes = np.array(fringes)
    
    return fringes.ravel()

def modelsinminus2(x, opd0, bw, *args):
    global wl, fit_wavelength
    
    phi=0
    fringes = []
    for i in range(wl.size):
        if fit_wavelength:
            wavel = wl[i]+args[2*wl.size+i]
        else:
            wavel = wl[i]
        fringe = (args[i] - args[wl.size+i] * np.sin(2*np.pi/wavel * (x-opd0) + phi) * np.sinc((x-opd0)/wavel**2 * bw))
        fringes.append(fringe)
    
    fringes = np.array(fringes)
    
    return fringes.ravel()

def processFullFrame(data_list, pos_seg2, maskbounds, swap_ref, which_null=None):
    global img, dark, dark_per_channel, wl, fichier, rtscan
    global data, spectral_axis, position_outputs, width_outputs, debug
    global spectral_binning, wl_bin_min, wl_bin_max,bandwidth_binning, mode_flux
    global scanrange, scanrange0, wl0, selected_Iminus, mask, data0, scans0
    
    scans = []
    opds = []
    rtscans = []
    
    if which_null == None:
        for i in range(6):
            if 'null'+str(i+1) in data_list[0]:
                which_null = i+1
                break
            
    ''' Start the data processing '''
    for f in data_list[:]:
        print("Process of : %s (%d / %d)" %(f, data_list.index(f)+1, len(data_list)))
        fichier = loadmat(f)
        '''Scan range parameters'''
            
        data0 = fichier['fullScanAllImages']
        if len(data0.shape) == 4:
            nloop = data0.shape[-1]
        else:
            nloop = 1
            
        rtscan = fichier['curVals'][:,0]
        rtscan /= rtscan.mean()
        rtscans.append(rtscan)
        scanrange = fichier['memsScanRange'][0]
        scanrange0 = fichier['memsScanRange'][0]
        opd = (scanrange - pos_seg2)*1000*2
        if swap_ref:
            opd = -opd

            
        opds.append(opd)
        
        for k in range(nloop)[:]:
            if nloop == 1:
                data = np.transpose(data0, axes=(2, 0, 1)) # New order: frame (or OPD), spatial, spectral
            else:
                data = np.transpose(data0[:,:,:,k], axes=(2, 0, 1))
                
            img = glint_classes.Null(None, nbimg=(0,data.shape[0]))
            img.data = data
            
            ''' Process frames '''
            if bin_frames:
                img.data = img.binning(img.data, nb_frames_to_bin, axis=0, avg=True)
                img.nbimg = img.data.shape[0]
            img.cosmeticsFrames(np.zeros(dark.shape), no_noise)
            
            ''' Insulating each track '''
            print('Getting channels')
            img.getChannels(channel_pos, sep, spatial_axis, dark=dark_per_channel)
            
            ''' Map the spectral channels between every chosen tracks before computing 
            the null depth'''
            img.matchSpectralChannels(wl_to_px_coeff, px_to_wl_coeff)
            
            ''' Measurement of flux per frame, per spectral channel, per track '''
            img.getSpectralFlux(spectral_axis, position_outputs, width_outputs, mode_flux, debug=debug)
            
            ''' Measure flux in photometric channels '''
            img.getIntensities(mode=mode_flux)
            if spectral_binning:
                img.spectralBinning(wl_bin_min, wl_bin_max, bandwidth_binning, wl_to_px_coeff)    
                
            Iminus = np.array([img.Iminus1, img.Iminus2, img.Iminus3, img.Iminus4, img.Iminus5, img.Iminus6])
            Iminus = Iminus / Iminus.mean(axis=1)[:,None,:]
            print('Null', which_null)
            selected_Iminus = Iminus[which_null-1].T
            wl = img.wl_scale[0]
            wl0 = wl.copy()
            scans.append(selected_Iminus)
           
    mask = np.ones_like(scanrange, dtype=np.bool)
    mask[(scanrange<maskbounds[0])|(scanrange>maskbounds[1])] = False
    scanrange = scanrange[mask]
    scans = np.array(scans)
    rtscans = np.array(rtscans)
    opds = np.array(opds)
    scans0 = scans.copy()
    scans = np.mean(scans, axis=0)
    rtscans = np.mean(rtscans, axis=0)
    opds = np.mean(opds, axis=0)
    return scans[:,mask], rtscans[mask], opds[mask]

    
''' How to run '''
run_mode = 'fullframe'
# run_mode = 'simplescan'
# run_mode = 'concatenate'

if run_mode ==  'fullframe':  
    print('*** FULL FRAME MODE ***')
    ''' Settings '''
    no_noise = False
    nb_img = (0, 1)
    debug = False
    save = False
    nb_files = (0, None)
    bin_frames = False
    nb_frames_to_bin = 1
    spectral_binning = True
    wl_bin_min, wl_bin_max = 1450, 1600# In nm
    mode_flux = 'raw'
    bandwidth_binning = 5 # In nm    
    #kind_data = 'fullframe'
    kind_data = 'scan'
    concatenate = False
    wl_offset = 0.
    maskbounds = (-2.4, 2.)
    find_null = True
    
    fit_wavelength = True
    delta_bounds = (-4000, -2000)
    delta0 = np.mean(delta_bounds)
    
    ''' Inputs '''
    datafolder = 'data202009/20200917/scans/'
    #root = "C:/Users/marc-antoine/glint/"
    root = "/mnt/96980F95980F72D3/glint/"
    output_path = root+'GLINTprocessed/'+datafolder
    spectral_calibration_path = output_path
    geometric_calibration_path = output_path
    data_path = '/mnt/96980F95980F72D3/glint_data/'+datafolder
    data_list = [data_path+'null4_24at1_fullIms_20200916T185129.mat']
    pos_seg_ref = 1. # Piston value of the refererence segment in micron
    swap_ref = False
            
    if no_noise:
        dark = np.zeros((344,96))
        dark_per_channel = np.zeros((96,16,20))
    else:
        dark = np.load(output_path+'superdark.npy')
        dark_per_channel = np.load(output_path+'superdarkchannel.npy')
    
    ''' Set processing configuration and load instrumental calibration data '''
    nb_tracks = 16 # Number of tracks
    if mode_flux == 'raw':
        pattern_coeff = np.zeros((96, 16, 4))
    else:
        pattern_coeff = np.load(geometric_calibration_path+'pattern_coeff.npy')
    position_outputs = pattern_coeff[:,:,1].T
    width_outputs = pattern_coeff[:,:,2].T
    wl_to_px_coeff = np.load(spectral_calibration_path+'20200906_wl_to_px.npy')
    px_to_wl_coeff = np.load(spectral_calibration_path+'20200906_px_to_wl.npy')
    
    
    spatial_axis = np.arange(dark.shape[0])
    spectral_axis = np.arange(dark.shape[1])
    
    ''' Define bounds of each track '''
    y_ends = [33, 329] # row of top and bottom-most Track
    sep = (y_ends[1] - y_ends[0])/(nb_tracks-1)
    channel_pos = np.around(np.arange(y_ends[0], y_ends[1]+sep, sep))
    
    ''' Get the flux respect to OPD for multi/omono-wavelength '''
    scans2, rtscans2, opds2 = processFullFrame(data_list, pos_seg_ref, maskbounds, swap_ref)
    # scans2 = np.mean(scans2, axis=0)
    # rtscans2 = np.mean(rtscans2, axis=0)
    # opds2 = np.mean(opds2, axis=0)
    wl0 = wl.copy()
    
    concatenated_opds = opds2
    concatenated_scans = scans2
    rtconcatenated_scans = rtscans2
    
    plt.figure(figsize=(19.32,10.8))
    plt.plot(scanrange, rtconcatenated_scans, lw=3)
    plt.grid()
    plt.xlabel(r'Piston value ($\mu$m)', size=35)
    plt.ylabel('Intensity (normalised)', size=35)
    plt.xticks(size=30);plt.yticks(size=30)
    plt.tight_layout()

    plt.figure(figsize=(19.32,10.8))
    plt.plot(concatenated_opds, rtconcatenated_scans)
    plt.grid()
    plt.xlabel('OPD between the segments (nm)', size=35)
    plt.ylabel('Intensity', size=35)
    plt.xticks(size=30);plt.yticks(size=30)
    plt.tight_layout()
    
    plt.figure(figsize=(19.32,10.8))
    plt.plot(concatenated_opds, concatenated_scans.std(axis=0), '.')
    plt.grid()
    plt.xlabel('OPD between the segments (nm)', size=35)
    plt.ylabel(r'Dispersion between wavelength', size=35)
    plt.xticks(size=30);plt.yticks(size=30)
    plt.tight_layout()
        
    plt.figure(figsize=(19.32,10.8))
    plt.plot(concatenated_opds, concatenated_scans.T, '.')
    plt.grid()
    plt.xlabel('OPD between the segments (nm)', size=35)
    plt.ylabel(r'Intensity (Multi-$\lambda$)', size=35)
    plt.xticks(size=30);plt.yticks(size=30)
    plt.tight_layout()

    x = np.linspace(0, 100000, 1000)
    y=np.sinc(x/(1560**2/165))+1
    y2=-np.sinc(x/(1560**2/165))+1
    y0 = np.sin(2*np.pi/1560*x) * (y-1) + 1
    plt.figure()
    plt.plot(x, y)
    plt.plot(x, y0)
    plt.plot(x, y2)
    plt.grid()
    plt.xlabel('OPD (nm)')
    plt.ylabel('Amplitude')
    plt.title('Generic envelop of coherence')
    del x, y, y2

    ''' Find the null '''
    print('')
    if find_null:
        wl = wl0.copy()
    
        print('Spectrally dispersed')
        bandwidth_binning = 5.4 # In nm
        if fit_wavelength:
            init_guess = [delta0, 0, *np.ones(wl.size), *np.ones(wl.size), *np.zeros(wl.size)]
            bounds = [(delta_bounds[0], -2*np.pi, *np.ones(wl.size)*0.9, *np.ones(wl.size)*0.8, *np.ones(wl.size)*(-50)), \
                      (delta_bounds[1], 2*np.pi, *np.ones(wl.size)*1.1, *np.ones(wl.size)*1.2, *np.ones(wl.size)*50)]
        else:
            init_guess = [delta0, 0, *np.ones(wl.size), *np.ones(wl.size)]
            bounds = [(delta_bounds[0], -2*np.pi, *np.ones(wl.size)*0.9, *np.ones(wl.size)*0.8), \
                      (delta_bounds[1], 2*np.pi, *np.ones(wl.size)*1.1, *np.ones(wl.size)*1.2)]

        popt_dispersed, pcov_dispersed = curve_fit(modelsinminus, concatenated_opds, concatenated_scans.ravel(), p0 = init_guess, bounds=bounds)

        curve = modelsinminus(concatenated_opds, *popt_dispersed)
        chi2_dispersed = np.sum((concatenated_scans.ravel() - curve)**2) / (concatenated_scans.size-popt_dispersed.size)
        print(popt_dispersed[:2])
        print(popt_dispersed[2:2+wl.size])
        print(popt_dispersed[2+wl.size:2+wl.size+wl.size])
        print(popt_dispersed[2+wl.size+wl.size:])
        print(chi2_dispersed)
        
        curve = np.reshape(curve, (wl.size, -1))
        plt.figure(figsize=(19.32,10.8))
        plt.plot(concatenated_opds, concatenated_scans.T, '.')
        plt.gca().set_prop_cycle(None)
        plt.plot(concatenated_opds, curve.T, '-')
        plt.grid()
        plt.xlabel('OPD between the segments (nm)', size=35)
        plt.ylabel(r'Intensity (Multi-$\lambda$)', size=35)
        plt.xticks(size=30);plt.yticks(size=30)
        plt.tight_layout()
    
        plt.figure(figsize=(19.32,10.8))
        plt.plot(concatenated_opds-popt_dispersed[0], concatenated_scans.T, '.')
        plt.gca().set_prop_cycle(None)
        plt.plot(concatenated_opds-popt_dispersed[0], curve.T, '-')
        plt.grid() 
        plt.xlabel('OPD (nm)', size=35)
        plt.ylabel('Intensity', size=35)
        plt.xticks(size=30);plt.yticks(size=30)
        plt.tight_layout()
        
        print('Non-spectrally dispersed')
        if kind_data == 'scan':
            wl = np.array([1600])
            bandwidth_binning = 200
        
        if fit_wavelength:
            init_guess2 = [delta0, 0, *np.ones(wl.size), *np.ones(wl.size), *np.zeros(wl.size)]
            bounds2 = [(delta_bounds[0], -2*np.pi, *np.ones(wl.size)*0.9, *np.ones(wl.size)*0.8, *np.ones(wl.size)*(-40)), \
                (delta_bounds[1], 2*np.pi, *np.ones(wl.size)*1.1, *np.ones(wl.size)*1.2, *np.ones(wl.size)*40)]
        else:
            init_guess2 = [delta0, 0, *np.ones(wl.size), *np.ones(wl.size)]
            bounds2 = [(delta_bounds[0], -2*np.pi, *np.ones(wl.size)*0.9, *np.zeros(wl.size)*0.8), \
                (delta_bounds[1], 2*np.pi, *np.ones(wl.size)*1.1, *np.ones(wl.size)*1.2)]
        
        popt2, pcov2 = curve_fit(modelsinminus, concatenated_opds, rtconcatenated_scans, p0 = init_guess2, bounds=bounds2)
        
        curve = modelsinminus(concatenated_opds, *popt2)
        chi2 = np.sum((rtconcatenated_scans - curve)**2) /(rtconcatenated_scans.size-popt2.size)
        print(popt2[:2])
        print(popt2[2:2+wl.size])
        print(popt2[2+wl.size:2+wl.size+wl.size])
        print(popt2[2+wl.size+wl.size:])
        print(chi2)
        
        curve = np.reshape(curve, (wl.size, -1))
        plt.figure(figsize=(19.32,10.8))
        plt.plot(concatenated_opds, rtconcatenated_scans, '.')
        plt.plot(concatenated_opds, curve.T, '-')
        plt.grid()
        plt.xlabel('OPD between the segments (nm)', size=35)
        plt.ylabel('Intensity', size=35)
        plt.xticks(size=30);plt.yticks(size=30)
        plt.tight_layout()
        
        plt.figure(figsize=(19.32,10.8))
        plt.plot(concatenated_opds-popt2[0], rtconcatenated_scans, '.')
        plt.plot(concatenated_opds-popt2[0], curve.T, '-')
        plt.grid()  
        plt.xlabel('OPD (nm)', size=35)
        plt.ylabel('Intensity', size=35)
        plt.xticks(size=30);plt.yticks(size=30)
        plt.tight_layout()

                
# =============================================================================
# Simple scan 
# =============================================================================
elif run_mode == 'simplescan':
    print('*** SIMPLE SCAN MODE ***')
    ''' Inputs '''
    datafolder = 'data202006/scans_20200601/'
    # datafolder = 'data202006/20200607/scans_lab_for_20200605/'
    #root = "C:/Users/marc-antoine/glint/"
    root = "/mnt/96980F95980F72D3/glint/"
    output_path = root+'GLINTprocessed/'+datafolder
    spectral_calibration_path = output_path
    geometric_calibration_path = output_path
    data_path = '/mnt/96980F95980F72D3/glint_data/'+datafolder
    data_list = [data_path+'null6_35at1.5_fullIms_20200601T192701.mat']
    pos_seg_ref = 1.5 # Piston value of the refererence segment in micron
    kind_data = 'scan'
    fit_wavelength = True
    delta_bounds = (-20000, -16000)
    delta0 = -18000
    
    rtscans, opds = processSimpleScan(data_list, pos_seg_ref)
    step = np.mean(np.diff(opds))
    
    rtscans_mean = rtscans.mean()
    freq = np.fft.fftfreq(rtscans.size, step)
    fft = np.fft.fft((rtscans-rtscans_mean)**2 * 2)
    fft0 = np.fft.fft(rtscans-rtscans_mean)
    
    wl = 1560
    fc = 1/wl
    lpfilter = lowpass_filter(freq, fc)
    filtered_fft = fft * lpfilter
    raw_envelop = np.fft.ifft(filtered_fft)**0.5
    
    plt.figure()
    plt.plot(freq, abs(fft))
    plt.plot(freq, abs(fft0))
    plt.plot(np.ones(2)*fc, [np.min(abs(fft)), np.max(abs(fft))])
    plt.grid()
    
    popt, pcov = curve_fit(envelope, opds, raw_envelop, p0=[delta0, 200, 1.], bounds=[(-np.inf, 100, 0.5), (np.inf, 250, 1.5)])
    print('Envelop', popt)
    plt.figure()
    plt.plot(opds, rtscans)
    # plt.plot(opds, (rtscans-rtscans_mean)**2*2)
    plt.plot(opds, raw_envelop+rtscans_mean)
    plt.plot(opds, envelope(opds, *popt)+rtscans_mean)
    plt.grid()
    
    
    if kind_data == 'scan':
        wl = np.array([1560])
        bandwidth_binning = 200
    
    init_guess = [delta0, 0, *np.ones(wl.size), *np.ones(wl.size), *np.zeros(wl.size)]
    bounds = [(delta_bounds[0], -np.pi*2, *np.zeros(wl.size), *np.zeros(wl.size), *np.ones(wl.size)*(-100)), \
              (delta_bounds[1], 2*np.pi, *np.ones(wl.size)*1.5, *np.ones(wl.size)*2, *np.ones(wl.size)*(100))]
    popt, pcov = curve_fit(modelsinminus, opds, rtscans, p0 = init_guess, bounds=bounds)
    
    print('')
    print('Non-spectrally dispersed')
    print(popt)
    curve = modelsinminus(opds, *popt)
    chi2 = np.sum((rtscans - curve)**2) /(rtscans.size-popt.size)
    print(chi2)
    
    plt.figure(figsize=(19.32,10.8))
    plt.plot(scanrange*1000, rtscans)
    plt.grid()
    plt.xlabel('Scan range (nm)', size=35)
    plt.ylabel('Intensity', size=35)
    plt.xticks(size=30);plt.yticks(size=30)
    plt.tight_layout()
    
    curve = np.reshape(curve, (wl.size, -1))

    plt.figure(figsize=(19.32,10.8))
    plt.plot(opds, rtscans, '.')
    plt.plot(opds, curve.T, '-')
    plt.grid()
    plt.xlabel('OPD between the segments (nm)', size=35)
    plt.ylabel('Intensity', size=35)
    plt.xticks(size=30);plt.yticks(size=30)
    plt.tight_layout()
    
    plt.figure(figsize=(19.32,10.8))
    plt.plot(opds-popt[0], rtscans, '.')
    plt.plot(opds-popt[0], curve.T, '-')
    plt.grid()  
    plt.xlabel('OPD (nm)', size=35)
    plt.ylabel('Intensity', size=35)
    plt.xticks(size=30);plt.yticks(size=30)
    plt.tight_layout()
    
else:
    print('*** Concatenate MODE ***')
    ''' Inputs '''
    datafolder = 'data202006/scans_20200531/'
    #root = "C:/Users/marc-antoine/glint/"
    root = "/mnt/96980F95980F72D3/glint/"
    output_path = root+'GLINTprocessed/'+datafolder
    spectral_calibration_path = output_path
    geometric_calibration_path = output_path
    data_path = '/mnt/96980F95980F72D3/glint_data/'+datafolder
    data_list = [data_path+'null5_29at2_20200601T000100.mat']+\
                [data_path+'null5_29at0_20200531T235948.mat']+\
                [data_path+'null5_29atm2_20200601T000234.mat']
    pos_seg_ref = [2, 0, -2]
    kind_data = 'scan'
    # delta0 = -10000
    # delta_bounds = (-12000, -8000)
    delta0 = -16000
    delta_bounds = (-17000, -16000)
    wl = np.array([1560])
    bandwidth_binning = 200
    
    scan_toconcatenate = []
    opd_toconcatenate = []
    for i in range(len(data_list)):
        rtscans, opds = processSimpleScan(data_list[i], pos_seg_ref[i])
        scan_toconcatenate.append(rtscans)
        opd_toconcatenate.append(opds)
        
    plt.figure()
    [plt.plot(opd_toconcatenate[i], scan_toconcatenate[i], '.-') for i in range(3)]
    plt.grid()
    
    # x = np.append(opd_toconcatenate[0][(opd_toconcatenate[0]>=-8000)&(opd_toconcatenate[0]<-800)],\
    #     opd_toconcatenate[1][(opd_toconcatenate[1]>=-800)&(opd_toconcatenate[1]<5000)])
    # y = np.append(scan_toconcatenate[0][(opd_toconcatenate[0]>=-8000)&(opd_toconcatenate[0]<-800)],\
    #     scan_toconcatenate[1][(opd_toconcatenate[1]>=-800)&(opd_toconcatenate[1]<5000)])
    x = np.append(opd_toconcatenate[0][(opd_toconcatenate[0]>=-8100)&(opd_toconcatenate[0]<200)],\
        opd_toconcatenate[2][(opd_toconcatenate[2]>=200)&(opd_toconcatenate[2]<7800)])
    y = np.append(scan_toconcatenate[0][(opd_toconcatenate[0]>=-8100)&(opd_toconcatenate[0]<200)],\
        scan_toconcatenate[2][(opd_toconcatenate[2]>=200)&(opd_toconcatenate[2]<7800)])
        
    plt.figure(figsize=(19.32,10.8))
    plt.plot(x, y)
    plt.grid()
    plt.xlabel('OPD between the segments (nm)', size=35)
    plt.ylabel('Intensity', size=35)
    plt.xticks(size=30);plt.yticks(size=30)
    plt.tight_layout()
   
    if fit_wavelength:
        init_guess = [delta0, bandwidth_binning, *np.ones(wl.size), *np.ones(wl.size), *np.zeros(wl.size)]
        bounds = [(delta_bounds[0], 50, *np.zeros(wl.size), *np.zeros(wl.size), *np.ones(wl.size)*(-100)), \
            (delta_bounds[1], 500, *np.ones(wl.size)*1.5, *np.ones(wl.size)*2, *np.ones(wl.size)*100)]
    else:
        init_guess = [delta0, bandwidth_binning, *np.ones(wl.size), *np.ones(wl.size)]
        bounds = [(delta_bounds[0], 50, *np.zeros(wl.size), *np.zeros(wl.size)), \
            (delta_bounds[1], 500, *np.ones(wl.size)*1.5, *np.ones(wl.size)*2)]

    popt, pcov = curve_fit(modelsinminus2, x, y, p0 = init_guess, bounds=bounds)
    print(popt)
    curve = modelsinminus2(x, *popt)
    chi2 = np.sum((y - curve)**2) /(y.size-popt.size)
    print(chi2)
    
    big_opd = np.linspace(0, 100000, 1000, endpoint=False)
    big_curve = np.sinc(big_opd / (wl[0]**2/popt[1]))*0.5 + 1
    plt.figure(figsize=(19.32,10.8))
    plt.plot(x-popt[0], y, '.')
    plt.plot(x-popt[0], curve, '-')
    plt.plot(big_opd, big_curve)
    plt.grid()  
    plt.xlabel('OPD (nm)', size=35)
    plt.ylabel('Intensity', size=35)
    plt.xticks(size=30);plt.yticks(size=30)
    plt.tight_layout()    