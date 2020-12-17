#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script gives the conversion coefficient between the null/antinull outputs and the photometric ones, for every spectral channels.
It also estimate the splitting and coupling coefficients for characterization purpose.
The data to process consists in files acquired when each beam is active one after the other.
The name of the file must contain the keyword ``pX`` where X is the id of the beam (1 to 4).

The script concatenate all the files of a given active beam into one frame then extract the intensities into everey output.
The zeta coefficient (conversion factor) are computed by doing the ratio of the interferometric output over the photometric one.

**Nota Bene**: one spectral channel consists in one column of pixel. 
The whole width of the frame is used, including the part with no signal.
Consequently, some coefficients (zeta, splitting or coupling) are absurd.

The inputs are set in the ``Inputs`` section:
    * **datafolder**: string, folder containing the files to process for all beams
    * **root**: string, path to the root folder containing all the subfolders containing all the products needed
    * **calibration_path**: string, folder containing the calibration data (spectral and geometric calibrations)
    * **output_path**: string, path to the folder where the products of the script are saved
    
The settings are in the ``Settings`` section:
    * **no_noise**: bool, ``True`` if data is noise-free (e.g. comes from a simulation)
    * **nb_img**: tuple, bounds between frames are selected. Leave ``None`` to start from the first frame or to finish to the last one (included).
    * **debug**: bool, set to ``True`` to check if the method ``getSpectralFlux`` correctly behaves (e.g. good geometric calibration). It is strongly adviced to change **nb_img** to only process one frame.
    * **save**: bool, set to ``True`` to save the zeta coefficient.
    
The outputs are:
    * Some plots for characterization and monitoring purpose, they are not automatically saved.
    * An HDF5 file containing the zeta coefficient.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glint_classes
import warnings

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
    warnings.filterwarnings(action="ignore", category=np.VisibleDeprecationWarning)
    
    ''' Settings '''
    no_noise = False
    nb_img = (None, None)
    debug = False
    save = True
    mode_flux = 'raw'
    suffix = ''
    spectral_binning = False
    wl_bin_min, wl_bin_max = 1525, 1575# In nm
    bandwidth_binning = 50 # In nm

    ''' Inputs '''
    datafolder = 'data202012/20201209/zeta/'
    root = "/mnt/96980F95980F72D3/glint/"
    # root = "//tintagel.physics.usyd.edu.au/snert/"
    output_path = root+'GLINTprocessed/'+datafolder
    spectral_calibration_path = output_path
    geometric_calibration_path = output_path
    data_path = '/mnt/96980F95980F72D3/glint_data/'+datafolder
    # data_path = '//tintagel.physics.usyd.edu.au/snert/GLINTData/'+datafolder
    dark = np.load(output_path+'superdark.npy')
    
    Iminus = []
    Iplus = []
    P1, P2, P3, P4 = 0, 0, 0, 0
    zeta_coeff = {}
    
    for beam in range(1,5):
        data_list = [data_path+f for f in os.listdir(data_path) if 'p'+str(beam) in f]
        
        ''' Output '''
        output_path = root+'GLINTprocessed/'+datafolder
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        dark = np.load(output_path+'superdark.npy')
        dark_per_channel = np.load(output_path+'superdarkchannel.npy')
        if no_noise:
            dark_per_channel[:] = 0.
        
        ''' Set processing configuration and load instrumental calibration data '''
        nb_tracks = 16 # Number of tracks
        which_tracks = np.arange(16) # Tracks to process
        # coeff_pos = np.load(geometric_calibration_path+'20200130_coeff_position_poly.npy')
        # coeff_width = np.load(geometric_calibration_path+'20200130_coeff_width_poly.npy')
        # position_poly = [np.poly1d(coeff_pos[i]) for i in range(nb_tracks)]
        # width_poly = [np.poly1d(coeff_width[i]) for i in range(nb_tracks)]
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
                
        ''' Start the data processing '''
        superData = np.zeros((344,96))
        superNbImg = 0
        for f in data_list:
            print("Process of : %s (%d / %d)" %(f, data_list.index(f)+1, len(data_list)))
            img = glint_classes.ChipProperties(f, nbimg=nb_img)
            print(img.data[:,:,10].mean())
    
            superData = superData + img.data.sum(axis=0)
            superNbImg += img.nbimg
    
        superData = superData / superNbImg
        
        plt.figure()
        plt.imshow(superData-dark, interpolation='none')
        plt.colorbar()
        plt.title('P'+str(beam)+' on')
    
        img2 = glint_classes.ChipProperties(nbimg=(0,1))
        img2.data = np.reshape(superData, (1,superData.shape[0], superData.shape[1]))
        
        img2.cosmeticsFrames(np.zeros(dark.shape), no_noise)
        
        ''' Insulating each track '''
        print('Getting channels')
        img2.getChannels(channel_pos, sep, spatial_axis, dark=dark_per_channel)
        
        ''' Map the spectral channels between every chosen tracks before computing 
        the null depth'''
        img2.matchSpectralChannels(wl_to_px_coeff, px_to_wl_coeff)
        
        ''' Measurement of flux per frame, per spectral channel, per track '''
        list_channels = np.arange(16) #[1,3,4,5,6,7,8,9,10,11,12,14]
        img2.getSpectralFlux(spectral_axis, position_outputs, width_outputs, mode_flux, debug=debug)
        img2.getIntensities(mode_flux)

        if spectral_binning:
            img2.spectralBinning(wl_bin_min, wl_bin_max, bandwidth_binning, wl_to_px_coeff)        
        
        ''' Get split and coupler coefficient, biased with transmission coeff between nulling-chip and detector '''
        img2.getRatioCoeff(beam, zeta_coeff)
        
        Iminus.append([img2.Iminus1[0], img2.Iminus2[0], img2.Iminus3[0], img2.Iminus4[0], img2.Iminus5[0], img2.Iminus6[0]])
        Iplus.append([img2.Iplus1[0], img2.Iplus2[0], img2.Iplus3[0], img2.Iplus4[0], img2.Iplus5[0], img2.Iplus6[0]])
        
        if beam == 1:
            P1 = img2.p1[0]
        if beam == 2:
            P2 = img2.p2[0]
        if beam == 3:
            P3 = img2.p3[0]
        if beam == 4:
            P4 = img2.p4[0]
        
    #    plt.figure(figsize=(19.20,10.80))
    #    plt.suptitle('P%s on'%beam)
    #    plt.subplot(4,4,1)
    #    plt.plot(img2.wl_scale[0], img2.p1[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.p1[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('P1')
    #    plt.subplot(4,4,2)
    #    plt.plot(img2.wl_scale[0], img2.p2[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.p2[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('P2')
    #    plt.subplot(4,4,3)
    #    plt.plot(img2.wl_scale[0], img2.p3[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.p3[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('P3')
    #    plt.subplot(4,4,4)
    #    plt.plot(img2.wl_scale[0], img2.p4[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.p4[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('P4')
    #    
    #    plt.subplot(4,4,5)
    #    plt.plot(img2.wl_scale[0], img2.Iminus1[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.Iminus1[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('N1 (12)')
    #    plt.subplot(4,4,6)
    #    plt.plot(img2.wl_scale[0], img2.Iminus2[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.Iminus2[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('N2 (23)')
    #    plt.subplot(4,4,7)
    #    plt.plot(img2.wl_scale[0], img2.Iminus3[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.Iminus3[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('N3 (14)')
    #    plt.subplot(4,4,8)
    #    plt.plot(img2.wl_scale[0], img2.Iminus4[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.Iminus4[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('N4 (34)')
    #    plt.subplot(4,4,9)
    #    plt.plot(img2.wl_scale[0], img2.Iminus5[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.Iminus5[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('N5 (13)')
    #    plt.subplot(4,4,10)
    #    plt.plot(img2.wl_scale[0], img2.Iminus6[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.Iminus6[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('N6 (24)')
    #    
    #    plt.subplot(4,4,11)
    #    plt.plot(img2.wl_scale[0], img2.Iplus1[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.Iplus1[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('AN1 (12)')
    #    plt.subplot(4,4,12)
    #    plt.plot(img2.wl_scale[0], img2.Iplus2[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.Iplus2[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('AN2 (23)')
    #    plt.subplot(4,4,13)
    #    plt.plot(img2.wl_scale[0], img2.Iplus3[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.Iplus3[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('AN3 (14)')
    #    plt.subplot(4,4,14)
    #    plt.plot(img2.wl_scale[0], img2.Iplus4[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.Iplus4[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('AN4 (34)')
    #    plt.subplot(4,4,15)
    #    plt.plot(img2.wl_scale[0], img2.Iplus5[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.Iplus5[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('AN5 (13)')
    #    plt.subplot(4,4,16)
    #    plt.plot(img2.wl_scale[0], img2.Iplus6[0])
    #    plt.grid()
    #    plt.ylim(-1)
    #    if np.max(np.abs(img2.Iplus6[0])) > 1500: plt.ylim(-1, 1500)
    #    plt.title('AN6 (24)')
    
        plt.figure(figsize=(19.20,10.80))
    #    plt.suptitle('P%s on'%beam)
        plt.subplot(3,4,1)
        plt.title('P1')
        plt.plot(img2.wl_scale[0], img2.p1[0])
        plt.grid()
        plt.ylim(-1)
        if np.max(img2.p1[0])>5000: plt.ylim(ymax=5000)
#        if np.max(np.abs(img2.p1[0])) > 1500: plt.ylim(-1, 1500)
        plt.ylabel('Intensity (AU)')
        plt.xlabel('Wavelength (nm)')
        plt.subplot(3,4,2)
        plt.title('P2')
        plt.plot(img2.wl_scale[0], img2.p2[0])
        plt.grid()
        plt.ylim(-1)
        if np.max(img2.p2[0])>5000: plt.ylim(ymax=5000)
#        if np.max(np.abs(img2.p2[0])) > 1500: plt.ylim(-1, 1500)
        plt.ylabel('Intensity (AU)')
        plt.xlabel('Wavelength (nm)')
        plt.subplot(3,4,3)
        plt.title('P3')
        plt.plot(img2.wl_scale[0], img2.p3[0])
        plt.grid()
        plt.ylim(-1)
        if np.max(img2.p3[0])>5000: plt.ylim(ymax=5000)
#        if np.max(np.abs(img2.p3[0])) > 1500: plt.ylim(-1, 1500)
        plt.ylabel('Intensity (AU)')
        plt.xlabel('Wavelength (nm)')
        plt.subplot(3,4,4)
        plt.title('P4')
        plt.plot(img2.wl_scale[0], img2.p4[0])
        plt.grid()
        plt.ylim(-1)
        if np.max(img2.p4[0])>5000: plt.ylim(ymax=5000)
#        if np.max(np.abs(img2.p4[0])) > 1500: plt.ylim(-1, 1500)
        plt.ylabel('Intensity (AU)')
        plt.xlabel('Wavelength (nm)')
        plt.subplot(3,4,5)
        plt.title('N1 and N7 (12)')
        plt.plot(img2.wl_scale[0], img2.Iminus1[0])
        plt.plot(img2.wl_scale[0], img2.Iplus1[0])
        plt.grid()
        plt.ylim(-1)
        if np.max(img2.Iminus1[0])>5000 or np.max(img2.Iplus1[0])>5000: plt.ylim(ymax=5000)
#        if np.max(np.abs(img2.Iplus1[0])) > 1500 or np.max(np.abs(img2.Iminus1[0])) > 1500: plt.ylim(-1, 1500)
        plt.ylabel('Intensity (AU)')
        plt.xlabel('Wavelength (nm)')
        plt.subplot(3,4,6)
        plt.title('N2 and N8 (23)')
        plt.plot(img2.wl_scale[0], img2.Iminus2[0])
        plt.plot(img2.wl_scale[0], img2.Iplus2[0])
        plt.grid()
        plt.ylim(-1)
        if np.max(img2.Iminus2[0])>5000 or np.max(img2.Iplus2[0])>5000: plt.ylim(ymax=5000)
#        if np.max(np.abs(img2.Iplus2[0])) > 1500 or np.max(np.abs(img2.Iminus2[0])) > 1500: plt.ylim(-1, 1500)
        plt.ylabel('Intensity (AU)')
        plt.xlabel('Wavelength (nm)')
        plt.subplot(3,4,7)
        plt.title('N3 and N9 (14)')
        plt.plot(img2.wl_scale[0], img2.Iminus3[0])
        plt.plot(img2.wl_scale[0], img2.Iplus3[0])
#        if np.max(np.abs(img2.Iplus3[0])) > 1500 or np.max(np.abs(img2.Iminus3[0])) > 1500: plt.ylim(-1, 1500)
        plt.grid()
        plt.ylim(-1)
        if np.max(img2.Iminus3[0])>5000 or np.max(img2.Iplus3[0])>5000: plt.ylim(ymax=5000)
        plt.ylabel('Intensity (AU)')
        plt.xlabel('Wavelength (nm)')
        plt.subplot(3,4,8)
        plt.title('N4 and N10 (34)')
        plt.plot(img2.wl_scale[0], img2.Iminus4[0])
        plt.plot(img2.wl_scale[0], img2.Iplus4[0])
        plt.grid()
        plt.ylim(-1)
#        if np.max(np.abs(img2.Iplus4[0])) > 1500 or np.max(np.abs(img2.Iminus4[0])) > 1500: plt.ylim(-1, 1500)
        plt.ylabel('Intensity (AU)')
        plt.xlabel('Wavelength (nm)')
        plt.subplot(3,4,9)
        plt.title('N5 and N11 (13)')
        plt.plot(img2.wl_scale[0], img2.Iminus5[0])
        plt.plot(img2.wl_scale[0], img2.Iplus5[0])
        plt.grid()
        plt.ylim(-1)
        if np.max(img2.Iminus5[0])>5000 or np.max(img2.Iplus5[0])>5000: plt.ylim(ymax=5000)
#        if np.max(np.abs(img2.Iplus5[0])) > 1500 or np.max(np.abs(img2.Iminus5[0])) > 1500: plt.ylim(-1, 1500)
        plt.ylabel('Intensity (AU)')
        plt.xlabel('Wavelength (nm)')
        plt.subplot(3,4,10)
        plt.title('N6 and N12 (24)')    
        plt.plot(img2.wl_scale[0], img2.Iminus6[0])
        plt.plot(img2.wl_scale[0], img2.Iplus6[0])
        plt.grid()
        plt.ylim(-1)
        if np.max(img2.Iminus6[0])>5000 or np.max(img2.Iplus6[0])>5000: plt.ylim(ymax=5000)
#        if np.max(np.abs(img2.Iplus6[0])) > 1500 or np.max(np.abs(img2.Iminus6[0])) > 1500: plt.ylim(-1, 1500)
        plt.ylabel('Intensity (AU)')
        plt.xlabel('Wavelength (nm)')
        plt.tight_layout()
        if save: plt.savefig(output_path+'fluxes_p%s'%(beam)+suffix+'.png')
        
    keys = np.array(list(zeta_coeff.keys()))
    keys_title = np.array([elt[0].upper()+'eam '+elt[1]+' to '+elt[2:6].capitalize()+' '+elt[6:] for elt in keys]).reshape(4,6)
    keys = keys.reshape(4,6)
    
    fig = plt.figure(figsize=(19.20,10.80))
    grid = plt.GridSpec(4, 6, wspace=0.2, hspace=0.4, left=0.03, bottom=0.05, right=0.98, top=0.92)
    plt.suptitle('Zeta coefficients')
    for i in range(4):
        for j in range(6):
            fig.add_subplot(grid[i,j])
            plt.plot(img2.wl_scale[0], zeta_coeff[keys[i,j]][0])
            plt.grid()
            plt.title(keys_title[i,j])
            if i == 3: plt.xlabel('Wavelength (nm)')
            if j == 0: plt.ylabel(r'$\zeta$ coeff')
            plt.ylim(-0.2,5)
    plt.tight_layout()
    if save: plt.savefig(output_path+'zeta_coeff'+suffix+'.png')
            
    Iplus = np.array(Iplus)
    Iminus = np.array(Iminus)
    
    tan2 = {'N1 (12)':(Iminus[0,0]/Iplus[0,0], Iminus[1,0]/Iplus[1,0]),
           'N2 (23)':(Iminus[1,1]/Iplus[1,1], Iminus[2,1]/Iplus[2,1]),
           'N3 (14)':(Iminus[0,2]/Iplus[0,2], Iminus[3,2]/Iplus[3,2]),
           'N4 (34)':(Iminus[2,3]/Iplus[2,3], Iminus[3,3]/Iplus[3,3]),
           'N5 (13)':(Iminus[0,4]/Iplus[0,4], Iminus[2,4]/Iplus[2,4]),
           'N6 (24)':(Iminus[1,5]/Iplus[1,5], Iminus[3,5]/Iplus[3,5])}
    
    plt.figure(figsize=(19.20,10.80))
    plt.subplot(2,3,1)
    plt.title('Coupling ratio for N1 (12)')
    plt.plot(img2.wl_scale[0], Iminus[0,0]/(Iminus[0,0]+Iplus[0,0]), label='Beam 1 to N1')
    plt.plot(img2.wl_scale[0], Iplus[0,0]/(Iminus[0,0]+Iplus[0,0]), label='Beam 1 to AN1')
    plt.plot(img2.wl_scale[0], Iminus[1,0]/(Iminus[1,0]+Iplus[1,0]), label='Beam 2 to N1')
    plt.plot(img2.wl_scale[0], Iplus[1,0]/(Iminus[1,0]+Iplus[1,0]), label='Beam 2 to AN1')
    plt.grid()
    plt.legend(loc='center left')
    plt.ylim(0,1.02)
    plt.xlim(1250)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Coupling ratio')
    plt.subplot(2,3,2)
    plt.title('Coupling ratio for N2 (23)')
    plt.plot(img2.wl_scale[0], Iminus[1,1]/(Iminus[1,1]+Iplus[1,1]), label='Beam 2 to N2')
    plt.plot(img2.wl_scale[0], Iplus[1,1]/(Iminus[1,1]+Iplus[1,1]), label='Beam 2 to AN2')
    plt.plot(img2.wl_scale[0], Iminus[2,1]/(Iminus[2,1]+Iplus[2,1]), label='Beam 3 to N2')
    plt.plot(img2.wl_scale[0], Iplus[2,1]/(Iminus[2,1]+Iplus[2,1]), label='Beam 3 to AN2')
    plt.grid()
    plt.legend(loc='center left')
    plt.ylim(0,1.02)
    plt.xlim(1250)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Coupling ratio')
    plt.subplot(2,3,3)
    plt.title('Coupling ratio for N3 (14)')
    plt.plot(img2.wl_scale[0], Iminus[0,2]/(Iminus[0,2]+Iplus[0,2]), label='Beam 1 to N3')
    plt.plot(img2.wl_scale[0], Iplus[0,2]/(Iminus[0,2]+Iplus[0,2]), label='Beam 1 to AN3')
    plt.plot(img2.wl_scale[0], Iminus[3,2]/(Iminus[3,2]+Iplus[3,2]), label='Beam 4 to N3')
    plt.plot(img2.wl_scale[0], Iplus[3,2]/(Iminus[3,2]+Iplus[3,2]), label='Beam 4 to AN3')
    plt.grid()
    plt.legend(loc='center left')
    plt.ylim(0,1.02)
    plt.xlim(1250)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Coupling ratio')
    plt.subplot(2,3,4)
    plt.title('Coupling ratio for N4 (34)')
    plt.plot(img2.wl_scale[0], Iminus[2,3]/(Iminus[2,3]+Iplus[2,3]), label='Beam 3 to N4')
    plt.plot(img2.wl_scale[0], Iplus[2,3]/(Iminus[2,3]+Iplus[2,3]), label='Beam 3 to AN4')
    plt.plot(img2.wl_scale[0], Iminus[3,3]/(Iminus[3,3]+Iplus[3,3]), label='Beam 4 to N4')
    plt.plot(img2.wl_scale[0], Iplus[3,3]/(Iminus[3,3]+Iplus[3,3]), label='Beam 4 to AN4')
    plt.grid()
    plt.legend(loc='center left')
    plt.ylim(0,1.02)
    plt.xlim(1250)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Coupling ratio')
    plt.subplot(2,3,5)
    plt.title('Coupling ratio for N5 (13)')
    plt.plot(img2.wl_scale[0], Iminus[0,4]/(Iminus[0,4]+Iplus[0,4]), label='Beam 1 to N5')
    plt.plot(img2.wl_scale[0], Iplus[0,4]/(Iminus[0,4]+Iplus[0,4]), label='Beam 1 to AN5')
    plt.plot(img2.wl_scale[0], Iminus[2,4]/(Iminus[2,4]+Iplus[2,4]), label='Beam 3 to N5')
    plt.plot(img2.wl_scale[0], Iplus[2,4]/(Iminus[2,4]+Iplus[2,4]), label='Beam 3 to AN5')
    plt.grid()
    plt.legend(loc='center left')
    plt.ylim(0,1.02)
    plt.xlim(1250)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Coupling ratio')
    plt.subplot(2,3,6)
    plt.title('Coupling ratio for N6 (24)')
    plt.plot(img2.wl_scale[0], Iminus[1,5]/(Iminus[1,5]+Iplus[1,5]), label='Beam 2 to N6')
    plt.plot(img2.wl_scale[0], Iplus[1,5]/(Iminus[1,5]+Iplus[1,5]), label='Beam 2 to AN6')
    plt.plot(img2.wl_scale[0], Iminus[3,5]/(Iminus[3,5]+Iplus[3,5]), label='Beam 4 to N6')
    plt.plot(img2.wl_scale[0], Iplus[3,5]/(Iminus[3,5]+Iplus[3,5]), label='Beam 4 to AN6')
    plt.grid()
    plt.legend(loc='center left')
    plt.ylim(0,1.02)
    plt.xlim(1250)
    plt.tight_layout()
    if save: plt.savefig(output_path+'coupling_ratios'+suffix+'.png')    
    
    plt.figure(figsize=(19.20,10.80))
    plt.subplot(2,2,1)
    plt.title('Splitting ratio for Beam 1')
    plt.plot(img2.wl_scale[0], P1/(P1 + Iminus[0,0] + Iplus[0,0] + Iminus[0,2] + Iplus[0,2] + Iminus[0,4] + Iplus[0,4]), label='Photometry')
    plt.plot(img2.wl_scale[0], Iminus[0,0]/(P1 + Iminus[0,0] + Iplus[0,0] + Iminus[0,2] + Iplus[0,2] + Iminus[0,4] + Iplus[0,4]), label='N1')
    plt.plot(img2.wl_scale[0], Iplus[0,0]/(P1 + Iminus[0,0] + Iplus[0,0] + Iminus[0,2] + Iplus[0,2] + Iminus[0,4] + Iplus[0,4]), label='AN1')
    plt.plot(img2.wl_scale[0], Iminus[0,2]/(P1 + Iminus[0,0] + Iplus[0,0] + Iminus[0,2] + Iplus[0,2] + Iminus[0,4] + Iplus[0,4]), label='N3')
    plt.plot(img2.wl_scale[0], Iplus[0,2]/(P1 + Iminus[0,0] + Iplus[0,0] + Iminus[0,2] + Iplus[0,2] + Iminus[0,4] + Iplus[0,4]), label='AN3')
    plt.plot(img2.wl_scale[0], Iminus[0,4]/(P1 + Iminus[0,0] + Iplus[0,0] + Iminus[0,2] + Iplus[0,2] + Iminus[0,4] + Iplus[0,4]), label='N5')
    plt.plot(img2.wl_scale[0], Iplus[0,4]/(P1 + Iminus[0,0] + Iplus[0,0] + Iminus[0,2] + Iplus[0,2] + Iminus[0,4] + Iplus[0,4]), label='AN5')
    plt.grid()
    plt.ylim(0,1.02)
    plt.xlim(1250)
    plt.legend(loc='best')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Splitting ratio')
    plt.subplot(2,2,2)
    plt.title('Splitting ratio for Beam 2')
    plt.plot(img2.wl_scale[0], P2/(P2 + Iminus[1,0] + Iplus[1,0] + Iminus[1,1] + Iplus[1,1] + Iminus[1,5] + Iplus[1,5]), label='Photometry')
    plt.plot(img2.wl_scale[0], Iminus[1,0]/(P2 + Iminus[1,0] + Iplus[1,0] + Iminus[1,1] + Iplus[1,1] + Iminus[1,5] + Iplus[1,5]), label='N1')
    plt.plot(img2.wl_scale[0], Iplus[1,0]/(P2 + Iminus[1,0] + Iplus[1,0] + Iminus[1,1] + Iplus[1,1] + Iminus[1,5] + Iplus[1,5]), label='AN1')
    plt.plot(img2.wl_scale[0], Iminus[1,1]/(P2 + Iminus[1,0] + Iplus[1,0] + Iminus[1,1] + Iplus[1,1] + Iminus[1,5] + Iplus[1,5]), label='N2')
    plt.plot(img2.wl_scale[0], Iplus[1,1]/(P2 + Iminus[1,0] + Iplus[1,0] + Iminus[1,1] + Iplus[1,1] + Iminus[1,5] + Iplus[1,5]), label='AN2')
    plt.plot(img2.wl_scale[0], Iminus[1,5]/(P2 + Iminus[1,0] + Iplus[1,0] + Iminus[1,1] + Iplus[1,1] + Iminus[1,5] + Iplus[1,5]), label='N6')
    plt.plot(img2.wl_scale[0], Iplus[1,5]/(P2 + Iminus[1,0] + Iplus[1,0] + Iminus[1,1] + Iplus[1,1] + Iminus[1,5] + Iplus[1,5]), label='AN6')
    plt.grid()
    plt.ylim(0,1.02)
    plt.xlim(1250)
    plt.legend(loc='best')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Splitting ratio')
    plt.subplot(2,2,3)
    plt.title('Splitting ratio for Beam 3')
    plt.plot(img2.wl_scale[0], P3/(P3 + Iminus[2,1] + Iplus[2,1] + Iminus[2,3] + Iplus[2,3] + Iminus[2,4] + Iplus[2,4]), label='Photometry')
    plt.plot(img2.wl_scale[0], Iminus[2,1]/(P3 + Iminus[2,1] + Iplus[2,1] + Iminus[2,3] + Iplus[2,3] + Iminus[2,4] + Iplus[2,4]), label='N2')
    plt.plot(img2.wl_scale[0], Iplus[2,1]/(P3 + Iminus[2,1] + Iplus[2,1] + Iminus[2,3] + Iplus[2,3] + Iminus[2,4] + Iplus[2,4]), label='AN2')
    plt.plot(img2.wl_scale[0], Iminus[2,3]/(P3 + Iminus[2,1] + Iplus[2,1] + Iminus[2,3] + Iplus[2,3] + Iminus[2,4] + Iplus[2,4]), label='N4')
    plt.plot(img2.wl_scale[0], Iplus[2,3]/(P3 + Iminus[2,1] + Iplus[2,1] + Iminus[2,3] + Iplus[2,3] + Iminus[2,4] + Iplus[2,4]), label='AN4')
    plt.plot(img2.wl_scale[0], Iminus[2,4]/(P3 + Iminus[2,1] + Iplus[2,1] + Iminus[2,3] + Iplus[2,3] + Iminus[2,4] + Iplus[2,4]), label='N5')
    plt.plot(img2.wl_scale[0], Iplus[2,4]/(P3 + Iminus[2,1] + Iplus[2,1] + Iminus[2,3] + Iplus[2,3] + Iminus[2,4] + Iplus[2,4]), label='AN5')
    plt.grid()
    plt.ylim(0,1.02)
    plt.xlim(1250)
    plt.legend(loc='best')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Splitting ratio')
    plt.subplot(2,2,4)
    plt.title('Splitting ratio for Beam 4')
    plt.plot(img2.wl_scale[0], P4/(P4 + Iminus[3,2] + Iplus[3,2] + Iminus[3,5] + Iplus[3,5] + Iminus[3,3] + Iplus[3,3]), label='Photometry')
    plt.plot(img2.wl_scale[0], Iminus[3,2]/(P4 + Iminus[3,2] + Iplus[3,2] + Iminus[3,5] + Iplus[3,5] + Iminus[3,3] + Iplus[3,3]), label='N3')
    plt.plot(img2.wl_scale[0], Iplus[3,2]/(P4 + Iminus[3,2] + Iplus[3,2] + Iminus[3,5] + Iplus[3,5] + Iminus[3,3] + Iplus[3,3]), label='AN3')
    plt.plot(img2.wl_scale[0], Iminus[3,5]/(P4 + Iminus[3,2] + Iplus[3,2] + Iminus[3,5] + Iplus[3,5] + Iminus[3,3] + Iplus[3,3]), label='N6')
    plt.plot(img2.wl_scale[0], Iplus[3,5]/(P4 + Iminus[3,2] + Iplus[3,2] + Iminus[3,5] + Iplus[3,5] + Iminus[3,3] + Iplus[3,3]), label='AN6')
    plt.plot(img2.wl_scale[0], Iminus[3,3]/(P4 + Iminus[3,2] + Iplus[3,2] + Iminus[3,5] + Iplus[3,5] + Iminus[3,3] + Iplus[3,3]), label='N4')
    plt.plot(img2.wl_scale[0], Iplus[3,3]/(P4 + Iminus[3,2] + Iplus[3,2] + Iminus[3,5] + Iplus[3,5] + Iminus[3,3] + Iplus[3,3]), label='AN4')
    plt.grid()
    plt.ylim(0,1.02)
    plt.xlim(1250)
    plt.legend(loc='best')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Splitting ratio')
    plt.tight_layout()
    if save: plt.savefig(output_path+'splitting_ratios'+suffix+'.png')
    
    #plt.figure(figsize=(19.20,10.80))
    #plt.subplot(2,3,1)
    #plt.plot(img2.wl_scale[0], np.arctan2(tan2['N1 (12)'][0]**0.25, tan2['N1 (12)'][1]**0.25))
    #plt.grid()
    #plt.title('coupling coeff for N1 (12) (arctan)')
    #plt.subplot(2,3,2)
    #plt.plot(img2.wl_scale[0], np.arctan2(tan2['N2 (23)'][0]**0.25, tan2['N2 (23)'][1]**0.25))
    #plt.grid()
    #plt.title('tan^2 or 1/tan^2 of coupling coeff for N2 (23)')
    #plt.subplot(2,3,3)
    #plt.plot(img2.wl_scale[0], np.arctan2(tan2['N3 (14)'][0]**0.25, tan2['N3 (14)'][1]**0.25))
    #plt.grid()
    #plt.title('tan^2 or 1/tan^2 of coupling coeff for N3 (14)')
    #plt.subplot(2,3,4)
    #plt.plot(img2.wl_scale[0], np.arctan2(tan2['N4 (34)'][0]**0.25, tan2['N4 (34)'][1]**0.25))
    #plt.grid()
    #plt.title('tan^2 or 1/tan^2 of coupling coeff for N4 (34)')
    #plt.subplot(2,3,5)
    #plt.plot(img2.wl_scale[0], np.arctan2(tan2['N5 (13)'][0]**0.25, tan2['N5 (13)'][1]**0.25))
    #plt.grid()
    #plt.title('tan^2 or 1/tan^2 of coupling coeff for N5 (13)')
    #plt.subplot(2,3,6)
    #plt.plot(img2.wl_scale[0], np.arctan2(tan2['N6 (24)'][0]**0.25, tan2['N6 (24)'][1]**0.25))
    #plt.grid()
    #plt.title('tan^2 or 1/tan^2 of coupling coeff for N6 (24)')
    #
    #plt.figure(figsize=(19.20,10.80))
    #plt.subplot(2,3,1)
    #plt.plot(img2.wl_scale[0], np.arctan2(tan2['N1 (12)'][1]**0.25, tan2['N1 (12)'][0]**0.25))
    #plt.grid()
    #plt.title('coupling coeff for N1 (12) (arctan)')
    #plt.subplot(2,3,2)
    #plt.plot(img2.wl_scale[0], np.arctan2(tan2['N2 (23)'][1]**0.25, tan2['N2 (23)'][0]**0.25))
    #plt.grid()
    #plt.title('tan^2 or 1/tan^2 of coupling coeff for N2 (23)')
    #plt.subplot(2,3,3)
    #plt.plot(img2.wl_scale[0], np.arctan2(tan2['N3 (14)'][1]**0.25, tan2['N3 (14)'][0]**0.25))
    #plt.grid()
    #plt.title('tan^2 or 1/tan^2 of coupling coeff for N3 (14)')
    #plt.subplot(2,3,4)
    #plt.plot(img2.wl_scale[0], np.arctan2(tan2['N4 (34)'][1]**0.25, tan2['N4 (34)'][0]**0.25))
    #plt.grid()
    #plt.title('tan^2 or 1/tan^2 of coupling coeff for N4 (34)')
    #plt.subplot(2,3,5)
    #plt.plot(img2.wl_scale[0], np.arctan2(tan2['N5 (13)'][1]**0.25, tan2['N5 (13)'][0]**0.25))
    #plt.grid()
    #plt.title('tan^2 or 1/tan^2 of coupling coeff for N5 (13)')
    #plt.subplot(2,3,6)
    #plt.plot(img2.wl_scale[0], np.arctan2(tan2['N6 (24)'][1]**0.25, tan2['N6 (24)'][0]**0.25))
    #plt.grid()
    #plt.title('tan^2 or 1/tan^2 of coupling coeff for N6 (24)')
    
    if save:
        import h5py
        with h5py.File(output_path+'/zeta_coeff_'+mode_flux+suffix+'.hdf5', 'w') as f:
            f.create_dataset('wl_scale', data=img2.wl_scale.mean(axis=0))
            f['wl_scale'].attrs['comment'] = 'wl in nm'
            for key in zeta_coeff.keys():
                f.create_dataset(key, data=zeta_coeff[key][0])
                
    masque = (img2.wl_scale[0]>=1300)&(img2.wl_scale[0]<=1650)
    longueuronde = img2.wl_scale[0][masque]      
    a = P1/(P1 + Iminus[0,0] + Iplus[0,0] + Iminus[0,2] + Iplus[0,2] + Iminus[0,4] + Iplus[0,4])
    b = Iminus[0,0]/(P1 + Iminus[0,0] + Iplus[0,0] + Iminus[0,2] + Iplus[0,2] + Iminus[0,4] + Iplus[0,4])
    c = Iplus[0,0]/(P1 + Iminus[0,0] + Iplus[0,0] + Iminus[0,2] + Iplus[0,2] + Iminus[0,4] + Iplus[0,4])
    d = Iminus[0,2]/(P1 + Iminus[0,0] + Iplus[0,0] + Iminus[0,2] + Iplus[0,2] + Iminus[0,4] + Iplus[0,4])
    e = Iplus[0,2]/(P1 + Iminus[0,0] + Iplus[0,0] + Iminus[0,2] + Iplus[0,2] + Iminus[0,4] + Iplus[0,4])
    f = Iminus[0,4]/(P1 + Iminus[0,0] + Iplus[0,0] + Iminus[0,2] + Iplus[0,2] + Iminus[0,4] + Iplus[0,4])
    g = Iplus[0,4]/(P1 + Iminus[0,0] + Iplus[0,0] + Iminus[0,2] + Iplus[0,2] + Iminus[0,4] + Iplus[0,4])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # plt.figure(figsize=(19.20,10.80))
    # # plt.subplot(2,1,1)
    # plt.title('Splitting ratio for beam 1', size=40)
    # plt.plot(longueuronde, a[masque], lw=6, c='k', label='To photometric output')
    # plt.plot(longueuronde, b[masque], '-', lw=4, c=colors[0], label='To coupler of null 1')
    # plt.plot(longueuronde, c[masque], '--', lw=4, c=colors[0], label='To antinull 1')
    # plt.plot(longueuronde, d[masque], '--', lw=4, c=colors[1], label='To coupler of null 3')
    # plt.plot(longueuronde, e[masque], '-.', lw=4, c=colors[1], label='To antinull 3')
    # plt.plot(longueuronde, f[masque], '-.', lw=4, c=colors[2], label='To coupler of null 5')
    # plt.plot(longueuronde, g[masque], '.', lw=4, c=colors[2], label='To antinull 5')
    # plt.grid()
    # plt.ylim(0,0.6)
    # # plt.xlim(1200)
    # plt.legend(loc='best', fontsize=30, ncol=2)
    # plt.xlabel('Wavelength (nm)', size=35)
    # plt.ylabel('Splitting ratio', size=35)
    # plt.xticks(size=30);plt.yticks(size=30)
    # plt.tight_layout()
    
    plt.figure(figsize=(19.20,10.80))
    ax = plt.subplot(111)
    # plt.subplot(2,1,1)
    # plt.title('Splitting ratio for beam 1', size=40)
    plt.plot(longueuronde, a[masque], lw=4, c='k', label='To photometric output')
    plt.plot(longueuronde, b[masque]+c[masque], ':', lw=4, c=colors[0], label='To coupler of Null 1')
    # plt.plot(longueuronde, c[masque], '--', lw=4, c=colors[0], label='To antinull 1')
    plt.plot(longueuronde, d[masque]+e[masque], '--', lw=4, c=colors[1], label='To coupler of Null 3')
    # plt.plot(longueuronde, e[masque], '-.', lw=4, c=colors[1], label='To antinull 3')
    plt.plot(longueuronde, f[masque]+g[masque], '-.', lw=4, c=colors[2], label='To coupler of Null 5')
    # plt.plot(longueuronde, g[masque], '.', lw=4, c=colors[2], label='To antinull 5')
    plt.grid()
    plt.ylim(-0.03,0.6)
    # plt.xlim(1200)
    plt.legend(loc='best', fontsize=34, ncol=2)
    plt.xlabel('Wavelength (nm)', size=45)
    plt.ylabel('Splitting ratio', size=45)
    plt.xticks(size=38);plt.yticks(size=38)
    plt.text(-0.09, 1.02, r'a)', weight='bold', fontsize=40, transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig('splitting_ratioN1bis.pdf', format='pdf')


    plt.figure(figsize=(19.20,10.80))
    ax = plt.subplot(111)
    a = Iminus[0,0]/(Iminus[0,0]+Iplus[0,0])
    b = Iplus[0,0]/(Iminus[0,0]+Iplus[0,0])
    c = Iminus[1,0]/(Iminus[1,0]+Iplus[1,0])
    d = Iplus[1,0]/(Iminus[1,0]+Iplus[1,0])
    # plt.title('Coupling ratio for Null 1 (Beams 1&2)', size=40)
    plt.plot(longueuronde, a[masque], lw=4, c=colors[0], label='Beam 1 to null output')
    plt.plot(longueuronde, b[masque], '--', lw=6, c=colors[0], label='Beam 1 to antinull output')
    plt.plot(longueuronde, c[masque], ':', lw=4, c=colors[1], label='Beam 2 to null output')
    plt.plot(longueuronde, d[masque], '--', lw=4, c=colors[1], label='Beam 2 to antinull output')
    plt.grid()
    plt.legend(loc='best', fontsize=34, ncol=2)
    plt.ylim(-0.05,1.3)
    # plt.xlim(1200)
    plt.xlabel('Wavelength (nm)', size=45)
    plt.ylabel('Coupling ratio', size=45)
    plt.xticks(size=38);plt.yticks(size=38)
    plt.text(-0.09, 1.02, r'b)', weight='bold', fontsize=40, transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig('coupling_ratioN1bis.pdf', format='pdf')


    plt.figure(figsize=(19.20,10.80))
    ax = plt.subplot(111)
    plt.plot(img2.wl_scale[0][masque], zeta_coeff[keys[0,0]][0][masque], '-', c=colors[0], lw=4, label='Beam 1 to null output')
    plt.plot(img2.wl_scale[0][masque], zeta_coeff[keys[0,3]][0][masque], ':', c=colors[0],  lw=4, label='Beam 1 to antinull output')
    plt.plot(img2.wl_scale[0][masque], zeta_coeff[keys[1,0]][0][masque], '--', c=colors[1], lw=4, label='Beam 2 to null output')
    plt.plot(img2.wl_scale[0][masque], zeta_coeff[keys[1,3]][0][masque], '-.', c=colors[1],  lw=4, label='Beam 2 to antinull output')
    plt.grid()
    plt.legend(loc='best', fontsize=34)
    plt.xticks(size=38);plt.yticks(size=38)
    plt.ylim(-0.1)
    plt.xlabel('Wavelength (nm)', size=45)
    plt.ylabel(r'$\zeta$ coefficient', size=45)
    plt.text(-0.09, 1.02, r'c)', weight='bold', fontsize=40, transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig('zeta_coeffN1bis.pdf', format='pdf')
