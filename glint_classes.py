# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:48:48 2019

@author: mamartinod

Classes used by the GLINT Data Reduction Software
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from functools import partial
from scipy.optimize import curve_fit
from numba import jit
import os


def gaussian(x, A, loc, sig):
    '''
    Computes a gaussian curve
    
    Parameters
    ----------
    A : amplitude of the gaussian curve
    loc : center of the curve
    sig : standard deviation
    
    Returns
    -----------
    The gaussian curve
    '''
    return A * np.exp(-(x-loc)**2/(2*sig**2))


def _getSpectralFlux(nbimg, which_tracks, slices_axes, slices, spectral_axis, positions, widths):
    ''' 
    Debug version of _getSpectralFluxNumba 
    For development and experimentation purpose
    '''
    nb_tracks = len(which_tracks)
    amplitude_fit = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
    amplitude = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
    integ_model = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
    integ_windowed = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
    residuals_fit = np.zeros((nbimg, nb_tracks, len(spectral_axis), slices_axes.shape[1]))
    residuals_reg = np.zeros((nbimg, nb_tracks, len(spectral_axis), slices_axes.shape[1]))
    cov = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
    weights = np.zeros((nbimg, nb_tracks, len(spectral_axis)))

    # With fitted amplitude
    for k in range(nbimg):
        for i in which_tracks:
            for j in range(len(spectral_axis)):
                gaus = partial(gaussian, loc=positions[i,j], sig=widths[i,j])
                popt, pcov = curve_fit(gaus, slices_axes[i], slices[k,j,i], p0=[slices[k,j,i].max()])
                amplitude_fit[k,i,j] = popt[0]
                cov[k,i,j] = pcov[0,0]
                integ_model[k,i,j] = np.sum(gaus(slices_axes[i], *popt))
                weight = gaus(slices_axes[i], 1.)
                weight /= weight.sum()
                integ_windowed[k,i,j] = np.sum(weight * slices[k,j,i])
                residuals_fit[k,i,j] = slices[k,j,i] - gaus(slices_axes[i], *popt)
                
                simple_gaus = np.exp(-(slices_axes[i]-positions[i,j])**2/(2*widths[i,j]**2))
                A = np.vstack((simple_gaus, np.zeros(simple_gaus.shape)))
                A = np.transpose(A)
                popt2 = np.linalg.lstsq(A, slices[k,j,i])[0]
                residuals_reg[k,i,j] = slices[k,j,i] - popt2[0] * simple_gaus
                amplitude[k,i,j] = popt2[0]
                integ_model[k,i,j] = np.sum(simple_gaus * popt2[0])
                weight = simple_gaus.copy()
                weight /= np.sum(weight)
                integ_windowed[k,i,j] = np.sum(weight * slices[k,j,i])
                weights[k,i,j] = weight.sum()
                
#                switch = True
#                if abs(popt) > 1.e+4 or abs(popt2[0]) > 1.e+4:
#                    if abs(popt) > 1.e+4:
#                        debug.append([0, k, i, j])
#                    if abs(popt2[0]) > 1.e+4:
#                        debug.append([1, k, i, j])
#                if j == 3:
#                    plt.figure()
#                    plt.subplot(311)
#                    plt.plot(slices_axes[i], slices[k,j,i], label='data')
#                    plt.plot(slices_axes[i], gaus(slices_axes[i], *popt), 'o', label='curve_fit '+str(popt[0]))
#                    plt.plot(slices_axes[i], popt2 [0]* simple_gaus, 'd', label='linear reg '+str(popt2[0]))
#                    plt.xlabel('Spatial position (px)')
#                    plt.ylabel('Amplitude')
#                    plt.grid()
#                    plt.legend(loc='best')
#                    plt.title('Frame '+str(k)+'/ Track '+str(i+1)+'/ Column '+str(j))
#                    plt.subplot(312)
#                    plt.plot(slices[k,j,i], residuals_fit[k,i,j], 'o', label='fit')
#                    plt.plot(slices[k,j,i], residuals_reg[k,i,j], 'd', label='linear reg')
#                    plt.xlabel('Amplitude')
#                    plt.ylabel('Residual')
#                    plt.grid()
#                    plt.legend(loc='best')
#                    plt.subplot(313)
#                    plt.plot(slices_axes[i], residuals_fit[k,i,j], 'o', label='fit')
#                    plt.plot(slices_axes[i], residuals_reg[k,i,j], 'd', label='linear reg')
#                    plt.xlabel('Spatial position (px)')
#                    plt.ylabel('Residual')
#                    plt.grid()
#                    plt.legend(loc='best')
#                    
#                    if switch == True:
#                        temp = gaus(slices_axes[i], 1.)
#                        temp2 = simple_gaus
#                        switch = False
    
    return amplitude_fit, amplitude, integ_model, integ_windowed, residuals_fit, residuals_reg, cov, weights


class File(object):
    ''' Management of the HDF5 datacube'''
    
    def __init__(self, data=None, nbimg=None, transpose=True):
        self.loadfile(data, nbimg, transpose)
            
            
    def loadfile(self, data=None, nbimg=None, transpose=True):
        ''' Loading of FITS file 
        -----------------
        data : string, path to data to process.
                If None, mock data is created
                
        nbimg : number of frames from the file to use.
                If None: the whole file is processed
        '''

        if data != None:
            with h5py.File(data) as dataFile:
                self.data   = np.array(dataFile['imagedata'])
                if nbimg == None:
                    self.nbimg  = self.data.shape[0]
                    self.index = np.arange(self.nbimg)
                else:
                    self.data = self.data[:nbimg]
                    self.nbimg  = nbimg
                    self.index = np.arange(self.nbimg)
                if transpose: self.data = np.transpose(self.data, axes=(0,2,1))
                
        else:
            print("Mock data created")
            self.header = {}
            self.nbimg = nbimg
            self.data = np.zeros((self.nbimg,344,96))
            self.index = np.arange(self.nbimg)

    def cosmeticsFrames(self, dark):
        self.data = self.data - dark
        self.data = self.data - self.data[:,:,:20].mean(axis=(1,2))[:,None,None]
        self.bg_std = self.data[:,:,:20].std(axis=(1,2))
        self.bg_var = self.data[:,:,:20].var(axis=(1,2))
            
    def binning(self, binning, axis=0, avg=False):
        '''
        Meaning of a sample of images to recreate a new stack of images
        HIW : we change the shape of the array (input stack of images) to put the element which will be summed on a specific axis.
        x : array with the stack of images to bin
        binning : number of image to bin
        axis : summation axis
        '''

        shape = self.data.shape
        if axis < 0:
            axis += self.data.ndim
        shape = shape[:axis] + (-1, binning) + shape[axis+1:]
        self.data = self.data.reshape(shape)
        if not avg:
            self.data = self.data.sum(axis=axis+1)
        else:
            self.data = self.data.mean(axis=axis+1)


class Null(File):
    ''' Getting the null and photometries'''
    
    def insulateTracks(self, channel_pos, sep, spatial_axis):
        ''' 
        Insulating each track 
        ------------------
        channel_pos : array, expected position of each track
        sep : float, separation between the tracks
        spatial_axis : array, coordinates of each tracks along the spatial axis on the detector
                         (perpendicular to the spectral one)
        '''
        self.slices = np.array([self.data[:,np.int(np.around(pos-sep/2)):np.int(np.around(pos+sep/2)),:] for pos in channel_pos])
        self.slices = np.transpose(self.slices, (1,3,0,2))
        self.slices_axes = np.array([spatial_axis[np.int(np.around(pos-sep/2)):np.int(np.around(pos+sep/2))] for pos in channel_pos])

        
    def getSpectralFlux(self, which_tracks, spectral_axis, position_poly, width_poly, debug=False):
        ''' 
        Measurement of flux per frame, per spectral channel, per track 
        ------------------
        which_tracks : list of tracks to process (null, anti null and photometric)
        spectral_axis : array, spectra axis on the detector
        position_poly : list of numpy polynom-function of the center of each track
        width_poly : the same for the width
        debug: bool, use the non-numba function for development purpose (use several fit methods, generate plots...)
        '''
        nbimg = self.data.shape[0]
        slices_axes, slices = self.slices_axes, self.slices
        positions = np.array([p(spectral_axis) for p in position_poly])
        widths = np.array([p(spectral_axis) for p in width_poly])
        self.raw = self.slices.sum(axis=-1)
        self.raw = np.transpose(self.raw, axes=(0,2,1))
        
        if debug:
            self.amplitude_fit, self.amplitude, self.integ_model, self.integ_windowed, self.residuals_fit, self.residuals_reg, self.cov, self.weights = \
        _getSpectralFlux(nbimg, which_tracks, slices_axes, slices, spectral_axis, positions, widths)
        else:
            self.amplitude, self.integ_model, self.integ_windowed, self.residuals_reg, self.weights = \
            self._getSpectralFluxNumba(nbimg, which_tracks, slices_axes, slices, spectral_axis, positions, widths)

        self.raw_err = self.bg_std * slices_axes.shape[-1]**0.5
        self.windowed_err = self.bg_std * np.sum(self.weights)**0.5
        
    @staticmethod
    @jit(nopython=True)
    def _getSpectralFluxNumba(nbimg, which_tracks, slices_axes, slices, spectral_axis, positions, widths):
        ''' Numba-ized function, to be routinely used 
        ------------------
        nbimg : int, number of frames to process
        which_tracks : list of tracks to process (null, anti null and photometric)
        slices_axes : array, spatial axis for each track
        slices : array containing the 16 individual tracks displayed on the detector
        spectral_axis : array, spectra axis on the detector
        positions : array of locations of the gaussian shape for linear least square fit in a spectral channel
        widths : array of widths of this gaussian
        '''
        
        nb_tracks = len(which_tracks)
        amplitude = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
        integ_model = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
        integ_windowed = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
        residuals_reg = np.zeros((nbimg, nb_tracks, len(spectral_axis), slices_axes.shape[1]))
        weights = np.zeros((nbimg, nb_tracks, len(spectral_axis)))
        
        # With fitted amplitude
        for k in range(nbimg):
            for i in which_tracks:
                for j in range(len(spectral_axis)):
                    # 1st estimator : amplitude of the Gaussian profil of the track, use of linear least square
                    simple_gaus = np.exp(-(slices_axes[i]-positions[i,j])**2/(2*widths[i,j]**2)) # Shape factor of the intensity profile, to be removed before computing Null depth
                    A = np.vstack((simple_gaus, np.zeros(simple_gaus.shape)))
                    A = np.transpose(A)
                    popt2 = np.linalg.lstsq(A, slices[k,j,i])[0]
                    residuals_reg[k,i,j] = slices[k,j,i] - popt2[0] * simple_gaus
                    amplitude[k,i,j] = popt2[0]
                    
                    # 2nd estimator : integration of the energy knowing the amplitude
                    integ_model[k,i,j] = np.sum(simple_gaus * popt2[0] / np.sum(simple_gaus))
                    
                    # 3rd estimator : weighted mean of the track in a column of pixels
                    weight = np.exp(-(slices_axes[i]-positions[i,j])**2/(2*2**2)) # Same weight window for every tracks and spectral channel to make it removed when doing the ratio of intensity
                    integ_windowed[k,i,j] = np.sum(weight * slices[k,j,i] / np.sum(simple_gaus))
                    weights[k,i,j] = weight.sum()
                    
        return amplitude, integ_model, integ_windowed, residuals_reg, weights
    
    def getTotalFlux(self):
        self.fluxes = np.sum(self.slices, axis=(1,3))
        
    def matchSpectralChannels(self, wl_to_px_coeff, px_to_wl_coeff, which_tracks):
        '''
        All tracks are slightly shifted respect to each other.
        Need to define the common wavelength to all of them and create a 
        matching map between the spectral channels of every tracks.
        --------------------------
        wl_to_px_coeff : array, conversion from wavelength to pixel for each track, 
                         from spectral_calibration script,
                        
        px_to_wl_coeff : array, inverted conversion, got from the same script,
                         for each track
                         
        which_tracks : list of tracks to process (null, anti null and photometric)
        '''
        
        wl_to_px_poly = [np.poly1d(wl_to_px_coeff[i]) for i in which_tracks]
        px_to_wl_poly = [np.poly1d(px_to_wl_coeff[i]) for i in which_tracks]
        shape = self.data.shape
        
        start_wl = [px_to_wl_poly[i](0) for i in which_tracks]
        end_wl = [px_to_wl_poly[i](shape[-1]) for i in which_tracks]
        
        start = np.around(min(start_wl))
        end = np.around(max(end_wl))

        self.wl_scale = np.array([np.arange(start, end, np.around(px_to_wl_coeff[i,0])) for i in which_tracks])
        self.px_scale = np.array([np.around(wl_to_px_poly[i](self.wl_scale[i])) for i in which_tracks], dtype=np.int)
        
    def computeNullDepth(self):
        ''' Compute null depths with the different estimators of the flux in a spectral channel'''
        
        # With amplitude
        self.null1 = self.amplitude[:,11][:,self.px_scale[11]]/self.amplitude[:,9][:,self.px_scale[9]]
        self.null2 = self.amplitude[:,3][:,self.px_scale[3]]/self.amplitude[:,12][:,self.px_scale[12]]
        self.null3 = self.amplitude[:,1][:,self.px_scale[1]]/self.amplitude[:,14][:,self.px_scale[14]]
        self.null4 = self.amplitude[:,6][:,self.px_scale[6]]/self.amplitude[:,4][:,self.px_scale[4]]
        self.null5 = self.amplitude[:,5][:,self.px_scale[5]]/self.amplitude[:,7][:,self.px_scale[7]]
        self.null6 = self.amplitude[:,8][:,self.px_scale[8]]/self.amplitude[:,10][:,self.px_scale[10]]
        
        self.null1_err = np.sqrt(self.bg_var[:,None]/(self.amplitude[:,9][:,self.px_scale[9]])**2 * (1 + self.null1**2))
        self.null2_err = np.sqrt(self.bg_var[:,None]/(self.amplitude[:,12][:,self.px_scale[12]])**2 * (1 + self.null2**2))
        self.null3_err = np.sqrt(self.bg_var[:,None]/(self.amplitude[:,14][:,self.px_scale[14]])**2 * (1 + self.null3**2))
        self.null4_err = np.sqrt(self.bg_var[:,None]/(self.amplitude[:,4][:,self.px_scale[4]])**2 * (1 + self.null4**2))
        self.null5_err = np.sqrt(self.bg_var[:,None]/(self.amplitude[:,7][:,self.px_scale[7]])**2 * (1 + self.null5**2))
        self.null6_err = np.sqrt(self.bg_var[:,None]/(self.amplitude[:,10][:,self.px_scale[10]])**2 * (1 + self.null6**2))
        
        # With full gaussian model
        self.null_model1 = self.integ_model[:,11][:,self.px_scale[11]]/self.integ_model[:,9][:,self.px_scale[9]]
        self.null_model2 = self.integ_model[:,3][:,self.px_scale[3]]/self.integ_model[:,12][:,self.px_scale[12]]
        self.null_model3 = self.integ_model[:,1][:,self.px_scale[1]]/self.integ_model[:,14][:,self.px_scale[14]]
        self.null_model4 = self.integ_model[:,6][:,self.px_scale[6]]/self.integ_model[:,4][:,self.px_scale[4]]
        self.null_model5 = self.integ_model[:,5][:,self.px_scale[5]]/self.integ_model[:,7][:,self.px_scale[7]]
        self.null_model6 = self.integ_model[:,8][:,self.px_scale[8]]/self.integ_model[:,10][:,self.px_scale[10]]
        
        self.null_model1_err = np.sqrt(self.raw_err[:,None]**2/(self.integ_model[:,9][:,self.px_scale[9]])**2 * (1 + self.null_model1**2))
        self.null_model2_err = np.sqrt(self.raw_err[:,None]**2/(self.integ_model[:,12][:,self.px_scale[12]])**2 * (1 + self.null_model2**2))
        self.null_model3_err = np.sqrt(self.raw_err[:,None]**2/(self.integ_model[:,14][:,self.px_scale[14]])**2 * (1 + self.null_model3**2))
        self.null_model4_err = np.sqrt(self.raw_err[:,None]**2/(self.integ_model[:,4][:,self.px_scale[4]])**2 * (1 + self.null_model4**2))
        self.null_model5_err = np.sqrt(self.raw_err[:,None]**2/(self.integ_model[:,7][:,self.px_scale[7]])**2 * (1 + self.null_model5**2))
        self.null_model6_err = np.sqrt(self.raw_err[:,None]**2/(self.integ_model[:,10][:,self.px_scale[10]])**2 * (1 + self.null_model6**2))
        
        # With windowed integration
        self.null_windowed1 = self.integ_windowed[:,11][:,self.px_scale[11]]/self.integ_windowed[:,9][:,self.px_scale[9]]
        self.null_windowed2 = self.integ_windowed[:,3][:,self.px_scale[3]]/self.integ_windowed[:,12][:,self.px_scale[12]]
        self.null_windowed3 = self.integ_windowed[:,1][:,self.px_scale[1]]/self.integ_windowed[:,14][:,self.px_scale[14]]
        self.null_windowed4 = self.integ_windowed[:,6][:,self.px_scale[6]]/self.integ_windowed[:,4][:,self.px_scale[4]]
        self.null_windowed5 = self.integ_windowed[:,5][:,self.px_scale[5]]/self.integ_windowed[:,7][:,self.px_scale[7]]
        self.null_windowed6 = self.integ_windowed[:,8][:,self.px_scale[8]]/self.integ_windowed[:,10][:,self.px_scale[10]]
        
        self.null_windowed1_err = np.sqrt(self.windowed_err[:,None]**2/(self.integ_windowed[:,9][:,self.px_scale[9]])**2 * (1 + self.null_windowed1**2))
        self.null_windowed2_err = np.sqrt(self.windowed_err[:,None]**2/(self.integ_windowed[:,12][:,self.px_scale[12]])**2 * (1 + self.null_windowed2**2))
        self.null_windowed3_err = np.sqrt(self.windowed_err[:,None]**2/(self.integ_windowed[:,14][:,self.px_scale[14]])**2 * (1 + self.null_windowed3**2))
        self.null_windowed4_err = np.sqrt(self.windowed_err[:,None]**2/(self.integ_windowed[:,4][:,self.px_scale[4]])**2 * (1 + self.null_windowed4**2))
        self.null_windowed5_err = np.sqrt(self.windowed_err[:,None]**2/(self.integ_windowed[:,7][:,self.px_scale[7]])**2 * (1 + self.null_windowed5**2))
        self.null_windowed6_err = np.sqrt(self.windowed_err[:,None]**2/(self.integ_windowed[:,10][:,self.px_scale[10]])**2 * (1 + self.null_windowed6**2))
        
        # With raw integration
        self.null_raw1 = self.raw[:,11][:,self.px_scale[11]]/self.raw[:,9][:,self.px_scale[9]]
        self.null_raw2 = self.raw[:,3][:,self.px_scale[3]]/self.raw[:,12][:,self.px_scale[12]]
        self.null_raw3 = self.raw[:,1][:,self.px_scale[1]]/self.raw[:,14][:,self.px_scale[14]]
        self.null_raw4 = self.raw[:,6][:,self.px_scale[6]]/self.raw[:,4][:,self.px_scale[4]]
        self.null_raw5 = self.raw[:,5][:,self.px_scale[5]]/self.raw[:,7][:,self.px_scale[7]]
        self.null_raw6 = self.raw[:,8][:,self.px_scale[8]]/self.raw[:,10][:,self.px_scale[10]]
        
        self.null_raw1_err = np.sqrt(self.raw_err[:,None]**2/(self.raw[:,9][:,self.px_scale[9]])**2 * (1 + self.null_raw1**2))
        self.null_raw2_err = np.sqrt(self.raw_err[:,None]**2/(self.raw[:,12][:,self.px_scale[12]])**2 * (1 + self.null_raw2**2))
        self.null_raw3_err = np.sqrt(self.raw_err[:,None]**2/(self.raw[:,14][:,self.px_scale[14]])**2 * (1 + self.null_raw3**2))
        self.null_raw4_err = np.sqrt(self.raw_err[:,None]**2/(self.raw[:,4][:,self.px_scale[4]])**2 * (1 + self.null_raw4**2))
        self.null_raw5_err = np.sqrt(self.raw_err[:,None]**2/(self.raw[:,7][:,self.px_scale[7]])**2 * (1 + self.null_raw5**2))
        self.null_raw6_err = np.sqrt(self.raw_err[:,None]**2/(self.raw[:,10][:,self.px_scale[10]])**2 * (1 + self.null_raw6**2))
        
    def getPhotometry(self):
        ''' Measure flux in a spectral channel with the different estimators'''
        
        # With amplitude
        self.p1 = self.amplitude[:,15,:][:,self.px_scale[15]]
        self.p2 = self.amplitude[:,13,:][:,self.px_scale[13]]
        self.p3 = self.amplitude[:,2,:][:,self.px_scale[2]]
        self.p4 = self.amplitude[:,0,:][:,self.px_scale[0]]
        
        self.p1_err = self.p2_err = self.p3_err = self.p4_err = self.bg_std

        # With full gaussian model
        self.p1_model = self.integ_model[:,15,:][:,self.px_scale[15]]
        self.p2_model = self.integ_model[:,13,:][:,self.px_scale[13]]
        self.p3_model = self.integ_model[:,2,:][:,self.px_scale[2]]
        self.p4_model = self.integ_model[:,0,:][:,self.px_scale[0]]
        
        self.p1_model_err = self.p2_model_err = self.p3_model_err = self.p4_model_err = self.raw_err
        
        # With windowed integration
        self.p1_windowed = self.integ_windowed[:,15,:][:,self.px_scale[15]]
        self.p2_windowed = self.integ_windowed[:,13,:][:,self.px_scale[13]]
        self.p3_windowed = self.integ_windowed[:,2,:][:,self.px_scale[2]]
        self.p4_windowed = self.integ_windowed[:,0,:][:,self.px_scale[0]]
        
        self.p1_windowed_err = self.p2_windowed_err = self.p3_windowed_err = self.p4_windowed_err = self.windowed_err
        
        # With raw integration
        self.p1_raw = self.raw[:,15,:][:,self.px_scale[15]]
        self.p2_raw = self.raw[:,13,:][:,self.px_scale[13]]
        self.p3_raw = self.raw[:,2,:][:,self.px_scale[2]]
        self.p4_raw = self.raw[:,0,:][:,self.px_scale[0]]
        
        self.p1_raw_err = self.p2_raw_err = self.p3_raw_err = self.p4_raw_err = self.raw_err        
                
    def save(self, path, date, mode):
        '''
        path : string, path of the file to save. Must contain the name of the file
        date : string, date of the acquisition of the data (YYYY-MM-DD)
        mode : string, which estimator to use
        '''
        
        beams_couple = {'null1':' Beams 1/2', 'null2':'Beams 2/3', 'null3':'Beams 1/4',\
                        'null4':'Beams 3/4', 'null5':'Beams 1/3', 'null6':'Beams 2/4'}
        
        if mode == 'amplitude':
            arrs = [[self.null1, self.null1_err, self.p1, self.p1_err, self.p2, self.p2_err],\
                [self.null2, self.null2_err, self.p2, self.p2_err, self.p3, self.p3_err],\
                [self.null3, self.null3_err, self.p1, self.p1_err, self.p4, self.p4_err],\
                [self.null4, self.null4_err, self.p3, self.p3_err, self.p4, self.p4_err],\
                [self.null5, self.null5_err, self.p1, self.p1_err, self.p3, self.p3_err],\
                [self.null6, self.null6_err, self.p2, self.p2_err, self.p4, self.p4_err]]
        elif mode == 'model':
            arrs = [[self.null_model1, self.null_model1_err, self.p1_model, self.p1_model_err, self.p2_model, self.p2_model_err],\
                [self.null_model2, self.null_model2_err, self.p2_model, self.p2_model_err, self.p3_model, self.p3_model_err],\
                [self.null_model3, self.null_model3_err, self.p1_model, self.p1_model_err, self.p4_model, self.p4_model_err],\
                [self.null_model4, self.null_model4_err, self.p3_model, self.p3_model_err, self.p4_model, self.p4_model_err],\
                [self.null_model5, self.null_model5_err, self.p1_model, self.p1_model_err, self.p3_model, self.p3_model_err],\
                [self.null_model6, self.null_model6_err, self.p2_model, self.p2_model_err, self.p4_model, self.p4_model_err]]
        elif mode == 'windowed':
            arrs = [[self.null_windowed1, self.null_windowed1_err, self.p1_windowed, self.p1_windowed_err, self.p2_windowed, self.p2_windowed_err],\
                [self.null_windowed2, self.null_windowed2_err, self.p2_windowed, self.p2_windowed_err, self.p3_windowed, self.p3_windowed_err],\
                [self.null_windowed3, self.null_windowed3_err, self.p1_windowed, self.p1_windowed_err, self.p4_windowed, self.p4_windowed_err],\
                [self.null_windowed4, self.null_windowed4_err, self.p3_windowed, self.p3_windowed_err, self.p4_windowed, self.p4_windowed_err],\
                [self.null_windowed5, self.null_windowed5_err, self.p1_windowed, self.p1_windowed_err, self.p3_windowed, self.p3_windowed_err],\
                [self.null_windowed6, self.null_windowed6_err, self.p2_windowed, self.p2_windowed_err, self.p4_windowed, self.p4_windowed_err]]
        else:
            arrs = [[self.null_raw1, self.null_raw1_err, self.p1_raw, self.p1_raw_err, self.p2_raw, self.p2_raw_err],\
                [self.null_raw2, self.null_raw2_err, self.p2_raw, self.p2_raw_err, self.p3_raw, self.p3_raw_err],\
                [self.null_raw3, self.null_raw3_err, self.p1_raw, self.p1_raw_err, self.p4_raw, self.p4_raw_err],\
                [self.null_raw4, self.null_raw4_err, self.p3_raw, self.p3_raw_err, self.p4_raw, self.p4_raw_err],\
                [self.null_raw5, self.null_raw5_err, self.p1_raw, self.p1_raw_err, self.p3_raw, self.p3_raw_err],\
                [self.null_raw6, self.null_raw6_err, self.p2_raw, self.p2_raw_err, self.p4_raw, self.p4_raw_err]]

        # Check if saved file exist
        if os.path.exists(path):
            opening_mode = 'w' # Overwright the whole existing file.
        else:
            opening_mode = 'a' # Create a new file at "path"
            
        with h5py.File(path, opening_mode) as f:
            f.attrs['date'] = date
            f.attrs['nbimg'] = self.nbimg
            
            for i in range(6):
                f.create_group('null%s'%(i+1))
                f.create_dataset('null%s/null'%(i+1), data=arrs[i][0])
                f.create_dataset('null%s/null_err'%(i+1), data=arrs[i][1])
                f.create_dataset('null%s/pA'%(i+1), data=arrs[i][2])
                f.create_dataset('null%s/pA_err'%(i+1), data=arrs[i][3])
                f.create_dataset('null%s/pB'%(i+1), data=arrs[i][4])
                f.create_dataset('null%s/pB_err'%(i+1), data=arrs[i][5])
                f.create_dataset('null%s/wl_scale'%(i+1), data=self.wl_scale.mean(axis=0))
                
                f['null%s'%(i+1)].attrs['comment'] = beams_couple['null%s'%(i+1)]
                f['null%s/null'%(i+1)].attrs['comment'] = 'python dim : (nb frame, wl channel)'
                f['null%s/null_err'%(i+1)].attrs['comment'] = 'python dim : (nb frame, wl channel)'
                f['null%s/pA'%(i+1)].attrs['comment'] = 'python ndim : (nb frame, wl channel)'
                f['null%s/pA_err'%(i+1)].attrs['comment'] = 'python dim : (nb frame, wl channel)'
                f['null%s/pB'%(i+1)].attrs['comment'] = 'python ndim : (nb frame, wl channel)'
                f['null%s/pB_err'%(i+1)].attrs['comment'] = 'python dim : (nb frame, wl channel)'
                f['null%s/wl_scale'%(i+1)].attrs['comment'] = 'wl in nm'
                