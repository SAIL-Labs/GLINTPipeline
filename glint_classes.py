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
    """
    Computes a gaussian curve
    
    :Parameters:
    
        **x: (N,) array**
            Values for which the gaussian is estimated
            
        **A: float**
            amplitude of the gaussian curve
        
        **loc: float**
            center of the curve
    
        **sig: float>0**
            scale factor of the curve
    
    :Returns:
        
        The gaussian curve respect to x values
    """
    return A * np.exp(-(x-loc)**2/(2*sig**2))


def _getSpectralFlux(nbimg, which_tracks, slices_axes, slices, spectral_axis, positions, widths):
    """ 
    Debug version of _getSpectralFluxNumba.
    Called when ''debug'' is ''True''.
    
    For development and experimentation purpose.
    Plot the linear fit and the gaussian profil for one spectral channel of the first frame for every tracks.
    Read the description of ''_getSpectralFluxNumba'' for details about the inputs.
    """
    nb_tracks = 16
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
#        print(k)
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
                if j==47 and k == 0:
                    print(k, i, j)
                    print('Weight on std', (np.sum((simple_gaus/simple_gaus.sum())**2))**0.5)
                    print(slices[k,j,i][:7].std())
                    plt.figure()
                    plt.subplot(211)
                    plt.plot(slices_axes[i], slices[k,j,i], 'o', label='data')
                    plt.plot(slices_axes[i], gaus(slices_axes[i], *popt), '+-', label='curve_fit '+str(popt[0]))
                    plt.plot(slices_axes[i], popt2 [0]* simple_gaus, '+--', label='linear reg '+str(popt2[0]))
                    plt.xlabel('Spatial position (px)')
                    plt.ylabel('Amplitude')
                    plt.grid()
                    plt.legend(loc='best')
                    plt.title('Frame '+str(k)+'/ Track '+str(i+1)+'/ Column '+str(j))
#                    plt.subplot(312)
#                    plt.plot(slices[k,j,i], residuals_fit[k,i,j], 'o', label='fit')
#                    plt.plot(slices[k,j,i], residuals_reg[k,i,j], 'd', label='linear reg')
#                    plt.xlabel('Amplitude')
#                    plt.ylabel('Residual')
#                    plt.grid()
#                    plt.legend(loc='best')
                    plt.subplot(212)
                    plt.plot(slices_axes[i], residuals_fit[k,i,j], 'o', label='fit')
                    plt.plot(slices_axes[i], residuals_reg[k,i,j], 'd', label='linear reg')
                    plt.xlabel('Spatial position (px)')
                    plt.ylabel('Residual')
                    plt.grid()
                    plt.legend(loc='best')
#                    
#                    if switch == True:
#                        temp = gaus(slices_axes[i], 1.)
#                        temp2 = simple_gaus
#                        switch = False
    
    return amplitude_fit, amplitude, integ_model, integ_windowed, residuals_fit, residuals_reg, cov, weights


class File(object):
    """
    Management of the HDF5 datacube generated by GLINT
    
    :Parameters:
        
        **data: string (optional)**
            Path to the datacube to load. 
            If ''None'', datacube full of 0 is created, with same dimension as real data (nbimg, 344, 96).
            In that case, parameter ''nbimg'' cannot be ''None''.
            
        **nbimg: tup (optional)**
            Load all frames of the datacube at path ''data'' from the first to the second-1 element of the tuple.
            It cannot be ''(None, None)'' if mock data is created.
            
        **transpose: bol (optional)**
            If ''True'', swappes the 2nd and 3rd axis of the datacube.
    """
    
    def __init__(self, data=None, nbimg=(None, None), transpose=False):
        """
        Init the instance class by calling the ''loadfile' method.
        """
        self.loadfile(data, nbimg, transpose)
            
            
    def loadfile(self, data=None, nbimg=None, transpose=False):
        """ 
        Load the datacube when a File-object is created.

        :Parameters:
            
            **data: string (optional)**
                Path to the datacube to load. 
                If ''None'', datacube full of 0 is created, with same dimension as real data (nbimg, 344, 96).
                In that case, parameter ''nbimg'' cannot be ''None''.
                
            **nbimg: tup (optional)**
                Load all frames of the datacube at path ''data'' from the first to the second-1 element of the tuple.
                It cannot be ''(None, None)'' if mock data is created.
                
            **transpose: bol (optional)**
                If ''True'', swappes the 2nd and 3rd axis of the datacube.
                
        :Attributes:
            Return the attributes
            
            **data**: ndarray 
                loaded or created datacube
            **nbimg**: float
                number of frames in the data ndarray
            
        """

        if data != None:
            with h5py.File(data) as dataFile:
                self.data   = np.array(dataFile['imagedata'])
                self.data = self.data[nbimg[0]:nbimg[1]]
                self.nbimg  = self.data.shape[0]
                if transpose: 
#                    print('Transpose')
#                    print(self.data.shape)
                    self.data = np.transpose(self.data, axes=(0,2,1))
#                    print(self.data.shape)
                self.data = np.transpose(self.data, axes=(0,2,1))
                    
        else:
            print("Mock data created")
            self.nbimg = nbimg[1]-nbimg[0]
            self.data = np.zeros((self.nbimg,344,96))

    def cosmeticsFrames(self, dark, nonoise=False):
        """ 
        Deprecated: ''dark'' should be a 2d-array full of 0.
        
        Remove dark and bias from data frames.
        Directly acts on ''data'' attribute.
        Compute the variance and standard deviation of the background noise 
        in a signal-free area of the frames.

        :Parameters:
            **dark: 2d-array**
                Average dark
                
            **nbimg: tup (optional)**
                Load all frames of the datacube at path ''data'' from the first 
                to the second-1 element of the tuple.
                It cannot be ''(None, None)'' if mock data is created.
                
            **nonoise: bol**
                Set to ''True'' if data does not have any detector noise (e.g. simulated one).
                It skips the cosmetics and set estimation of the 
                background variance/std to zero
                
        :Attributes:
            Change the attributes
            
            **data**: ndarray 
                data frame
                
            **bg_std**: ndarray
                Standard deviation of the background of each frame
            
            **bg_var**: ndarray
                Variance of the background of each frame
            
        """
        if nonoise:
            self.bg_std = np.zeros(self.data.shape[0])
            self.bg_var = np.zeros(self.data.shape[0])
        else:
            if not np.all(dark==0): #If 'dark' is not a 0-array
                self.data = self.data - dark
                self.data = self.data - self.data[:,:,:20].mean(axis=(1,2))[:,None,None]
                
            self.bg_std = self.data[:,:,:20].std(axis=(1,2))
            self.bg_var = self.data[:,:,:20].var(axis=(1,2))
            
    def binning(self, binning, axis=0, avg=False):
        """
        Bin frames together
        
        :Parameters
            **binning**: int
                Number of frames to bin
            **axis**: int
                axis along which the frames are
            **avg**: bol
                If ''True'', the method returns the average of the binned frame.
                Otherwise, return its sum.
                
        :Attributes:
            Change the attributes
            
            **data**: ndarray 
                datacube
        """

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
    """
    Class handling the measurement of the null and photometries 
    from bias-corrected frame.
    """
        
    def getChannels(self, channel_pos, sep, spatial_axis, **kwargs):
        """
        Extract the 16 channels/outputs from frames
        
        :Parameters
            **channel_pos**: list, array-like
                Expected position of the arrays
            **sep**: float
                Separation (in pixels) between two consecutive channels
            **spatial_axis**: 1d-array
                Position-coordinate of each channel
            **kwargs**: optional
                Can add the keyword ''dark'' associated with the average dark 
                of each channel to perform their cosmetics instead of using the
                ''cosmeticsFrames'' method from class ''File''.
                
        :Attributes:
            Create the attributes
            
            **slices**: 4d-darray 
                Subframes of each channel.
                Structure as follow: (frame, spectral axis, channel ID, spatial axis)
            **slices_axes**: ndarray
                Spatial coordinates of each channel
        """
        self.slices = np.array([self.data[:,np.int(np.around(pos-sep/2)):np.int(np.around(pos+sep/2)),:] for pos in channel_pos])
        self.slices = np.transpose(self.slices, (1,3,0,2))
        self.slices_axes = np.array([spatial_axis[np.int(np.around(pos-sep/2)):np.int(np.around(pos+sep/2))] for pos in channel_pos])
        
        if 'dark' in kwargs:
            self.slices = self.slices - kwargs['dark'][None,:]

        
    def getSpectralFlux(self, which_tracks, spectral_axis, position_poly, width_poly, debug=False):
        """
        Wrapper getting the flux per spectral channel of each output.
        
        :Parameters
            **which_tracks**: list
                list of the output from which to measure the flux.
                Outputs are numbered from 0 to 15 from top to bottom 
                (in the way the frame are loaded)
            **spectral_axis**: ndarray
                Common spectral axis in pixel for every outputs
            **position_poly**: array-like
                Estimated polynomial coefficients of the positions of each 
                output respect to wavelength.
            **width_poly**: array-like
                Estimated polynomial coefficients of the widths of each 
                output respect to wavelength, assuming a gaussian profile.
            **debug**: bol
                Debug mode.
                If ''True'', use the python/numpy function ''_getSpectralFlux'' which is 
                slow and allow visual check of the well behaviour of the model fitting.
                Strongly recommended to load only one block and one frame of data and deactivated
                the save of the final products.
                If ''False'', use the numba function ''_getSpectralFluxNumba''.
                For fast and routine use of the measurement of the flux.
                
        :Attributes:
            Creates the following attributes
            
            **raw**: ndarray 
                Estimation of the spectral flux by simply summing ''slices'' along spatial axis
            **raw_err**:ndarray
                Estimation of the uncertainty of the estimation of the raw flux.
            **amplitude**: ndarray
                Estimation of the spectral flux as the amplitude of the Gaussian 
                profile fitted by numpy's linear leastsquare method.
            **integ_model**: ndarray
                Estimation of the spectral flux as the integral of the theoretical
                Gaussian profile which parameters are the amplitude 
                (used in the ''amplitude'' method) and the measured positions and width
                of this profile per wavelength.
            **integ_windowed**: ndarray
                Like raw integral but with a Gaussian window giving different 
                weights to pixels along spatial axis
            **weights**:
                Weights in the windowing.
            **residuals_reg**: ndarray
                Residuals from the fit which gives ''amplitude'' attribute.
            **amplitude_fit**: ndarray
                From debug-mode only.
                Estimation of the spectral flux as the amplitude of the Gaussian 
                profile fitted by the scipy's curve_fit method, 
                assuming a gaussian profile.
            **residuals_fit**: ndarray
                From debug-mode only.
                Residuals of the fit performed by the scipy's curve_fit method
            **cov**: scalar
                From debug-mode only.
                Covariance estimated by the scipy's curve_fit method
        """
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
        """
        Numba-ized function measuring the flux per spectral channel (1 pixel width)
        
        :Parameters
            **nbimg**: int
                Number fo frames to process
            **which_tracks**: array-like
                list of the output from which to measure the flux.
                Outputs are numbered from 0 to 15 from top to bottom 
                (in the way the frame are loaded).
            **slices_axes**: ndarray
                Spatial axis in pixel for every outputs.
            **slices**: ndarray
                Array containing the flux in the 16 outputs on every frames.
            **spectral_axis**: ndarray
                Common spectral axis in pixel for every outputs.
            **positions**: array-like
                Estimated positions of each output respect to wavelength.
            **widths**: array-like
                Estimated widths of each output respect to wavelength.
                
        :Returns:
            
            **amplitude**: ndarray
                Estimation of the spectral flux as the amplitude of the Gaussian 
                profile fitted by numpy's linear leastsquare method.
            **integ_model**: ndarray
                Estimation of the spectral flux as the integral of the theoretical
                Gaussian profile which parameters are the amplitude 
                (used in the ''amplitude'' method) and the measured positions and width
                of this profile per wavelength.
            **integ_windowed**: ndarray
                Like raw integral but with a Gaussian window giving different 
                weights to pixels along spatial axis
            **weights**:
                Weights in the windowing.
            **residuals_reg**: ndarray
                Residuals from the fit which gives ''amplitude'' attribute.
                            **amplitude**: ndarray
                Estimation of the spectral flux as the amplitude of the Gaussian 
                profile fitted by numpy's linear leastsquare method.
            **integ_model**: ndarray
                Estimation of the spectral flux as the integral of the theoretical
                Gaussian profile which parameters are the amplitude 
                (used in the ''amplitude'' method) and the measured positions and width
                of this profile per wavelength.
            **integ_windowed**: ndarray
                Like raw integral but with a Gaussian window giving different 
                weights to pixels along spatial axis
            **weights**:
                Weights in the windowing.
            **residuals_reg**: ndarray
                Residuals from the fit which gives ''amplitude'' attribute.
        """
        nb_tracks = 16
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
    
#    def getTotalFlux(self):
#        self.fluxes = np.sum(self.slices[:,56:57,:,:], axis=(1,3))
#        self.fluxes = np.array([self.fluxes[:,15], self.fluxes[:,13], self.fluxes[:,2], self.fluxes[:,0]])
#        
#    def getSpectrum(self):
#        self.spectra = np.mean(self.slices, axis=(0,3))
#        self.spectra = np.array([self.spectra[:,15], self.spectra[:,13], self.spectra[:,2], self.spectra[:,0]])
#        self.spectra /= np.sum(self.spectra, axis=1)[:,None]
#        
#    def getPhotoFluctuations(self, spectral_axis, position_poly, width_poly):
#        self.getTotalFlux()
#        self.getSpectrum()
#        self.spectral_fluxes = self.fluxes[:,:,None] * self.spectra[:,None,:]
#        
#        position_poly_photo = [position_poly[15], position_poly[13], position_poly[2], position_poly[0]]
#        width_poly_photo = [width_poly[15], width_poly[13], width_poly[2], width_poly[0]]
#        slices_axes = np.array([self.slices_axes[15], self.slices_axes[13], self.slices_axes[2], self.slices_axes[0]])
#        positions = np.array([p(spectral_axis) for p in position_poly_photo])
#        widths = np.array([p(spectral_axis) for p in width_poly_photo])
#        
#        amplitude = []
#        for i in range(len(position_poly_photo)):
#            temp = []
#            for j in range(len(spectral_axis)):
#                simple_gaus = np.exp(-(slices_axes[i]-positions[i,j])**2/(2*widths[i,j]**2)) # Shape factor of the intensity profile, to be removed before computing Null depth
#                sum_gaus = np.sum(simple_gaus)
#                amp = self.spectral_fluxes[i,:,j] / sum_gaus
#                temp.append(amp)
#            amplitude.append(temp)
#        
#        amplitude = np.array(amplitude)
#        var = (96*20*self.bg_var/self.fluxes[:,None,:]**2 + self.bg_var/(3000*20) * self.spectra[:,:,None]**2) * np.transpose(self.spectral_fluxes, axes=(0,2,1))**2 / sum_gaus**2
#        self.photo_std = var**0.5
#        self.amplitude[:,15,:] = amplitude[0].T
#        self.amplitude[:,13,:] = amplitude[1].T
#        self.amplitude[:,2,:] = amplitude[2].T
#        self.amplitude[:,0,:] = amplitude[3].T

    
    def matchSpectralChannels(self, wl_to_px_coeff, px_to_wl_coeff, which_tracks):
        """
        All tracks are slightly shifted respect to each other.
        Need to define the common wavelength to all of them and create a 
        matching map between the spectral channels of every tracks.

        :Parameters        
            **wl_to_px_coeff**: ndarray
                Polynomial coefficients converting wavelength to pixel position
                along the spectral axis of the frames.
            **px_to_wl_coeff**: ndarray
                Polynomial coefficients converting pixel position to wavelength 
                along the spectral axis of the frames.
            **which_tracks**: array-like
                List of the output from which to measure the flux.
                Outputs are numbered from 0 to 15 from top to bottom 
                (in the way the frame are loaded).            
 
        :Attributes:
            Creates the following attributes
            **wl_scale**: ndarray
                Wavelength scale for each output, in nanometer.
            **px_scale**: ndarray
                Wavelength scale for each output, in pixel.
            
        """
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
        """
        Compute the null depth per spectral channel, per frame, per model, for each output.
        
        :Attributes
            Creates the following attributes
            
            **nullX**: ndarray
                Estimated null depth of the Xth null, based on the ''amplitude'' attribute.
            **nullX_err**: ndarray
                Estimated uncertainty of the estimated Xth null from the ''amplitude'' attribute..
            **null_modelX**: ndarray
                Estimated null depth of the Xth null, based on the ''integ_model'' attribute.
            **null_modelX_err**: ndarray
                Estimated uncertainty of the estimated Xth null from the ''integ_model'' attribute.
            **null_windowedX**: ndarray
                Estimated null depth of the Xth null, based on the ''integ_windowed'' attribute.
            **null_windowedX_err**: ndarray
                Estimated uncertainty of the estimated Xth null from the ''integ_windowed'' attribute.                
            **null_rawX**: ndarray
                Estimated null depth of the Xth null, based on the ''raw'' attribute.
            **null_raw1_err**: ndarray
                Estimated uncertainty of the estimated Xth null from the ''raw'' attribute.   
        """
        
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
        
    def getIntensities(self):
        ''' Measure flux in a spectral channel with the different estimators'''
        """
        Gets the intensity per spectral channel, per frame, per model, for each output.
        
        :Attributes
            **pX**: ndarray
                Estimated flux in the photometric output X=1..4, from the ''amplitude'' attribute.
            **pX_err**: ndarray
                Uncertainty on the estimated flux in the photometric output X
            **IminusX**: ndarray
                Estimated flux in the null output X=1..6, from the ''amplitude'' attribute.
            **IplusX**: ndarray
                Estimated flux in the antinull output X=1..6, from the ''amplitude'' attribute.
            **pX_model**: ndarray
                Estimated flux in the photometric output X=1..4, from the ''integ_model'' attribute.
            **pX_model_err**: ndarray
                Uncertainty on the estimated flux in the photometric output X
            **Iminus_modelX**: ndarray
                Estimated flux in the null output X=1..6, from the ''integ_model'' attribute.
            **Iplus_modelX**: ndarray
                Estimated flux in the antinull output X=1..6, from the ''integ_model'' attribute.
            **pX_windowed**: ndarray
                Estimated flux in the photometric output X=1..4, from the ''integ_windowed'' attribute.
            **pX_windowed_err**: ndarray
                Uncertainty on the estimated flux in the photometric output X
            **Iminus_windowedX**: ndarray
                Estimated flux in the null output X=1..6, from the ''integ_windowed'' attribute.
            **Iplus_windowedX**: ndarray
                Estimated flux in the antinull output X=1..6, from the ''integ_windowed'' attribute.
            **pX_raw**: ndarray
                Estimated flux in the photometric output X=1..4, from the ''raw'' attribute.
            **pX_raw_err**: ndarray
                Uncertainty on the estimated flux in the photometric output X
            **Iminus_rawX**: ndarray
                Estimated flux in the null output X=1..6, from the ''raw'' attribute.
            **Iplus_rawX**: ndarray
                Estimated flux in the antinull output X=1..6, from the ''raw'' attribute.                     
        """
        
        # With amplitude
        self.p1 = self.amplitude[:,15][:,self.px_scale[15]]
        self.p2 = self.amplitude[:,13][:,self.px_scale[13]]
        self.p3 = self.amplitude[:,2][:,self.px_scale[2]]
        self.p4 = self.amplitude[:,0][:,self.px_scale[0]]
        self.Iminus1, self.Iplus1 = self.amplitude[:,11][:,self.px_scale[11]], self.amplitude[:,9][:,self.px_scale[9]]
        self.Iminus2, self.Iplus2 = self.amplitude[:,3][:,self.px_scale[3]], self.amplitude[:,12][:,self.px_scale[12]]
        self.Iminus3, self.Iplus3 = self.amplitude[:,1][:,self.px_scale[1]], self.amplitude[:,14][:,self.px_scale[14]]
        self.Iminus4, self.Iplus4 = self.amplitude[:,6][:,self.px_scale[6]], self.amplitude[:,4][:,self.px_scale[4]]
        self.Iminus5, self.Iplus5 = self.amplitude[:,5][:,self.px_scale[5]], self.amplitude[:,7][:,self.px_scale[7]]
        self.Iminus6, self.Iplus6 = self.amplitude[:,8][:,self.px_scale[8]], self.amplitude[:,10][:,self.px_scale[10]] 
        
#        self.p1_err = self.photo_std[0,self.px_scale[15]]
#        self.p2_err = self.photo_std[0,self.px_scale[13]]
#        self.p3_err = self.photo_std[0,self.px_scale[2]]
#        self.p4_err = self.photo_std[0,self.px_scale[0]]
        self.p1_err = self.p2_err = self.p3_err = self.p4_err = self.bg_std

        # With full gaussian model
        self.p1_model = self.integ_model[:,15,:][:,self.px_scale[15]]
        self.p2_model = self.integ_model[:,13,:][:,self.px_scale[13]]
        self.p3_model = self.integ_model[:,2,:][:,self.px_scale[2]]
        self.p4_model = self.integ_model[:,0,:][:,self.px_scale[0]]
        self.Iminus_model1, self.Iplus_model1 = self.integ_model[:,11][:,self.px_scale[11]], self.integ_model[:,9][:,self.px_scale[9]]
        self.Iminus_model2, self.Iplus_model2 = self.integ_model[:,3][:,self.px_scale[3]], self.integ_model[:,12][:,self.px_scale[12]]
        self.Iminus_model3, self.Iplus_model3 = self.integ_model[:,1][:,self.px_scale[1]], self.integ_model[:,14][:,self.px_scale[14]]
        self.Iminus_model4, self.Iplus_model4 = self.integ_model[:,6][:,self.px_scale[6]], self.integ_model[:,4][:,self.px_scale[4]]
        self.Iminus_model5, self.Iplus_model5 = self.integ_model[:,5][:,self.px_scale[5]], self.integ_model[:,7][:,self.px_scale[7]]
        self.Iminus_model6, self.Iplus_model6 = self.integ_model[:,8][:,self.px_scale[8]], self.integ_model[:,10][:,self.px_scale[10]]
        
        self.p1_model_err = self.p2_model_err = self.p3_model_err = self.p4_model_err = self.raw_err
        
        # With windowed integration
        self.p1_windowed = self.integ_windowed[:,15,:][:,self.px_scale[15]]
        self.p2_windowed = self.integ_windowed[:,13,:][:,self.px_scale[13]]
        self.p3_windowed = self.integ_windowed[:,2,:][:,self.px_scale[2]]
        self.p4_windowed = self.integ_windowed[:,0,:][:,self.px_scale[0]]
        self.Iminus_windowed1, self.Iplus_windowed1 = self.integ_windowed[:,11][:,self.px_scale[11]], self.integ_windowed[:,9][:,self.px_scale[9]]
        self.Iminus_windowed2, self.Iplus_windowed2 = self.integ_windowed[:,3][:,self.px_scale[3]], self.integ_windowed[:,12][:,self.px_scale[12]]
        self.Iminus_windowed3, self.Iplus_windowed3 = self.integ_windowed[:,1][:,self.px_scale[1]], self.integ_windowed[:,14][:,self.px_scale[14]]
        self.Iminus_windowed4, self.Iplus_windowed4 = self.integ_windowed[:,6][:,self.px_scale[6]], self.integ_windowed[:,4][:,self.px_scale[4]]
        self.Iminus_windowed5, self.Iplus_windowed5 = self.integ_windowed[:,5][:,self.px_scale[5]], self.integ_windowed[:,7][:,self.px_scale[7]]
        self.Iminus_windowed6, self.Iplus_windowed6 = self.integ_windowed[:,8][:,self.px_scale[8]], self.integ_windowed[:,10][:,self.px_scale[10]]
        
        self.p1_windowed_err = self.p2_windowed_err = self.p3_windowed_err = self.p4_windowed_err = self.windowed_err
        
        # With raw integration
        self.p1_raw = self.raw[:,15,:][:,self.px_scale[15]]
        self.p2_raw = self.raw[:,13,:][:,self.px_scale[13]]
        self.p3_raw = self.raw[:,2,:][:,self.px_scale[2]]
        self.p4_raw = self.raw[:,0,:][:,self.px_scale[0]]
        self.Iminus_raw1, self.Iplus_raw1 = self.raw[:,11][:,self.px_scale[11]], self.raw[:,9][:,self.px_scale[9]]
        self.Iminus_raw2, self.Iplus_raw2 = self.raw[:,3][:,self.px_scale[3]], self.raw[:,12][:,self.px_scale[12]]
        self.Iminus_raw3, self.Iplus_raw3 = self.raw[:,1][:,self.px_scale[1]], self.raw[:,14][:,self.px_scale[14]]
        self.Iminus_raw4, self.Iplus_raw4 = self.raw[:,6][:,self.px_scale[6]], self.raw[:,4][:,self.px_scale[4]]
        self.Iminus_raw5, self.Iplus_raw5 = self.raw[:,5][:,self.px_scale[5]], self.raw[:,7][:,self.px_scale[7]]
        self.Iminus_raw6, self.Iplus_raw6 = self.raw[:,8][:,self.px_scale[8]], self.raw[:,10][:,self.px_scale[10]]
        
        self.p1_raw_err = self.p2_raw_err = self.p3_raw_err = self.p4_raw_err = self.raw_err        
                
    def save(self, path, date, mode):
        '''
        path : string, path of the file to save. Must contain the name of the file
        date : string, date of the acquisition of the data (YYYY-MM-DD)
        mode : string, which estimator to use
        '''
        """
        Saves intermediate products for further analyses, into HDF5 file format.
        The different intensities and null are gathered into dictionaries, 
        according to which estimator is used.
        
        :Parameters
            **path**: str
                Path of the file to save. Must contain the name of the file.
            **date**: str
                date of the acquisition of the data (YYYY-MM-DD).
            **mode**: str
                Select which estimator to save among ''amplitude'', ''model'',
                ''windowed'', ''raw''.
                
        :Returns
            HDF5 file containing the measured spectral null depths, 
            spectral intensities of each output, for each frames, 
            and their uncertainties.
            
            Keywords identifies the nature of the stored data.
            Comments into the file contains the following attributes:
                *date: date of the acquisition of the data:
                *nbimg: number of frames:
                *array shape: shape of the data into the data sets.
        """
        
        beams_couple = {'null1':'Beams 1/2', 'null2':'Beams 2/3', 'null3':'Beams 1/4',\
                        'null4':'Beams 3/4', 'null5':'Beams 3/1', 'null6':'Beams 4/2'}
        
        if mode == 'amplitude':
            dictio = {'p1':self.p1, 'p1err':self.p1_err,
                      'p2':self.p2, 'p2err':self.p2_err,
                      'p3':self.p3, 'p3err':self.p3_err,
                      'p4':self.p4, 'p4err':self.p4_err,
                      'null1':self.null1, 'null1err':self.null1_err,
                      'null2':self.null2, 'null2err':self.null2_err,
                      'null3':self.null3, 'null3err':self.null3_err,
                      'null4':self.null4, 'null4err':self.null4_err,
                      'null5':self.null5, 'null5err':self.null5_err,
                      'null6':self.null6, 'null6err':self.null6_err,
                      'Iminus1':self.Iminus1, 'Iplus1':self.Iplus1,
                      'Iminus2':self.Iminus2, 'Iplus2':self.Iplus2,
                      'Iminus3':self.Iminus3, 'Iplus3':self.Iplus3,
                      'Iminus4':self.Iminus4, 'Iplus4':self.Iplus4,
                      'Iminus5':self.Iminus5, 'Iplus5':self.Iplus5,
                      'Iminus6':self.Iminus6, 'Iplus6':self.Iplus6}
                        
        elif mode == 'model':
            dictio = {'p1':self.p1_model, 'p1err':self.p1_model_err,
                      'p2':self.p2_model, 'p2err':self.p2_model_err,
                      'p3':self.p3_model, 'p3err':self.p3_model_err,
                      'p4':self.p4_model, 'p4err':self.p4_model_err,
                      'null1':self.null_model1, 'null1err':self.null_model1_err,
                      'null2':self.null_model2, 'null2err':self.null_model2_err,
                      'null3':self.null_model3, 'null3err':self.null_model3_err,
                      'null4':self.null_model4, 'null4err':self.null_model4_err,
                      'null5':self.null_model5, 'null5err':self.null_model5_err,
                      'null6':self.null_model6, 'null6err':self.null_model6_err,
                      'Iminus1':self.Iminus_model1, 'Iplus1':self.Iplus_model1,
                      'Iminus2':self.Iminus_model2, 'Iplus2':self.Iplus_model2,
                      'Iminus3':self.Iminus_model3, 'Iplus3':self.Iplus_model3,
                      'Iminus4':self.Iminus_model4, 'Iplus4':self.Iplus_model4,
                      'Iminus5':self.Iminus_model5, 'Iplus5':self.Iplus_model5,
                      'Iminus6':self.Iminus_model6, 'Iplus6':self.Iplus_model6}
            
        elif mode == 'windowed':
            dictio = {'p1':self.p1_windowed, 'p1err':self.p1_windowed_err,
                      'p2':self.p2_windowed, 'p2err':self.p2_windowed_err,
                      'p3':self.p3_windowed, 'p3err':self.p3_windowed_err,
                      'p4':self.p4_windowed, 'p4err':self.p4_windowed_err,
                      'null1':self.null_windowed1, 'null1err':self.null_windowed1_err,
                      'null2':self.null_windowed2, 'null2err':self.null_windowed2_err,
                      'null3':self.null_windowed3, 'null3err':self.null_windowed3_err,
                      'null4':self.null_windowed4, 'null4err':self.null_windowed4_err,
                      'null5':self.null_windowed5, 'null5err':self.null_windowed5_err,
                      'null6':self.null_windowed6, 'null6err':self.null_windowed6_err,
                      'Iminus1':self.Iminus_windowed1, 'Iplus1':self.Iplus_windowed1,
                      'Iminus2':self.Iminus_windowed2, 'Iplus2':self.Iplus_windowed2,
                      'Iminus3':self.Iminus_windowed3, 'Iplus3':self.Iplus_windowed3,
                      'Iminus4':self.Iminus_windowed4, 'Iplus4':self.Iplus_windowed4,
                      'Iminus5':self.Iminus_windowed5, 'Iplus5':self.Iplus_windowed5,
                      'Iminus6':self.Iminus_windowed6, 'Iplus6':self.Iplus_windowed6}
        else:
            dictio = {'p1':self.p1_raw, 'p1err':self.p1_raw_err,
                      'p2':self.p2_raw, 'p2err':self.p2_raw_err,
                      'p3':self.p3_raw, 'p3err':self.p3_raw_err,
                      'p4':self.p4_raw, 'p4err':self.p4_raw_err,
                      'null1':self.null_raw1, 'null1err':self.null_raw1_err,
                      'null2':self.null_raw2, 'null2err':self.null_raw2_err,
                      'null3':self.null_raw3, 'null3err':self.null_raw3_err,
                      'null4':self.null_raw4, 'null4err':self.null_raw4_err,
                      'null5':self.null_raw5, 'null5err':self.null_raw5_err,
                      'null6':self.null_raw6, 'null6err':self.null_raw6_err,
                      'Iminus1':self.Iminus_raw1, 'Iplus1':self.Iplus_raw1,
                      'Iminus2':self.Iminus_raw2, 'Iplus2':self.Iplus_raw2,
                      'Iminus3':self.Iminus_raw3, 'Iplus3':self.Iplus_raw3,
                      'Iminus4':self.Iminus_raw4, 'Iplus4':self.Iplus_raw4,
                      'Iminus5':self.Iminus_raw5, 'Iplus5':self.Iplus_raw5,
                      'Iminus6':self.Iminus_raw6, 'Iplus6':self.Iplus_raw6}
            
        # Check if saved file exist
        if os.path.exists(path):
            opening_mode = 'w' # Overwright the whole existing file.
        else:
            opening_mode = 'a' # Create a new file at "path"
            
        with h5py.File(path, opening_mode) as f:
            f.attrs['date'] = date
            f.attrs['nbimg'] = self.nbimg
            f.attrs['array shape'] = 'python ndim : (nb frame, wl channel)'
            
            f.create_dataset('wl_scale', data=self.wl_scale.mean(axis=0))
            f['wl_scale'].attrs['comment'] = 'wl in nm'
            
            for key in dictio.keys():
                f.create_dataset(key, data=dictio[key])
                try:
                    f[key].attrs['comment'] = beams_couple[key]
                except KeyError:
                    pass
                
class ChipProperties(Null):
    """
    Class handling the determination of the properties of the chip.
    """
    def getRatioCoeff(self, beam, zeta_coeff):
        """
        Determines the flux ratio between the different outputs
        
        :Parameters
            **beam**: int,
            id (1..4) of the intput.
            **zeta_coeff**: dic
                Dictionary in which the flux ratios  are stored.
                
        :Returns
            **zeta_coeff**: dic
                Same dictionary as required in the parameters, with the new 
                entries set by the ''beam'' parameters.
        """
        beam = int(beam)
        if beam == 1:
            zeta_coeff['b1null1'] = self.Iminus1 / self.p1
            zeta_coeff['b1null5'] = self.Iminus5 / self.p1
            zeta_coeff['b1null3'] = self.Iminus3 / self.p1
            zeta_coeff['b1null7'] = self.Iplus1 / self.p1
            zeta_coeff['b1null11'] = self.Iplus5 / self.p1
            zeta_coeff['b1null9'] = self.Iplus3 / self.p1
            
        elif beam == 2:
            zeta_coeff['b2null1'] = self.Iminus1 / self.p2
            zeta_coeff['b2null2'] = self.Iminus2 / self.p2
            zeta_coeff['b2null6'] = self.Iminus6 / self.p2
            zeta_coeff['b2null7'] = self.Iplus1 / self.p2
            zeta_coeff['b2null8'] = self.Iplus2 / self.p2
            zeta_coeff['b2null12'] = self.Iplus6 / self.p2
            
        elif beam == 3:
            zeta_coeff['b3null5'] = self.Iminus5 / self.p3
            zeta_coeff['b3null2'] = self.Iminus2 / self.p3
            zeta_coeff['b3null4'] = self.Iminus4 / self.p3
            zeta_coeff['b3null11'] = self.Iplus5 / self.p3
            zeta_coeff['b3null8'] = self.Iplus2 / self.p3
            zeta_coeff['b3null10'] = self.Iplus4 / self.p3
            
        elif beam == 4:
            zeta_coeff['b4null3'] = self.Iminus3 / self.p4
            zeta_coeff['b4null6'] = self.Iminus6 / self.p4
            zeta_coeff['b4null4'] = self.Iminus4 / self.p4
            zeta_coeff['b4null9'] = self.Iplus3 / self.p4
            zeta_coeff['b4null12'] = self.Iplus6 / self.p4
            zeta_coeff['b4null10'] = self.Iplus4 / self.p4
            
        else:
            raise AssertionError('No beam selected (beam = 1..4)')  
            
        return zeta_coeff
    