#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library of the ``glint_fitting_gpu6.py``.
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq, least_squares
from timeit import default_timer as time
import h5py
import os
from cupyx.scipy.special.statistics import ndtr
import scipy.special as sp
from scipy.stats import norm
from scipy.linalg import svd
import warnings
from scipy.optimize import OptimizeWarning, minimize

interpolate_kernel = cp.ElementwiseKernel(
    'float32 x_new, raw float32 xp, int32 xp_size, raw float32 yp', 
    'raw float32 y_new',
    
    '''  
    int high = xp_size - 1;
    int low = 0;
    int mid = 0;
    
    while(high - low > 1)
    {
        mid = (high + low)/2;
        
        if (xp[mid] <= x_new)
        {
            low = mid;
        }
        else
        {
            high = mid;
        }
    }
    y_new[i] = yp[low] + (x_new - xp[low])  * (yp[low+1] - yp[low]) / (xp[low+1] - xp[low]);

    if (x_new < xp[0])
    {
         y_new[i] = yp[0];
    }
    else if (x_new > xp[xp_size-1])
    {
         y_new[i] = yp[xp_size-1];
    }
        
    '''
    )

computeCdfCuda = cp.ElementwiseKernel(
    'float32 x_axis, raw float32 rv, float32 rv_sz',
    'raw float32 cdf',
    '''
    int low = 0;
    int high = rv_sz;
    int mid = 0;
    
    while(low < high){
        mid = (low + high) / 2;
        if(rv[mid] <= x_axis){
            low = mid + 1;
        }
        else{
            high = mid;
        }
    }
    cdf[i] = high;
    '''
    )

def computeCdf(absc, data, mode, normed):
    """
    Compute the empirical cumulative density function (CDF) on GPU with CUDA.
    
    :Parameters:
        
        **absc**: array
            Abscissa of the CDF.
        
        **data**: array
            Data used to create the CDF.
        
        **mode**: string
            If ``ccdf``, the survival function (complementary of the CDF) is calculated instead.
        
        **normed**: bool
            If ``True``, the CDF is normed so that the maximum is equal to 1.
        
    :Returns:
        **cdf**: CDF of **data**.
    """
    cdf = cp.zeros(absc.shape, dtype=cp.float32)
    data = cp.asarray(data, dtype=cp.float32)
    absc = cp.asarray(absc, dtype=cp.float32)
    
    data = cp.sort(data)
        
    computeCdfCuda(absc, data, data.size, cdf)
    
    if mode == 'ccdf':
        cdf = data.size - cdf
    
    if normed:
        cdf = cdf/data.size
        
    return cdf
    
def rv_generator_wPDF(bins_cent, pdf, nsamp):
    """
    Random values generator based on the PDF.
    
    :Parameters:
        
        **bins_cent**: array
            Centered bins of the PDF.
        
        **pdf**: array
            Normalized arbitrary PDF to use to generate rv.
        
        **nsamp**: integer
            Number of values to generate.
        
    :Returns:
        **output_samples**: array
            Array of random values generated from the PDF.
    """

    bin_width = bins_cent[1] - bins_cent[0]
    cdf = cp.cumsum(pdf, dtype=cp.float32) * bin_width
    cdf, mask = cp.unique(cdf, True)
    
    cdf_bins_cent = bins_cent[mask]
    cdf_bins_cent = cdf_bins_cent +  bin_width/2.

    rv_uniform = cp.random.rand(nsamp, dtype=cp.float32)
    output_samples = cp.zeros(rv_uniform.shape, dtype=cp.float32)
    interpolate_kernel(rv_uniform, cdf, cdf.size, cdf_bins_cent, output_samples)
    
    return output_samples

def rv_generator(absc, cdf, nsamp):
    """
    Random values generator based on the CDF.
    
    :Parameters:
        
        **absc**: array
                Abscissa of the CDF;
        
        **cdf**: array
            Normalized arbitrary CDF to use to generate rv.
        
        **nsamp**: integer
            Number of values to generate.
        
    :Returns:
        **output_samples**: array
            Array of random values generated from the CDF.
    """

    cdf, mask = cp.unique(cdf, True)    
    cdf_absc = absc[mask]

    rv_uniform = cp.random.rand(nsamp, dtype=cp.float32)
    output_samples = cp.zeros(rv_uniform.shape, dtype=cp.float32)
    interpolate_kernel(rv_uniform, cdf, cdf.size, cdf_absc, output_samples)
    
    return output_samples

def computeCdfCpu(rv, x_axis, normed=True):
    """
    Compute the empirical cumulative density function (CDF) on CPU.
    
    :Parameters:
        
        **rv**: array
            data used to compute the CDF.
        
        **x_axis**: array
            Abscissa of the CDF.
        
        **normed**: bool
            If ``True``, the CDF is normed so that the maximum is equal to 1.
        
    :Returns:
        
        **cdf**: array
            CDF of the **data**.
        
        **mask**: array
            Indexes of cumulated values.
    """
    
    cdf = np.ones(x_axis.size)*rv.size
    temp = np.sort(rv)
    idx = 0
    for i in range(x_axis.size):
#        idx = idx + len(np.where(temp[idx:] <= x_axis[i])[0])
        mask = np.where(temp <= x_axis[i])[0]
        idx = len(mask)

        if len(temp[idx:]) != 0:
            cdf[i] = idx
        else:
            print('pb', i, idx)
            break

    if normed:        
        cdf /= float(rv.size)
        return cdf
    else:
        return cdf, mask

def computeCdfCupy(rv, x_axis):
    """
    Compute the empirical cumulative density function (CDF) on GPU with cupy.
    
    :Parameters:
        
        **rv**: array
            Data used to compute the CDF.
        
        **x_axis**: array
            Abscissa of the CDF.
        
    :Returns:
        
        **cdf**: array
            CDF of **data**.
    """    
    cdf = cp.ones(x_axis.size, dtype=cp.float32)*rv.size
    temp = cp.asarray(rv, dtype=cp.float32)
    temp = cp.sort(rv)
    idx = 0
    for i in range(x_axis.size):
        idx = idx + len(cp.where(temp[idx:] <= x_axis[i])[0])

        if len(temp[idx:]) != 0:
            cdf[i] = idx
        else:
            break
        
    cdf = cdf / rv.size
    
    return cdf
    
def load_data(data, wl_edges, null_key, nulls_to_invert, *args, **kwargs):
    """
    Load data from data file to create the histograms of the null depths and do Monte-Carlo.
    
    :Parameters:
        
        **data**: array
            List of data files.
        
        **wl_edges**: 2-tuple
            Lower and upper bounds of the spectrum to load.
        
        **null_key**: string
            Baseline to load.
        
        **nulls_to_invert**: list
            List of nulls to invert because their null and antinull outputs are swapped.
        
        **args**: extra arguments
            Use dark data to get the error on the null depth.
        
        **kwargs**: extra keyword arguments
            Performs temporal binning of frames.

    :Returns:
        
        **out**: dictionary
            Includes data to use for the fit: flux in (anti-)null and phtometric outputs, errors, wavelengths.

    """
    # Null table for getting the null and associated photometries in the intermediate data
    # Structure = Chosen null:[number of null, photometry A and photometry B]
    null_table = {'null1':[1,1,2], 'null2':[2,2,3], 'null3':[3,1,4], \
                  'null4':[4,3,4], 'null5':[5,3,1], 'null6':[6,4,2]}
    
    indexes = null_table[null_key]

    null_data = []
    Iminus_data = []
    Iplus_data = []
    photo_data = [[],[]]
    photo_err_data = [[],[]]
    wl_scale = []
    
    for d in data:
        with h5py.File(d, 'r') as data_file:
            wl_scale.append(np.array(data_file['wl_scale']))
            
#            null_data.append(np.array(data_file['null%s'%(indexes[0])]))
            Iminus_data.append(np.array(data_file['Iminus%s'%(indexes[0])]))
            Iplus_data.append(np.array(data_file['Iplus%s'%(indexes[0])]))
                
            photo_data[0].append(np.array(data_file['p%s'%(indexes[1])])) # Fill with beam A intensity
            photo_data[1].append(np.array(data_file['p%s'%(indexes[2])])) # Fill with beam B intensity
            photo_err_data[0].append(np.array(data_file['p%serr'%(indexes[1])])) # Fill with beam A error
            photo_err_data[1].append(np.array(data_file['p%serr'%(indexes[2])])) # Fill with beam B error
            
            if 'null%s'%(indexes[0]) in nulls_to_invert:
                n = np.array(data_file['Iplus%s'%(indexes[0])]) / np.array(data_file['Iminus%s'%(indexes[0])])
            else:
                n = np.array(data_file['Iminus%s'%(indexes[0])]) / np.array(data_file['Iplus%s'%(indexes[0])])
            null_data.append(n)


    # Merge data along frame axis
    null_data = [selt for elt in null_data for selt in elt]
    Iminus_data = [selt for elt in Iminus_data for selt in elt]
    Iplus_data = [selt for elt in Iplus_data for selt in elt]
        
    for i in range(2):
        photo_data[i] = [selt for elt in photo_data[i] for selt in elt]
        photo_err_data[i] = [selt for elt in photo_err_data[i] for selt in elt]


    null_data = np.array(null_data)
    Iminus_data = np.array(Iminus_data)
    Iplus_data = np.array(Iplus_data)
    photo_data = np.array(photo_data)
    photo_err_data = np.array(photo_err_data)
    wl_scale = wl_scale[0] #All the wl scale are supposed to be the same, just pick up the first of the list
    mask = np.arange(wl_scale.size)
    
    wl_min, wl_max = wl_edges
    mask = mask[(wl_scale>=wl_min)&(wl_scale <= wl_max)]
    
    if 'flag' in kwargs:
        flags = kwargs['flag']
        mask = mask[flags]
        
    null_data = null_data[:,mask]
    Iminus_data = Iminus_data[:,mask]
    Iplus_data = Iplus_data[:,mask]
    photo_data = photo_data[:,:,mask]
    wl_scale = wl_scale[mask]
    
    null_data = np.transpose(null_data)
    photo_data = np.transpose(photo_data, axes=(0,2,1))
    Iminus_data = np.transpose(Iminus_data)
    Iplus_data = np.transpose(Iplus_data)
    
    if 'frame_binning' in kwargs:
        if not kwargs['frame_binning'] is None:
            if kwargs['frame_binning'] > 1:
                nb_frames_to_bin = int(kwargs['frame_binning'])
                null_data, dummy = binning(null_data, nb_frames_to_bin, axis=1, avg=True)
                photo_data, dummy = binning(photo_data, nb_frames_to_bin, axis=2, avg=True)
                Iminus_data, dummy = binning(Iminus_data, nb_frames_to_bin, axis=1, avg=True)
                Iplus_data, dummy = binning(Iplus_data, nb_frames_to_bin, axis=1, avg=True)
    
    out = {'null':null_data, 'photo':photo_data, 'wl_scale':wl_scale,\
            'photo_err':photo_err_data, 'wl_idx':mask, 'Iminus':Iminus_data, 'Iplus':Iplus_data}
    
    if len(args) > 0:
        null_err_data = getErrorNull(out, args[0])
    else:
        null_err_data = np.zeros(null_data.shape)
    out['null_err'] = null_err_data
    
    return out

def getErrorNull(data_dic, dark_dic):
    """
    Compute the error of the null depth.

    :Parameters:
        
        **data_dic**: dictionary
            Dictionary of the data from ``load_data``.
        
        **dark_dic**: dictionary
            Dictionary of the dark from ``load_data``.

    :Returns:
        **std_null** : array
            Array of the error on the null depths.
    """
    var_Iminus = dark_dic['Iminus'].var(axis=-1)[:,None]
    var_Iplus = dark_dic['Iplus'].var(axis=-1)[:,None]
    Iminus = data_dic['Iminus']
    Iplus = data_dic['Iplus']
    null = data_dic['null']
    
    std_null = (null**2 * (var_Iminus/Iminus**2 + var_Iplus/Iplus**2))**0.5
    return std_null

def getHistogram(data, bins, density, target='cpu'):
    """
    **DISCARDED**
    
    Compute the histogram of the data.

    :Parameters:
        
        **data**: array
            Data from which we want the histogram.
        
        **bins**: array
            Left-edge of the bins of the histogram but the last value which is the right-edge of the last bin.
        
        **density**: bool
            If ``True``, the histogram is normalized as described in documentation of ``np.histogram``.
        
        **target**: string, optional
            Indicates what creates the histograms: the ``cpu`` or the ``gpu``. The default is 'cpu'.

    :Returns:
        **pdf**: array
            PDF of the data.
        
        **bins_cent**: array
            Centered bins of the histogram.
    """
    pdf, bin_edges = np.histogram(data, bins=bins, density=density)
    bins_cent = bin_edges[:-1] + np.diff(bin_edges[:2])/2.
    
    if target == 'gpu':
        pdf, bins_cent = cp.asarray(pdf, dtype=cp.float32), cp.asarray(bins_cent, dtype=cp.float32)
        
    return pdf, bins_cent

def getHistogramOfIntensities(data, bins, split, target='cpu'):
    """
    **DISCARDED**
    
    Compute the histograms of the photometric outputs.

    :Parameters:
        
        **data**: array
            Data from which we want the histogram.
        
        **bins**: array
            Left-edge of the bins of the histogram but the last value which is the right-edge of the last bin.
        
        **split**: ????
        
        **target**: string, optional
            Indicates what creates the histograms: the ``cpu`` or the ``gpu``. The default is 'cpu'.


    :Returns:
        
        **pdf_I_interf**: array
            PDF of the intensities.
        
        **bins_cent**: array
            Centered bins of the histogram.
    """
    pdf_I = [[np.histogram(selt, bins) for selt in elt] for elt in data]
    bin_edges = np.array([[selt[1] for selt in elt] for elt in pdf_I])
    pdf_I = np.array([[selt[0] for selt in elt] for elt in pdf_I])
    
    bin_edges_interf = bin_edges[:,None,:] * split[:,:,:,None]
    pdf_I_interf = pdf_I[:,None,:] / np.sum(pdf_I[:,None,:] * np.diff(bin_edges_interf), axis=-1, keepdims=True)
    
    bins_cent = bin_edges_interf[:,:,:,:-1] + np.diff(bin_edges_interf[:,:,:,:2])/2.
    
    if target=='gpu':
        pdf_I_interf, bins_cent = cp.asarray(pdf_I_interf, dtype=cp.float32), cp.asarray(bins_cent, dtype=cp.float32)
    
    return  pdf_I_interf, bins_cent

def computeNullDepth(na, IA, IB, wavelength, opd, phase_bias, dphase_bias, dark_null, dark_antinull, 
                     zeta_minus_A, zeta_minus_B, zeta_plus_A, zeta_plus_B, spec_chan_width, oversampling_switch, switch_invert_null):
    """
    Compute the null depth from generated random values of photometries, detector noise and OPD. 
    The estimator is the ratio of the null over the antinull fluxes.

    :Parameters:
        
        **na**: float
            Astrophysical null depth.
        
        **IA**: array
            Values of intensity of beam A in the fringe pattern.
        
        **IB**: array
            Values of intensity of beam B in the fringe pattern.
        
        **wavelength** : float
            Wavelength of the fringe pattern.

        **opd**: array
            Value of OPD in nm.

        **phase_bias**: float
            Achromatic phase offset in radian.

        **dphase_bias**: float
            Achromatic phase offset complement in radian (originally supposed to be fitted but now set to 0).

        **dark_null**: array
            Synthetic values of detector noise in the null output.

        **dark_antinull**: array
            Synthetic values of detector noise in the antinull output. 

        **zeta_minus_A**: float
            Value of the zeta coefficient between null and photometric outputs for beam B.

        **zeta_minus_B**: float
            Value of the zeta coefficient between null and photometric outputs for beam B.

        **zeta_plus_A**: float
            Value of the zeta coefficient between antinull and photometric outputs for beam A.

        **zeta_plus_B**: float
            Value of the zeta coefficient between antinull and photometric outputs for beam B.

        **spec_chan_width**: float
            Width of a spectral channel in nm.

        **oversampling_switch**: bool
            If ``True``, the spectral channel is oversampled and averaged to take into account the loss of temporal coherence.

        **switch_invert_null**: bool
            If ``True``, the null and antinull sequences are swapped because they are swapped on real data.

    :Returns:
        
        **null**: array
            Synthetic sequence of null dephts.
        
        **Iminus**: array
            Synthetic sequence of flux in the null output.
        
        **Iplus**: array
            Synthetic sequence of flux in the antinull output.
    """
    
    visibility = (1 - na) / (1 + na)
    wave_number = 1./wavelength
    sine = cp.sin(2*np.pi*wave_number*(opd) + phase_bias + dphase_bias)
    if oversampling_switch:
        delta_wave_number = abs(1/(wavelength + spec_chan_width/2) - 1/(wavelength - spec_chan_width/2))
        arg = np.pi*delta_wave_number * (opd)
        sinc = cp.sin(arg) / arg
        sine = sine * sinc
        
    if switch_invert_null: # Data was recorded with a constant pi shift
        Iminus = IA*zeta_minus_A + IB*zeta_minus_B + \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A*zeta_minus_B) * visibility * sine #+ dark_null
        Iplus = IA*zeta_plus_A + IB*zeta_plus_B - \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B) * visibility * sine #+ dark_antinull
#        Iminus = cp.random.normal(Iminus, Iminus**0.5, size=Iminus.shape)
#        Iplus = cp.random.normal(Iplus, Iplus**0.5, size=Iplus.shape)
        Iminus = Iminus + dark_null
        Iplus = Iplus + dark_antinull
        null = Iplus / Iminus        
    else:
        Iminus = IA*zeta_minus_A + IB*zeta_minus_B - \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A*zeta_minus_B) * visibility * sine #+ dark_null
        Iplus = IA*zeta_plus_A + IB*zeta_plus_B + \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B) * visibility * sine #+ dark_antinull
#        Iminus = cp.random.normal(Iminus, Iminus**0.5, size=Iplus.shape)
#        Iplus = cp.random.normal(Iplus, Iplus**0.5, size=Iplus.shape)
        Iminus = Iminus + dark_null
        Iplus = Iplus + dark_antinull
        null = Iminus / Iplus
    return null, Iminus, Iplus

def computeNullDepthNoAntinull(IA, IB, wavelength, opd, dark_null, dark_antinull, 
                     zeta_minus_A, zeta_minus_B, zeta_plus_A, zeta_plus_B, spec_chan_width, oversampling_switch, switch_invert_null):
    """
    Compute the null depth from generated random values of photometries, detector noise and OPD. 
    The estimator is the ratio of the null over the antinull fluxes.
    The antinull flux is considered as a pure constructive fringe.

    :Parameters:        
        
        **IA**: array
            Values of intensity of beam A in the fringe pattern.
        
        **IB**: array
            Values of intensity of beam B in the fringe pattern.
        
        **wavelength** : float
            Wavelength of the fringe pattern.

        **opd**: array
            Value of OPD in nm.

        **dark_null**: array
            Synthetic values of detector noise in the null output.

        **dark_antinull**: array
            Synthetic values of detector noise in the antinull output. 

        **zeta_minus_A**: float
            Value of the zeta coefficient between null and photometric outputs for beam B.

        **zeta_minus_B**: float
            Value of the zeta coefficient between null and photometric outputs for beam B.

        **zeta_plus_A**: float
            Value of the zeta coefficient between antinull and photometric outputs for beam A.

        **zeta_plus_B**: float
            Value of the zeta coefficient between antinull and photometric outputs for beam B.

        **spec_chan_width**: float
            Width of a spectral channel in nm.

        **oversampling_switch**: bool
            If ``True``, the spectral channel is oversampled and averaged to take into account the loss of temporal coherence.

        **switch_invert_null**: bool
            If ``True``, the null and antinull sequences are swapped because they are swapped on real data.

    :Returns:        
        
        **Iminus**: array
            Synthetic sequence of flux in the null output.
        
        **Iplus**: array
            Synthetic sequence of flux in the antinull output.
    """    
    wave_number = 1./wavelength
    sine = cp.sin(2*np.pi*wave_number*(opd))
    if oversampling_switch:
        delta_wave_number = abs(1/(wavelength + spec_chan_width/2) - 1/(wavelength - spec_chan_width/2))
        arg = np.pi*delta_wave_number * (opd)
        sinc = cp.sin(arg) / arg
        sine = sine * sinc
        
    if switch_invert_null: # Data was recorded with a constant pi shift
        Iminus = IA*zeta_minus_A + IB*zeta_minus_B + \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A*zeta_minus_B)
        Iplus = IA*zeta_plus_A + IB*zeta_plus_B - \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B) * sine + \
            dark_antinull
    else:
        Iminus = IA*zeta_minus_A + IB*zeta_minus_B - \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A*zeta_minus_B) * sine + \
            dark_null
        Iplus = IA*zeta_plus_A + IB*zeta_plus_B + \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B)
    return Iminus, Iplus

def computeNullDepth2(na, IA, IB, wavelength, opd, phase_bias, dphase_bias, dark_null, dark_antinull, 
                     zeta_minus_A, zeta_minus_B, zeta_plus_A, zeta_plus_B, spec_chan_width, oversampling_switch, switch_invert_null, sig_opd):
 
    """
    **DISCARDED**
    
    Compute the null depth from generated random values of photometries, detector noise and OPD. 
    The interferometric term is weighted by the loss of coherence expressed as the exponential form :math:`e^{-(2\pi/\lambda \sigma_{OPD})^2 / 2}`.

    :Parameters:
        
        **na**: float
            Astrophysical null depth.
        
        **IA**: array
            Values of intensity of beam A in the fringe pattern.
        
        **IB**: array
            Values of intensity of beam B in the fringe pattern.
        
        **wavelength** : float
            Wavelength of the fringe pattern.

        **opd**: array
            Value of OPD in nm.

        **dark_null**: array
            Synthetic values of detector noise in the null output.

        **dark_antinull**: array
            Synthetic values of detector noise in the antinull output. 

        **zeta_minus_A**: float
            Value of the zeta coefficient between null and photometric outputs for beam B.

        **zeta_minus_B**: float
            Value of the zeta coefficient between null and photometric outputs for beam B.

        **zeta_plus_A**: float
            Value of the zeta coefficient between antinull and photometric outputs for beam A.

        **zeta_plus_B**: float
            Value of the zeta coefficient between antinull and photometric outputs for beam B.

        **spec_chan_width**: float
            Width of a spectral channel in nm.

        **oversampling_switch**: bool
            If ``True``, the spectral channel is oversampled and averaged to take into account the loss of temporal coherence.

        **switch_invert_null**: bool
            If ``True``, the null and antinull sequences are swapped because they are swapped on real data.
        
        **sig_opd**: float
            Standard deviation of the fluctuations of OPD.

    :Returns:
        
        **null**: array
            Synthetic sequence of null dephts.
        
        **Iminus**: array
            Synthetic sequence of flux in the null output.
        
        **Iplus**: array
            Synthetic sequence of flux in the antinull output.
    """
    visibility = (1 - na) / (1 + na)
    wave_number = 1./wavelength
    sine = cp.sin(2*np.pi*wave_number*(opd) + phase_bias + dphase_bias)
    if oversampling_switch:
        delta_wave_number = abs(1/(wavelength + spec_chan_width/2) - 1/(wavelength - spec_chan_width/2))
        arg = np.pi*delta_wave_number * (opd)
        sinc = cp.sin(arg) / arg
        sine = sine * sinc
        
    if switch_invert_null:
        Iminus = IA*zeta_minus_A + IB*zeta_minus_B + \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A*zeta_minus_B) * visibility * sine * np.exp(-(2*np.pi/wavelength*sig_opd)**2/2) + \
            dark_null
        Iplus = IA*zeta_plus_A + IB*zeta_plus_B - \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B) * visibility *sine * np.exp(-(2*np.pi/wavelength*sig_opd)**2/2) + \
            dark_antinull
        null = Iplus / Iminus        
    else:
        Iminus = IA*zeta_minus_A + IB*zeta_minus_B - \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A*zeta_minus_B) * visibility * sine + \
            dark_null
        Iplus = IA*zeta_plus_A + IB*zeta_plus_B + \
            2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B) * visibility *sine + \
            dark_antinull
        null = Iminus / Iplus
    return null, Iminus, Iplus

def computeNullDepthLinear(na, IA, IB, wavelength, opd, phase_bias, dphase_bias, dark_null, dark_antinull, 
                     zeta_minus_A, zeta_minus_B, zeta_plus_A, zeta_plus_B, spec_chan_width, oversampling_switch, switch_invert_null):

    """
    Compute the null depth from generated random values of photometries, detector noise and OPD. 
    The estimator is the linear expression :math:`N =  N_a + N_{instr}`.

    :Parameters:
        
        **na**: float
            Astrophysical null depth.
        
        **IA**: array
            Values of intensity of beam A in the fringe pattern.
        
        **IB**: array
            Values of intensity of beam B in the fringe pattern.
        
        **wavelength** : float
            Wavelength of the fringe pattern.

        **opd**: array
            Value of OPD in nm.

        **phase_bias**: float
            Achromatic phase offset in radian.

        **dphase_bias**: float
            Achromatic phase offset complement in radian (originally supposed to be fitted but now set to 0).

        **dark_null**: array
            Synthetic values of detector noise in the null output.

        **dark_antinull**: array
            Synthetic values of detector noise in the antinull output. 

        **zeta_minus_A**: float
            Value of the zeta coefficient between null and photometric outputs for beam B.

        **zeta_minus_B**: float
            Value of the zeta coefficient between null and photometric outputs for beam B.

        **zeta_plus_A**: float
            Value of the zeta coefficient between antinull and photometric outputs for beam A.

        **zeta_plus_B**: float
            Value of the zeta coefficient between antinull and photometric outputs for beam B.

        **spec_chan_width**: float
            Width of a spectral channel in nm.

        **oversampling_switch**: bool
            If ``True``, the spectral channel is oversampled and averaged to take into account the loss of temporal coherence.

        **switch_invert_null**: bool
            If ``True``, the null and antinull sequences are swapped because they are swapped on real data.

    :Returns:
    
        **null**: array
            Synthetic sequence of null dephts.
        
        **Iminus**: array
            Synthetic sequence of flux in the null output.
        
        **Iplus**: array
            Synthetic sequence of flux in the antinull output.
    """
    
    astroNull = na
    wave_number = 1./wavelength
    sine = cp.sin(2*np.pi*wave_number*(opd) + phase_bias + dphase_bias)
    if oversampling_switch:
        delta_wave_number = abs(1/(wavelength + spec_chan_width/2) - 1/(wavelength - spec_chan_width/2))
        arg = np.pi*delta_wave_number * (opd)
        sinc = cp.sin(arg) / arg
        sine = sine * sinc
        
    if switch_invert_null:
        Iminus = IA*zeta_minus_A + IB*zeta_minus_B + 2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A*zeta_minus_B) * sine + dark_null
        Iplus = IA*zeta_plus_A + IB*zeta_plus_B - 2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B) * sine + dark_antinull
        null = Iplus / Iminus
    else:
        Iminus = IA*zeta_minus_A + IB*zeta_minus_B - 2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A*zeta_minus_B) * sine + dark_null
        Iplus = IA*zeta_plus_A + IB*zeta_plus_B + 2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B) * sine + dark_antinull  
        null = Iminus / Iplus
    
    return null + astroNull, Iminus, Iplus

def computeHanot(na, IA, IB, wavelength, opd, phase_bias, dphase_bias, dark_null, dark_antinull, 
                     zeta_minus_A, zeta_minus_B, zeta_plus_A, zeta_plus_B, spec_chan_width, oversampling_switch, switch_invert_null):
    """
    Compute the null depth from generated random values of photometries, detector noise and OPD. 
    The estimator is the one used in Hanot et al. (2011)(https://ui.adsabs.harvard.edu/abs/2011ApJ...729..110H/abstract).

    :Parameters:
        
        **na**: float
            Astrophysical null depth.
        
        **IA**: array
            Values of intensity of beam A in the fringe pattern.
        
        **IB**: array
            Values of intensity of beam B in the fringe pattern.
        
        **wavelength** : float
            Wavelength of the fringe pattern.

        **opd**: array
            Value of OPD in nm.

        **phase_bias**: float
            Achromatic phase offset in radian.

        **dphase_bias**: float
            Achromatic phase offset complement in radian (originally supposed to be fitted but now set to 0).

        **dark_null**: array
            Synthetic values of detector noise in the null output.

        **dark_antinull**: array
            Synthetic values of detector noise in the antinull output. 

        **zeta_minus_A**: float
            Value of the zeta coefficient between null and photometric outputs for beam B.

        **zeta_minus_B**: float
            Value of the zeta coefficient between null and photometric outputs for beam B.

        **zeta_plus_A**: float
            Value of the zeta coefficient between antinull and photometric outputs for beam A.

        **zeta_plus_B**: float
            Value of the zeta coefficient between antinull and photometric outputs for beam B.

        **spec_chan_width**: float
            Width of a spectral channel in nm.

        **oversampling_switch**: bool
            If ``True``, the spectral channel is oversampled and averaged to take into account the loss of temporal coherence.

        **switch_invert_null**: bool
            If ``True``, the null and antinull sequences are swapped because they are swapped on real data.

    :Returns:
        
        **null**: array
            Synthetic sequence of null dephts.
        
        **Iminus**: array
            Synthetic sequence of flux in the null output.
        
        **Iplus**: array
            Synthetic sequence of flux in the antinull output.
    """    
    astroNull = na
    wave_number = 1./wavelength
    DeltaPhi = 2*np.pi*wave_number*(opd) + phase_bias + dphase_bias

    if switch_invert_null:
        dI = (IA*zeta_plus_A - IB*zeta_plus_B) / (IA*zeta_plus_A + IB*zeta_plus_B)
        Nb = dark_antinull / (IA*zeta_plus_A + IB*zeta_plus_B)
    else:
        dI = (IA*zeta_minus_A - IB*zeta_minus_B) / (IA*zeta_minus_A + IB*zeta_minus_B)
        Nb = dark_null / (IA*zeta_minus_A + IB*zeta_minus_B)
    
    null = 0.25 * (dI**2 + DeltaPhi**2)
    return null + astroNull + Nb

def computeNullDepthCos(IA, IB, wavelength, offset_opd, dopd, phase_bias, dphase_bias, na, dark_null, dark_antinull, 
                     zeta_minus_A, zeta_minus_B, zeta_plus_A, zeta_plus_B, spec_chan_width, oversampling_switch):
    """
    Compute the null depth from generated random values of photometries, detector noise and OPD. 
    The estimator is the ratio of the null over the antinull fluxes.
    The interferometric term uses a cosine and not a sine function.

    :Parameters:
        
        **na**: float
            Astrophysical null depth.
        
        **IA**: array
            Values of intensity of beam A in the fringe pattern.
        
        **IB**: array
            Values of intensity of beam B in the fringe pattern.
        
        **wavelength** : float
            Wavelength of the fringe pattern.

        **opd**: array
            Value of OPD in nm.

        **phase_bias**: float
            Achromatic phase offset in radian.

        **dphase_bias**: float
            Achromatic phase offset complement in radian (originally supposed to be fitted but now set to 0).

        **dark_null**: array
            Synthetic values of detector noise in the null output.

        **dark_antinull**: array
            Synthetic values of detector noise in the antinull output. 

        **zeta_minus_A**: float
            Value of the zeta coefficient between null and photometric outputs for beam B.

        **zeta_minus_B**: float
            Value of the zeta coefficient between null and photometric outputs for beam B.

        **zeta_plus_A**: float
            Value of the zeta coefficient between antinull and photometric outputs for beam A.

        **zeta_plus_B**: float
            Value of the zeta coefficient between antinull and photometric outputs for beam B.

        **spec_chan_width**: float
            Width of a spectral channel in nm.

        **oversampling_switch**: bool
            If ``True``, the spectral channel is oversampled and averaged to take into account the loss of temporal coherence.

        **switch_invert_null**: bool
            If ``True``, the null and antinull sequences are swapped because they are swapped on real data.

    :Returns:
        
        **null**: array
            Synthetic sequence of null dephts.
    """    
    visibility = (1 - na) / (1 + na)
    wave_number = 1./wavelength
    sine = cp.cos(2*np.pi*wave_number*(offset_opd + dopd) + phase_bias + dphase_bias)
    if oversampling_switch:
        delta_wave_number = abs(1/(wavelength + spec_chan_width/2) - 1/(wavelength - spec_chan_width/2))
        arg = np.pi*delta_wave_number * (offset_opd + dopd)
        sinc = cp.sin(arg) / arg
        sine = sine * sinc
        
    Iminus = IA*zeta_minus_A + IB*zeta_minus_B - \
        2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A*zeta_minus_B) * visibility * sine + \
        dark_null
    Iplus = IA*zeta_plus_A + IB*zeta_plus_B + \
        2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B) * visibility *sine + \
        dark_antinull
    null = Iminus / Iplus
    return null

def computeNullDepthLinearCos(na, IA, IB, wavelength, opd, phase_bias, dphase_bias, dark_null, dark_antinull, 
                     zeta_minus_A, zeta_minus_B, zeta_plus_A, zeta_plus_B, spec_chan_width, oversampling_switch):
    """
    Compute the null depth from generated random values of photometries, detector noise and OPD. 
    The estimator is the linear expression :math:`N =  N_a + N_{instr}`.
    The interferometric term uses a cosine and not a sine function.

    :Parameters:
        **na**: float
            Astrophysical null depth.
        
        **IA**: array
            Values of intensity of beam A in the fringe pattern.
        
        **IB**: array
            Values of intensity of beam B in the fringe pattern.
        
        **wavelength** : float
            Wavelength of the fringe pattern.

        **opd**: array
            Value of OPD in nm.

        **phase_bias**: float
            Achromatic phase offset in radian.

        **dphase_bias**: float
            Achromatic phase offset complement in radian (originally supposed to be fitted but now set to 0).

        **dark_null**: array
            Synthetic values of detector noise in the null output.

        **dark_antinull**: array
            Synthetic values of detector noise in the antinull output. 

        **zeta_minus_A**: float
            Value of the zeta coefficient between null and photometric outputs for beam B.

        **zeta_minus_B**: float
            Value of the zeta coefficient between null and photometric outputs for beam B.

        **zeta_plus_A**: float
            Value of the zeta coefficient between antinull and photometric outputs for beam A.

        **zeta_plus_B**: float
            Value of the zeta coefficient between antinull and photometric outputs for beam B.

        **spec_chan_width**: float
            Width of a spectral channel in nm.

        **oversampling_switch**: bool
            If ``True``, the spectral channel is oversampled and averaged to take into account the loss of temporal coherence.

        **switch_invert_null**: bool
            If ``True``, the null and antinull sequences are swapped because they are swapped on real data.

    :Returns:
        
        **null**: array
            Synthetic sequence of null dephts.
        
        **Iminus**: array
            Synthetic sequence of flux in the null output.
        
        **Iplus**: array
            Synthetic sequence of flux in the antinull output.
    """    
    astroNull = na
    wave_number = 1./wavelength
    cosine = cp.cos(2*np.pi*wave_number*(opd) + phase_bias + dphase_bias)
    if oversampling_switch:
        delta_wave_number = abs(1/(wavelength + spec_chan_width/2) - 1/(wavelength - spec_chan_width/2))
        arg = np.pi*delta_wave_number * (opd)
        sinc = cp.sin(arg) / arg
        cosine = cosine * sinc
        
    Iminus = IA*zeta_minus_A + IB*zeta_minus_B - 2 * np.sqrt(IA * IB) * np.sqrt(zeta_minus_A*zeta_minus_B) * cosine + dark_null
    Iplus = IA*zeta_plus_A + IB*zeta_plus_B + 2 * np.sqrt(IA * IB) * np.sqrt(zeta_plus_A*zeta_plus_B) * cosine + dark_antinull
    null = Iminus / Iplus
    return null + astroNull


def get_zeta_coeff(path, wl_scale, plot=False, **kwargs):
    """
    Interpolate the zeta coefficients for the requested wavelengths.

    :Parameters:
        
        **path**: string
            Path to the zeta coefficients' file.

        **wl_scale**: array
            List of wavelength for which we want the zeta coefficients

        **plot**: bool, optional
            If ``True``, the plot of the interpolated zeta coefficients curve is displayed. The default is False.
    
        **kwargs**: extra keyword arguments
            Bins the zeta coefficient between the specified wavelength in this keyword.

    :Returns:
        
        **coeff_new** : dictionary
            Dictionary of the interpolated zeta coefficients.
    """
    coeff_new = {}
    with h5py.File(path, 'r') as coeff:
        wl = np.array(coeff['wl_scale'])[::-1]
        if 'wl_bounds' in kwargs: # Average zeta coeff in the bandwidth
            wl_bounds = kwargs['wl_bounds']
            wl_scale = wl[(wl>=wl_bounds[0])&(wl<=wl_bounds[1])]
        else:
            pass
            
        for key in coeff.keys():
            if 'wl_bounds' in kwargs: # Average zeta coeff in the bandwidth
                if key != 'wl_scale':
                    interp_zeta = np.interp(wl_scale[::-1], wl, np.array(coeff[key])[::-1])
                    coeff_new[key] = np.array([np.mean(interp_zeta[::-1])])
                else:
                    coeff_new[key] = np.array([wl_scale.mean()])
            else:
                if key != 'wl_scale':
                    interp_zeta = np.interp(wl_scale[::-1], wl, np.array(coeff[key])[::-1])
                    coeff_new[key] = interp_zeta[::-1]
                else:
                    coeff_new[key] = wl_scale
        if plot:
            plt.figure()
            plt.plot(np.array(coeff['wl_scale']), np.array(coeff['b1null1']), 'o-')
            plt.plot(coeff_new['wl_scale'], coeff_new['b1null1'], '+-')
            plt.grid()
            plt.ylim(-1,10)
    
    return coeff_new
 

def getErrorCDF(data_null, data_null_err, null_axis):
    """
    Calculate the error of the CDF. It uses the cupy library.

    :Parameters:
    
        **data_null**: array
            Null depth measurements used to create the CDF.
            
        **data_null_err**: array
            Error on the null depth measurements.
            
        **null_axis**: array
            Abscissa of the CDF.
            

    :Returns:
    
            **std**: cupy array
                Error of the CDF.
    """
    data_null = cp.asarray(data_null)
    data_null_err = cp.asarray(data_null_err)
    null_axis = cp.asarray(null_axis)
    var_null_cdf = cp.zeros(null_axis.size, dtype=cp.float32)
    for k in range(null_axis.size):
        prob = ndtr((null_axis[k]-data_null)/data_null_err)
        variance = cp.sum(prob * (1-prob), axis=-1)
        var_null_cdf[k] = variance / data_null.size**2
                   
    std = cp.sqrt(var_null_cdf)
    return cp.asnumpy(std)

def getErrorPDF(data_null, data_null_err, null_axis):
    """
    Calculate the error of the PDF. It uses the cupy library.

    :Parameters:
    
        **data_null**: array
            Null depth measurements used to create the PDF.
            
        **data_null_err**: array
            Error on the null depth measurements.
            
        **null_axis**: array
            Abscissa of the CDF.
            

    :Returns:
    
            **std**: array
                Error of the PDF.
    """    
    data_null = cp.asarray(data_null)
    data_null_err = cp.asarray(data_null_err)
    null_axis = cp.asarray(null_axis)
    var_null_hist = cp.zeros(null_axis.size-1, dtype=cp.float32)
    for k in range(null_axis.size-1):
        prob = ndtr((null_axis[k+1]-data_null)/data_null_err) - ndtr((null_axis[k]-data_null)/data_null_err)
        variance = cp.sum(prob * (1-prob))
        var_null_hist[k] = variance / data_null.size**2
    
    std = cp.sqrt(var_null_hist)
    std[std==0] = std[std!=0].min()
    return cp.asnumpy(std)
   
def doubleGaussCdf(x, mu1, mu2, sig1, sig2, A):
    """
    Calculate the CDF of the sum of two normal distributions.

    :Parameters:

        **x** : array
            Abscissa of the CDF.
            
        **mu1**: float
            Location parameter of the first normal distribution.
            
        **mu2**: float
            Location parameter of the second normal distribution.
            
        **sig1**: float
            Scale parameter of the first normal distribution.
            
        **sig2**: float
            Scale parameter of the second normal distribution.
            
        **A**: float
            Relative amplitude of the second distribution with respect to the first one.

    :Returns:
        
        Array
            CDF of the double normal distribution.
    """
    return sig1/(sig1+A*sig2) * ndtr((x-mu1)/(sig1)) + A*sig2/(sig1+A*sig2) * ndtr((x-mu2)/(sig2))

def getErrorBinomNorm(pdf, data_size):
    """
    Calculate the error of the PDF knowing the number of elements in a bin is a random value following a binomial distribution.

    :Parameters:

        **pdf** : array
            Normalized PDF which the error is calculated.
            
        **data_size**: integer
            Number of elements used to calculate the PDF.

    :Returns:
    
        **pdf_err** : array
            Error of the PDF.
    """
    pdf_err = ((pdf * (1 - pdf))/(data_size))**0.5 # binom-norm
    pdf_err[pdf_err==0] = pdf_err[pdf_err!=0].min()
    return pdf_err

def getErrorWilson(cdf, data_size, confidence):
    """
    DISCARDED.
    Calculate the error of the CDF following the Wilson estimator (https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval).

    :Parameters:
    
        **cdf** : array
            CDF for which we want the error.
            
        **data_size**: integer
            Number of elements used to calculate the CDF.
            
        **confidence**: float between 0 and 1
            Requested confident interval.
            

    :Returns:
    
        **cdf_err**: array
            Error on the CDF.
    """
    z = norm.ppf((1+confidence)/2)
    cdf_err = z / (1 + z**2/data_size) * np.sqrt(cdf*(1-cdf)/data_size + z**2/(4*data_size**2))# Wilson
    return cdf_err


def rv_gen_doubleGauss(nsamp, mu1, mu2, sig, A, target):
    """
    Random values generator according to a double normal distribution with the same scale factor.
    This function uses cupy to generate the values.
    
    This function can be used to model the phase distribution as a double normal distribution if the fluctuations present two modes.

    :Parameters:
    
        **nsamp** : integer
            Number of random values to generate.
            
        **mu1**: float
            Location parameter of the first normal distribution.
            
        **mu2**: float
            Location parameter of the second normal distribution.
            
        **sig**: float
            Scale parameter of the both normal distributions.
            
        **A**: float
            Relative amplitude of the second normal distribution with respect to the first one.
            
        **target**: string
            If ``target = cpu``, the random values are transferred from the graphic card memory to the RAM.

    :Returns:
    
        **rv** : array or cupy array
            Random values generated according to the double normal distribution.
    """
    x, step = cp.linspace(-2500,2500, 10000, endpoint=False, retstep=True, dtype=cp.float32)
    cdf = doubleGaussCdf(x, mu1, mu2, sig, A)
    cdf = cp.asarray(cdf, dtype=cp.float32)
    if target == 'cpu':
        rv = cp.asnumpy(rv_generator(x, cdf, nsamp))
    else:
        rv = rv_generator(x, cdf, nsamp)
        rv = cp.array(rv, dtype=cp.float32)
    return rv


def _wrap_func(func, xdata, ydata, transform):
    """
    Wrapper called by ``curvefit`` to calculate the cost function to minimize.
    
    Copy/pasted and adpated from https://github.com/scipy/scipy/blob/v1.5.4/scipy/optimize/minpack.py, line 481.

    :Parameters:
    
        **func**: function to fit.
            
        **xdata** : array
            The independent variable where the data is measured.
            Should usually be an M-length sequence or an (k,M)-shaped array for
            functions with k predictors, but can actually be any object.
        
        **ydata**: array
            The dependent data, a length M array - nominally ``f(xdata, ...)``.
            
        **transform**: array
            Weight on the data defined by :math:``1/\sigma`` where :math:``\sigma`` is the error on the y-values.

    :Returns:
    
        Cost function to minimize.
    """
    if transform is None:
        def func_wrapped(params):
            return func(xdata, *params) - ydata
    else:
        def func_wrapped(params):
            return transform * (func(xdata, *params) - ydata)

    return func_wrapped

def curvefit(func, xdata, ydata, p0=None, sigma=None, bounds=(None, None), diff_step=None, x_scale=1):
    """
    Adaptation from the Scipy wrapper ``curve_fit``.
    The Scipy wrapper ``curve_fit`` does not give all the outputs of the least_squares function but gives the covariance matrix 
    (not given by the latter function).
    So I create a new wrapper giving both.
    I just copy/paste the code source of the official wrapper (https://github.com/scipy/scipy/blob/v1.5.4/scipy/optimize/minpack.py#L532-L834) 
    and create new outputs for getting the information I need.
    The algorithm is Trust-Reflective-Region.
    
    For exact documentation of the arguments ``diff_step`` and ``x_scale``, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares.

    :Parameters:
    
        **func**: function to fit on data.
            
        **xdata** : array.
            The independent variable where the data is measured.
            Should usually be an M-length sequence or an (k,M)-shaped array for
            functions with k predictors, but can actually be any object.
        
        **ydata**: array
            The dependent data, a length M array - nominally ``f(xdata, ...)``.
            
        **p0**: list, optional
            Initial guess for the parameters (length N). If None, then the
            initial values will all be 1 (if the number of parameters for the
            function can be determined using introspection, otherwise a
            The default is None.
        
        **sigma**: None or M-length sequence, optional. 
            Determines the uncertainty in `ydata`. If we define residuals as
            ``r = ydata - f(xdata, *popt)``. The default is None.
        
        **bounds**: 2-tuple of array_like, optional
            Lower and upper bounds on parameters. Defaults to no bounds.
            Each element of the tuple must be either an array with the length equal
            to the number of parameters, or a scalar (in which case the bound is
            taken to be the same for all parameters). Use ``np.inf`` with an
            appropriate sign to disable bounds on all or some parameters. 
            The default is (None, None).
        
        **diff_step**:  None or scalar or array_like, optional
            Determines the relative step size for the finite difference
            approximation of the Jacobian. The actual step is computed as
            ``x * diff_step``. If None (default), then `diff_step` is taken to be
            a conventional "optimal" power of machine epsilon for the finite
            difference scheme used `William H. Press et. al., “Numerical Recipes. The Art of Scientific Computing. 3rd edition”, Sec. 5.7.`.
            The default is None.
        
        **x_scale**: array_like or scalar, optional
            Characteristic scale of each variable. Setting `x_scale` is equivalent
            to reformulating the problem in scaled variables ``xs = x / x_scale``.
            An alternative view is that the size of a trust region along jth
            dimension is proportional to ``x_scale[j]``. Improved convergence may
            be achieved by setting `x_scale` such that a step of a given size
            along any of the scaled variables has a similar effect on the cost
            function. If set to 'jac', the scale is iteratively updated using the
            inverse norms of the columns of the Jacobian matrix (as described in
            `J. J. More, “The Levenberg-Marquardt Algorithm: Implementation and Theory,” Numerical Analysis, ed. G. A. Watson, Lecture Notes in Mathematics 630, Springer Verlag, pp. 105-116, 1977.`).
            The default is 1.

    :Raises:
    
        **ValueError**: "Unable to determine number of fit parameters". 
            If either `ydata` or `xdata` contain NaNs, or if incompatible options are used.

    :Returns:
    
        **popt**: array
        Optimal values for the parameters so that the sum of the squared
        residuals of ``f(xdata, *popt) - ydata`` is minimized.

        **pcov**: 2-D array
            The estimated covariance of popt. The diagonals provide the variance
            of the parameter estimate. To compute one standard deviation errors
            on the parameters use ``perr = np.sqrt(np.diag(pcov))``.

        **res** : `OptimizeResult` with the following fields defined:
            
            * x : ndarray, shape (n,)
                Solution found.
            * cost : float
                Value of the cost function at the solution.
            * fun : ndarray, shape (m,)
                Vector of residuals at the solution.
            * jac : ndarray, sparse matrix or LinearOperator, shape (m, n)
                Modified Jacobian matrix at the solution, in the sense that J^T J
                is a Gauss-Newton approximation of the Hessian of the cost function.
                The type is the same as the one used by the algorithm.
            * grad : ndarray, shape (m,)
                Gradient of the cost function at the solution.
            * optimality : float
                First-order optimality measure. In unconstrained problems, it is always
                the uniform norm of the gradient. In constrained problems, it is the
                quantity which was compared with `gtol` during iterations.
            * active_mask : ndarray of int, shape (n,)
                Each component shows whether a corresponding constraint is active
                (that is, whether a variable is at the bound):
                    
                    *  0 : a constraint is not active.
                    * -1 : a lower bound is active.
                    *  1 : an upper bound is active.
                    
                Might be somewhat arbitrary for 'trf' method as it generates a sequence
                of strictly feasible iterates and `active_mask` is determined within a
                tolerance threshold.
            * nfev : int
                Number of function evaluations done. Methods 'trf' and 'dogbox' do not
                count function calls for numerical Jacobian approximation, as opposed
                to 'lm' method.
            * njev : int or None
                Number of Jacobian evaluations done. If numerical Jacobian
                approximation is used in 'lm' method, it is set to None.
            * status : int
                The reason for algorithm termination:
                    
                    * -1 : improper input parameters status returned from MINPACK.
                    *  0 : the maximum number of function evaluations is exceeded.
                    *  1 : `gtol` termination condition is satisfied.
                    *  2 : `ftol` termination condition is satisfied.
                    *  3 : `xtol` termination condition is satisfied.
                    *  4 : Both `ftol` and `xtol` termination conditions are satisfied.
                    
            * message : str
                Verbal description of the termination reason.
            * success : bool
                True if one of the convergence criteria is satisfied (`status` > 0).
    """
    ''' '''
    
    if bounds[0] is None:
        bounds[0] = -np.inf
    if bounds[1] is None:
        bounds[1] = np.inf
        
    if p0 is None:
        # determine number of parameters by inspecting the function
        from scipy._lib._util import getargspec_no_self as _getargspec
        args = _getargspec(func)[0]
        if len(args) < 2:
            raise ValueError("Unable to determine number of fit parameters.")
        n = len(args) - 1
        p0 = np.ones(n,)
    else:
        p0 = np.atleast_1d(p0)
        
    if sigma is not None:
        sigma = np.array(sigma)
        transform = 1/sigma
    else:
        transform = None

    cost_func = _wrap_func(func, xdata, ydata, transform)    
    jac = '3-point'
    res = least_squares(cost_func, p0, jac=jac, bounds=bounds, method='trf', diff_step=diff_step, x_scale=x_scale, loss='huber', 
                        verbose=2)#, xtol=None)#, max_nfev=100)
    popt = res.x

    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)
    
    warn_cov = False
    if pcov is None:
        # indeterminate covariance
        pcov = np.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(np.inf)
        warn_cov = True

    if warn_cov:
        warnings.warn('Covariance of the parameters could not be estimated',
                      category=OptimizeWarning)
            
    return popt, pcov, res

def _objective_func(parameters, *args):
    func, xdata, ydata, transform = args
    if transform is None:
        obj_fun = func(xdata, *parameters) - ydata
    else:
        obj_fun = transform * (func(xdata, *parameters) - ydata)
    
    return np.sum(obj_fun**2)
    
def curvefit2(func, xdata, ydata, p0=None, sigma=None):
    """
    New wrapper where one of the algorithm of the Scipy minimize library is used instead of the TRF.
    Documentation based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.

    :Parameters:
    
        **func**: function to fit on data.
            
        **xdata** : array
            The independent variable where the data is measured.
            Should usually be an M-length sequence or an (k,M)-shaped array for
            functions with k predictors, but can actually be any object.
        
        **ydata**: array
            The dependent data, a length M array - nominally ``f(xdata, ...)``.
            
        **p0**: list, optional
            Initial guess for the parameters (length N). If None, then the
            initial values will all be 1 (if the number of parameters for the
            function can be determined using introspection, otherwise a
            The default is None.
        
        **sigma**: None or M-length sequence, optional 
            Determines the uncertainty in `ydata`. If we define residuals as
            ``r = ydata - f(xdata, *popt)``. The default is None.

    :Raises:
    
        **ValueError**: "Unable to determine number of fit parameters" 
            If either `ydata` or `xdata` contain NaNs, or if incompatible options are used.

    :Returns:
        
        **popt**: array
            Optimal values for the parameters so that the sum of the squared
            residuals of ``f(xdata, *popt) - ydata`` is minimized.

        **pcov**: 2-D array
            The estimated covariance of popt. The diagonals provide the variance
            of the parameter estimate. To compute one standard deviation errors
            on the parameters use ``perr = np.sqrt(np.diag(pcov))``.
    
        **res**: OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.

    """
    ''' '''
    if p0 is None:
        # determine number of parameters by inspecting the function
        from scipy._lib._util import getargspec_no_self as _getargspec
        args = _getargspec(func)[0]
        if len(args) < 2:
            raise ValueError("Unable to determine number of fit parameters.")
        n = len(args) - 1
        p0 = np.ones(n,)
    else:
        p0 = np.atleast_1d(p0)
        n = len(p0)
        
    if sigma is not None:
        sigma = np.array(sigma)
        transform = 1/sigma
    else:
        transform = None

    cost_func = _objective_func
    arguments = (func, xdata, ydata, transform)
    res = minimize(cost_func, p0, args=arguments, method='Powell', options={'disp': True, 'return_all': True},
                   callback=lambda x:print(x[0], x[0], x[0], x[0], x[0]))
    res.x = np.atleast_1d(res.x)
    popt = res.x
    pcov = np.ones((n,n))
    
    return popt, pcov, res


# Yield successive n-sized 
# chunks from l. 
def divide_chunks(l, n): 
	"""
    Yield successive n-sized chunks from l.     

    :Parameters:
    
        **l**: integer
            Size of the list to chunk.
            
        **n**: integer
            Size of a chunk of the list.

    :Yields:
    
        Generator of the chunks.
    """
	# looping till length l 
	for i in range(0, len(l), n): 
		yield l[i:i + n] 

def getInjectionAndSpectrum(photoA, photoB, wl_scale, wl_bounds=(1400,1700)):
    """
    Get the distributions of the broadband injections and the spectra of beams A and B.

    :Parameters:
    
        **photoA**: array-like
            Values of the photometric output of beam A.
        
        **photoB**: array-like
            Values of the photometric output of beam B.
        
        **wl_scale**: array-like
            Wavelength of the spectra in nm.
            
        **wl_bounds** : 2-tuple, optional
            Boundaries between which the spectra are extracted. The wavelengths are expressed in nm. The default is (1400,1700).

    :Returns:
    
        2-tuple of 2-tuple
            The first tuple contains the histograms of the broadband injection of beams A and B, respectively.
            The second tuple contains the spectra of beams A and B, respectively.

    """
    # Select the large bandwidth on which we measure the injection
    idx_wl = np.arange(wl_scale.size)
    idx_wl = idx_wl[(wl_scale>=wl_bounds[0])&(wl_scale<=wl_bounds[1])]
    photoA = photoA[idx_wl]
    photoB = photoB[idx_wl]
    
    # Extract the spectrum
    spectrumA = photoA.mean(axis=1)
    spectrumB = photoB.mean(axis=1)
    spectrumA = spectrumA / spectrumA.sum()
    spectrumB = spectrumB / spectrumB.sum()
    
    # Extract the injection for generating random values
    fluxA = photoA.sum(axis=0)
    fluxB = photoB.sum(axis=0)
    
    return ((fluxA, fluxB),(spectrumA, spectrumB))
    
def binning(arr, binning, axis=0, avg=False):
    """
    Bin elements together
    
    :Parameters:
        
        **arr**: nd-array
            Array containing data to bin
            
        **binning**: int
            Number of frames to bin
            
        **axis**: int
            axis along which the frames are binned
            
        **avg**: bool
            If ``True``, the method returns the average of the binned frame.
            Otherwise, return its sum.
            
    :Attributes:
        Change the attributes
        
        **data**: ndarray 
            datacube
    """
    if binning is None or binning > arr.shape[axis]:
        binning = arr.shape[axis]
        
    shape = arr.shape
    crop = shape[axis]//binning*binning # Number of frames which can be binned respect to the input value
    arr = np.take(arr, np.arange(crop), axis=axis)
    shape = arr.shape
    if axis < 0:
        axis += arr.ndim
    shape = shape[:axis] + (-1, binning) + shape[axis+1:]
    arr = arr.reshape(shape)
    if not avg:
        arr = arr.sum(axis=axis+1)
    else:
        arr = arr.mean(axis=axis+1)
        
    cropped_idx = np.arange(crop).reshape(shape[axis], shape[axis+1])
    
    return arr, cropped_idx

def sortFrames(dic_data, binned_frames, quantile, factor_minus, factor_plus, which_null, plot=False, save_path=''):
    """
    Sigma-clipping to filter the frames which phase is not Gaussian (e.g. because of LWE).
    Fluxes of the null and antinull outputs are analysed in two steps.
    In the first step, for a given output, values between two thresholds are kept.
    The `base` is the upper bound for the antinull output or the lower bound for the null output.
    The base is defined as the median of the measurements which lower than the quantile (typically 10%) of the total sample in the null output,
    and upper for the antinull output.
    The second threshold is defined as the `base` plus or minus the standard deviation of the global sample wieghted by a coefficient.
    In the second step, frames for which both fluxes are kept are saved, the others are discarded.

    :Parameters:
    
        **dic_data**: dictionary
            Contains the extracted data from files by the function ``load_data``.
            
        **binned_frames**: integer
            Number of frames to bin before applying the filter.
            It is used to increase the SNR and exhibit the phase noise over the detector noise.
            
        **quantile**: float between 0 and 1
            first quantile taken to determine the `base` threshold.
            
        **factor_minus**: float
            Factor applied to the std of the null flux to determine the second threshold.
            
        **factor_plus**: float
            Factor applied to the std of the antinull flux to determine the second threshold.
            
        **which_null**: string
            Indicates on which baseline the filter is applied.
            
        **plot** : bool, optional
            If ``True``, it displays the time serie of the binned frames, the thresholds and highlights the filtered frames. 
            The default is False.
            
        **save_path**: string, optional
            Path where the plots is saved in png format (dpi = 300). The default is ''.

    :Returns:
    
        **new_dic**: dictionary
            New dictionary with only the saved data points.
            
        **idx_good_frames**: array
            Index of the kept frames in the input dictionary.
    """
    nb_frames_total = dic_data['Iminus'].shape[1]
    Iminus = dic_data['Iminus'].mean(axis=0)
    Iplus = dic_data['Iplus'].mean(axis=0)
    Iminus, cropped_idx_minus = binning(Iminus, binned_frames, avg=True)
    Iplus, cropped_idx_plus = binning(Iplus, binned_frames, avg=True)
    std_plus = Iplus.std()
    std_minus = Iminus.std()
#    std_plus = std_minus = max(std_plus, std_minus)
    
    x = np.arange(Iminus.size)
    Iminus_quantile = Iminus[Iminus<=np.quantile(Iminus, quantile)]
    Iminus_quantile_med = np.median(Iminus_quantile)
    Iplus_quantile = Iplus[Iplus>=np.quantile(Iplus, 1-quantile)]
    Iplus_quantile_med = np.median(Iplus_quantile)
    idx_plus = np.where(Iplus >= Iplus_quantile_med-factor_plus*std_plus)[0]
    idx_minus = np.where(Iminus <= Iminus_quantile_med+factor_minus*std_minus)[0]
    idx_good_values = np.intersect1d(idx_plus, idx_minus)
    idx_good_frames = np.ravel(cropped_idx_plus[idx_good_values,:])
        
    new_dic = {}
    for key in dic_data.keys():
        new_dic[key] = dic_data[key]
        if dic_data[key].shape[-1] == nb_frames_total:
            new_dic[key] = np.take(new_dic[key], idx_good_frames, axis=-1)

    if plot:
        str_null = which_null.capitalize()
        str_null = str_null[:-1]+' '+str_null[-1]
        plt.figure(figsize=(19.2, 10.8))
        plt.title(str_null + ' %s %s'%(factor_minus, factor_plus), size=20)
        plt.plot(x, Iminus, '.', label='I-')
        plt.plot(x, Iplus, '.', label='I+')
        plt.plot(x, Iplus_quantile_med*np.ones_like(Iplus), 'r--', lw=3)
        plt.plot(x, (Iplus_quantile_med-factor_plus*std_plus)*np.ones_like(Iplus), c='r', lw=3)
        plt.plot(x, Iminus_quantile_med*np.ones_like(Iminus), 'g--', lw=3)
        plt.plot(x, (Iminus_quantile_med+factor_minus*std_minus)*np.ones_like(Iminus), c='g', lw=3)
        plt.plot(x[idx_good_values], Iminus[idx_good_values], '+', label='Selected I-')
        plt.plot(x[idx_good_values], Iplus[idx_good_values], 'x', label='Selected I+')
        plt.grid()
        plt.legend(loc='best', fontsize=25)
        plt.xticks(size=25)
        plt.yticks(size=25)
        plt.ylabel('Intensity (count)', size=30)
        plt.xlabel('Frames', size=30)
        plt.tight_layout()
        string = which_null+'_frame_selection_monitor_%s_%s'%(factor_minus, factor_plus)
        plt.savefig(save_path+string+'.png', dpi=300)

#        width = 6.528
#        height = width / 1.5
#        sz = 12
#        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#        plt.figure(figsize=(width, height))
#        plt.plot(x, Iminus, '.', label=r'I$^{-}$', markersize=sz, c=colors[0])
#        plt.plot(x, Iplus, '.', label=r'I$^{+}$', markersize=sz, c=colors[1])
#        plt.plot(x[idx_good_values], Iminus[idx_good_values], '+', label=r'Selected I$^{-}$', c=colors[8])
#        plt.plot(x[idx_good_values], Iplus[idx_good_values], 'x', label=r'Selected I$^{+}$', c='k')
#        plt.fill_between(x, np.ones_like(x)*(Iplus_quantile_med-factor_plus*std_plus), Iplus.max()*np.ones_like(Iplus), color=colors[2], alpha=0.3, label=r'sigma clipping on I$^{+}$')
#        plt.fill_between(x, np.ones_like(x)*(Iminus_quantile_med+factor_minus*std_minus), Iminus.min()*np.ones_like(Iplus), color=colors[3], alpha=0.3, label=r'sigma clipping on I$^{-}$')
#        plt.grid()
#        plt.legend(loc='center', fontsize=sz, ncol=3, bbox_to_anchor=(0.5, -0.4))
#        plt.annotate('LWE', xy=(110, 15), xytext=(0,18), fontsize=sz, arrowprops ={'width':2, 'headlength':8, 'headwidth':4, 'color':'k'})
#        plt.annotate('LWE', xy=(220, 15), xytext=(0,18), fontsize=sz, arrowprops ={'width':2, 'headlength':8, 'headwidth':4, 'color':'k'})
#        plt.xticks(size=sz)
#        plt.yticks(size=sz)
#        plt.ylabel('Intensity (count)', size=sz+2)
#        plt.xlabel('Frames', size=sz+2)
#        plt.tight_layout()
#        string = 'sigma_clipping_'+which_null+'_frame_selection_monitor_%s_%s'%(factor_minus, factor_plus)
#        plt.savefig(save_path+string+'.png', dpi=300)

#        width = 6.528
#        height = width / 1.618
#        sz = 12
#        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#        plt.figure(figsize=(width, height))
#        plt.plot(x, Iminus, '.', label=r'I$^{-}$', markersize=sz, c=colors[0])
#        plt.plot(x, Iplus, '.', label=r'I$^{+}$', markersize=sz, c=colors[1])
#        plt.grid()
#        plt.annotate('LWE', xy=(110, 15), xytext=(0,18), fontsize=sz, arrowprops ={'width':2, 'headlength':8, 'headwidth':4, 'color':'k'})
#        plt.annotate('LWE', xy=(220, 15), xytext=(0,18), fontsize=sz, arrowprops ={'width':2, 'headlength':8, 'headwidth':4, 'color':'k'})
#        plt.legend(loc='best', fontsize=sz, ncol=2)
#        plt.xticks(size=sz)
#        plt.yticks(size=sz)
#        plt.ylabel('Intensity (count)', size=sz+2)
#        plt.xlabel('Frames', size=sz+2)
#        plt.tight_layout()
#        string = 'lwe_'+which_null+'_frame_selection_monitor_%s_%s'%(factor_minus, factor_plus)
#        plt.savefig(save_path+string+'.png', dpi=300)
         
    return new_dic, idx_good_frames


         
# =============================================================================
# Some test scripts for some functions above
# =============================================================================
if __name__ == '__main__':
#    offset_opd = (0.39999938011169434 - (-1.500000000000056843e-02))*1000
#    phase_bias = -0.9801769079200153
#    a = computeNullDepth(1, 1, 1552, offset_opd, 0, phase_bias, 1, 0, 0, 1, 1, 1, 1)
#    print(a)
#    
#    rv_opd = rv_gen_doubleGauss(1000000, 0, 0+1602/2, 100, 0.5, 'cpu')
#    
#    hist, bin_edges = np.histogram(rv_opd, 1000, density=True)
#        
#    plt.figure()
#    plt.plot(bin_edges[:-1], hist)
#    plt.grid()    

    def model(x, a):
        global counter
        print(counter, a)
        counter += 1
        return a*x

    counter = 1
    slope, offset = 2, 0
    x = np.arange(100)
    y = model(x, slope) + np.random.normal(0, 0.2, x.size)
    yerr = 0.2 * np.ones(y.shape)
    
    x0 = [2.001]
    
    counter = 1
    popt, pcov, res = curvefit(model, x, y, x0, yerr, bounds=([0],[10]))
#    popt3, pcov3 = curve_fit(model, x, y, x0, sigma=yerr, absolute_sigma=True)
    
    chi2 = np.sum((y-model(x, *res.x))**2/yerr**2) * 1/(y.size-res.x.size)
    print('chi2', chi2)
    
    print('--------')
    counter = 1
    popt2, pcov2, res2 = curvefit2(model, x, y, x0, yerr)
    chi2 = np.sum((y-model(x, *popt2))**2/yerr**2) * 1/(y.size-popt2.size)
    print('chi2', chi2)
#    
#    chi2map = []
#    slopes = np.linspace(1.995,2.005,1001)
#    for s in slopes:
#        a = model(x, s)
#        res = y - a
#        chi = np.sum(res**2/yerr**2) * 1/(res.size-1)
#        chi2map.append(chi)
#    chi2map = np.array(chi2map)
#    plt.figure()
#    plt.plot(slopes, chi2map)
#    plt.grid()
#    from scipy.interpolate import interp1d
#    inter = interp1d(slopes, chi2map)