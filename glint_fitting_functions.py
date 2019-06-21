#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:59:49 2019

@author: mam
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq, least_squares
from timeit import default_timer as time
import h5py
import os

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
    int high = rv_sz - 1;
    int mid = 0;
    
    while(low <= high){
        mid = (low + high) / 2;
        if(rv[mid] <= x_axis){
            low = mid + 1;
        }
        else{
            high = mid - 1;
        }
    }
    cdf[i] = high + 1
    '''
    )

def computeCdf(absc, data, mode, normed):
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
    '''
    bins_cent : x-axis of the histogram
    pdf : normalized arbitrary pdf to use to generate rv
    '''

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
    '''
    absc : cupy-array, x-axis of the histogram
    cdf : cupy-array, normalized arbitrary pdf to use to generate rv
    nsamp : int, number of samples to draw
    '''

    cdf, mask = cp.unique(cdf, True)    
    cdf_absc = absc[mask]

    rv_uniform = cp.random.rand(nsamp, dtype=cp.float32)
    output_samples = cp.zeros(rv_uniform.shape, dtype=cp.float32)
    interpolate_kernel(rv_uniform, cdf, cdf.size, cdf_absc, output_samples)
    
    return output_samples

def load_dark(data, channel):
    data_file = h5py.File(data)
    params = np.array(data_file[channel])[1:]
    data_file.close()
    return params
    
def load_data(data, wl_edges=None):
    null_data = [[],[],[],[],[],[]]
    Iminus_data = [[],[],[],[],[],[]]
    Iplus_data = [[],[],[],[],[],[]]
    null_err_data = [[],[],[],[],[],[]]
    beams_couple = []
    photo_data = [[],[],[],[]]
    photo_err_data = [[],[],[],[]]
    wl_scale = []
    
    for d in data:
        with h5py.File(d) as data_file:
            wl_scale.append(np.array(data_file['wl_scale']))
            
            for i in range(6):
                null_data[i].append(np.array(data_file['null%s/null'%(i+1)]))
                null_err_data[i].append(np.array(data_file['null%s/null_err'%(i+1)]))
                Iminus_data[i].append(np.array(data_file['null%s/Iminus'%(i+1)]))
                Iplus_data[i].append(np.array(data_file['null%s/Iplus'%(i+1)]))
                if data.index(d) == 0: beams_couple.append(data_file['null%s'%(i+1)].attrs['comment'])
                
            photo_data[0].append(np.array(data_file['null1/pA'])) # Fill with beam 1 intensity
            photo_data[1].append(np.array(data_file['null1/pB'])) # Fill with beam 2 intensity
            photo_data[2].append(np.array(data_file['null4/pA'])) # Fill with beam 3 intensity
            photo_data[3].append(np.array(data_file['null4/pB'])) # Fill with beam 4 intensity
            photo_err_data[0].append(np.array(data_file['null1/pA_err'])) # Fill with beam 1 error
            photo_err_data[1].append(np.array(data_file['null1/pB_err'])) # Fill with beam 2 error
            photo_err_data[2].append(np.array(data_file['null4/pA_err'])) # Fill with beam 3 error
            photo_err_data[3].append(np.array(data_file['null4/pB_err'])) # Fill with beam 4 error          
            
    # Merge data along frame axis
    for i in range(6):
        null_data[i] = [selt for elt in null_data[i] for selt in elt]
        null_err_data[i] = [selt for elt in null_err_data[i] for selt in elt]
        Iminus_data[i] = [selt for elt in Iminus_data[i] for selt in elt]
        Iplus_data[i] = [selt for elt in Iplus_data[i] for selt in elt]
        
        if i < 4:
            photo_data[i] = [selt for elt in photo_data[i] for selt in elt]
            photo_err_data[i] = [selt for elt in photo_err_data[i] for selt in elt]
            
    null_data = np.array(null_data)
    null_err_data = np.array(null_err_data)
    Iminus_data = np.array(Iminus_data)
    Iplus_data = np.array(Iplus_data)
    photo_data = np.array(photo_data)
    photo_err_data = np.array(photo_err_data)
    wl_scale = np.array(wl_scale)[0]
    mask = np.arange(wl_scale.size)
    
    if wl_edges != None:
        wl_min, wl_max = wl_edges
        mask = np.arange(wl_scale.size)[(wl_scale>=wl_min)&(wl_scale <= wl_max)]
        
    null_data = null_data[:,:,mask[0]:mask[-1]+1]
    null_err_data = null_err_data[:,:,mask[0]:mask[-1]+1]
    Iminus_data = Iminus_data[:,:,mask[0]:mask[-1]+1]
    Iplus_data = Iplus_data[:,:,mask[0]:mask[-1]+1]
    photo_data = photo_data[:,:,mask[0]:mask[-1]+1]
    wl_scale = wl_scale[mask[0]:mask[-1]+1]
    
    null_data = np.transpose(null_data, axes=(0,2,1))
    photo_data = np.transpose(photo_data, axes=(0,2,1))
    Iminus_data = np.transpose(Iminus_data, axes=(0,2,1))
    Iplus_data = np.transpose(Iplus_data, axes=(0,2,1))
#    null_err_data = np.transpose(null_err_data, axes=(0,2,1))
#    photo_err_data = np.transpose(photo_err_data, axes=(0,2,1))
    
    return {'null':null_data, 'photo':photo_data, 'wl_scale':wl_scale, 'null_err':null_err_data,\
            'photo_err':photo_err_data, 'beams couples':beams_couple, 'wl_idx':mask, 'Iminus':Iminus_data, 'Iplus':Iplus_data}


def getHistogram(data, bins, density, target='cpu'):
    pdf, bin_edges = np.histogram(data, bins=bins, density=density)
    bins_cent = bin_edges[:-1] + np.diff(bin_edges[:2])/2.
    
    if target == 'gpu':
        pdf, bins_cent = cp.asarray(pdf, dtype=cp.float32), cp.asarray(bins_cent, dtype=cp.float32)
        
    return pdf, bins_cent

def getHistogramOfIntensities(data, bins, split, target='cpu'):
    pdf_I = [[np.histogram(selt, bins) for selt in elt] for elt in data]
    bin_edges = np.array([[selt[1] for selt in elt] for elt in pdf_I])
    pdf_I = np.array([[selt[0] for selt in elt] for elt in pdf_I])
    
    bin_edges_interf = bin_edges[:,None,:] * split[:,:,:,None]
    pdf_I_interf = pdf_I[:,None,:] / np.sum(pdf_I[:,None,:] * np.diff(bin_edges_interf), axis=-1, keepdims=True)
    
    bins_cent = bin_edges_interf[:,:,:,:-1] + np.diff(bin_edges_interf[:,:,:,:2])/2.
    
    if target=='gpu':
        pdf_I_interf, bins_cent = cp.asarray(pdf_I_interf, dtype=cp.float32), cp.asarray(bins_cent, dtype=cp.float32)
    
    return  pdf_I_interf, bins_cent

def computeNullDepth(I1, I2, wavelength, opd, visibility, dark_null, dark_antinull, kappa_l):
    wave_number = 1./wavelength
    Iminus = I1*np.sin(kappa_l)**2 + I2*np.cos(kappa_l)**2 - \
        np.sqrt(I1 * I2) * np.sin(2*kappa_l) * visibility * np.sin(2*np.pi*wave_number*opd) + dark_null
    Iplus = I1*np.cos(kappa_l)**2 + I2*np.sin(kappa_l)**2 + \
        np.sqrt(I1 * I2) * np.sin(2*kappa_l) * visibility * np.sin(2*np.pi*wave_number*opd) + dark_antinull
    null = Iminus / Iplus
    return null

def get_splitting_coeff(path, wl_idx):
    if path == 'mock':
        split = np.ones((4,4,96))
    else:
        split = path
        
    split = split[:,:,wl_idx[0]:wl_idx[-1]+1]
        
    return split

def flattop(x):
    y = 2/(np.pi * x**2) * (np.cos(x/2) - np.cos(x))
    y[np.isnan(y)] = 1/np.pi * 0.75
    return y

def pdfDeconvolution(bins, histo_photo, histo_noise, bandwidth, plots=False):
    '''
    pdfs must be normalised by their sum
    '''
    step = bins[1]-bins[0]
    freq = np.fft.fftshift(np.fft.fftfreq(bins.size, step))
    ffthisto = np.fft.fftshift(np.fft.fft(histo_photo))
    fftnoise = np.fft.fftshift(np.fft.fft(histo_noise))
        
    kernel = flattop(bins*bandwidth)
    kernel /= kernel.sum()
    
    fftk = np.fft.fftshift(np.fft.fft(kernel))    
    fftz = fftk * ffthisto
    fftdeconv = fftz / fftnoise

    deconv = np.real(np.fft.ifft(np.fft.fftshift(fftdeconv)))
    
    if plots:
        plt.figure()
        plt.semilogy(freq, abs(ffthisto), label='Data')
        plt.semilogy(freq, abs(fftnoise), label='Noise')
        plt.semilogy(freq, abs(fftk), 'o', label='Kernel')
        plt.semilogy(freq, abs(fftz), '^', label='Kerneled data')
        plt.semilogy(freq, abs(fftdeconv), label='Deconvolved')
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel(r'Frequency (binwidth$^{-1}$)')
        plt.ylabel('Amplitude (AU)')
        
        plt.figure()
        plt.plot(bins, histo_photo/histo_photo.max(), label='Data')
        plt.plot(bins, histo_noise/histo_noise.max(), label='Noise')
        plt.plot(bins, deconv/deconv.max(), label='Deconvolved')
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('Bins')
        plt.ylabel('Count (normalised)')
        
    return deconv

def getErrorNull(data_dic, dark_dic):
    var_Iminus = dark_dic['Iminus'].var(axis=-1)
    var_Iplus = dark_dic['Iplus'].var(axis=-1)
    Iminus = data_dic['Iminus']
    Iplus = data_dic['Iplus']
    null = data_dic['null']
    
    std_null = (null**2 * (var_Iminus/Iminus**2 + var_Iplus/Iplus**2))**0.5
    return std_null
        
    