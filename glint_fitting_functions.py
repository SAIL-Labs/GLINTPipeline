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
from cupyx.scipy.special.statistics import ndtr
import scipy.special as sp
from scipy.stats import norm
from scipy.linalg import svd
import warnings
from scipy.optimize import OptimizeWarning

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

def computeCdfCpu(rv, x_axis, normed=True):
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
    
    return 1-cdf
    
def load_data(data, wl_edges, null_key, *args):
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
            
            null_data.append(np.array(data_file['null%s'%(indexes[0])]))
            Iminus_data.append(np.array(data_file['Iminus%s'%(indexes[0])]))
            Iplus_data.append(np.array(data_file['Iplus%s'%(indexes[0])]))
                
            photo_data[0].append(np.array(data_file['p%s'%(indexes[1])])) # Fill with beam A intensity
            photo_data[1].append(np.array(data_file['p%s'%(indexes[2])])) # Fill with beam B intensity
            photo_err_data[0].append(np.array(data_file['p%serr'%(indexes[1])])) # Fill with beam A error
            photo_err_data[1].append(np.array(data_file['p%serr'%(indexes[2])])) # Fill with beam B error        
            

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
    mask = mask[(wl_scale>=wl_min)&(wl_scale < wl_max)]
        
    null_data = null_data[:,mask[0]:mask[-1]+1]
    Iminus_data = Iminus_data[:,mask[0]:mask[-1]+1]
    Iplus_data = Iplus_data[:,mask[0]:mask[-1]+1]
    photo_data = photo_data[:,:,mask[0]:mask[-1]+1]
    wl_scale = wl_scale[mask]
    
    null_data = np.transpose(null_data)
    photo_data = np.transpose(photo_data, axes=(0,2,1))
    Iminus_data = np.transpose(Iminus_data)
    Iplus_data = np.transpose(Iplus_data)
    
    out = {'null':null_data, 'photo':photo_data, 'wl_scale':wl_scale,\
            'photo_err':photo_err_data, 'wl_idx':mask, 'Iminus':Iminus_data, 'Iplus':Iplus_data}
    
    if len(args) > 0:
        null_err_data = getErrorNull(out, args[0])
    else:
        null_err_data = np.zeros(null_data.shape)
    out['null_err'] = null_err_data
    
    return out

def getErrorNull(data_dic, dark_dic):
    var_Iminus = dark_dic['Iminus'].var(axis=-1)[:,None]
    var_Iplus = dark_dic['Iplus'].var(axis=-1)[:,None]
    Iminus = data_dic['Iminus']
    Iplus = data_dic['Iplus']
    null = data_dic['null']
    
    std_null = (null**2 * (var_Iminus/Iminus**2 + var_Iplus/Iplus**2))**0.5
    return std_null

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

def computeNullDepth(IA, IB, wavelength, offset_opd, dopd, phase_bias, dphase_bias, visibility, dark_null, dark_antinull, 
                     zeta_minus_A, zeta_minus_B, zeta_plus_A, zeta_plus_B, step, oversampling_switch):
    
    wave_number = 1./wavelength
    sine = cp.sin(2*np.pi*wave_number*(offset_opd + dopd) + phase_bias + dphase_bias)
    if oversampling_switch:
        delta_wave_number = abs(1/(wavelength + step/2) - 1/(wavelength - step/2))
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

def get_zeta_coeff(path, wl_scale, plot=False):
    coeff_new = {}
    with h5py.File(path, 'r') as coeff:
        wl = np.array(coeff['wl_scale'])[::-1]
        
        for key in coeff.keys():
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
 

def getErrorCDF(data_null, data_null_err, null_axis):
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
   
def doubleGaussCdf(x, mu1, mu2, sig, A):
    return 1/(1+A) * ndtr((x-mu1)/(sig)) + A/(1+A) * ndtr((x-mu2)/(sig))

def getErrorBinomNorm(cdf, data_size, width):
    cdf_err = ((cdf * (1 - cdf*width))/(data_size**width))**0.5 # binom-norm
    cdf_err[cdf_err==0] = cdf_err[cdf_err!=0].min()
    return cdf_err

def getErrorWilson(cdf, data_size, confidence):
    z = norm.ppf((1+confidence)/2)
    cdf_err = z / (1 + z**2/data_size) * np.sqrt(cdf*(1-cdf)/data_size + z**2/(4*data_size**2))# Wilson
    return cdf_err


def rv_gen_doubleGauss(nsamp, mu1, mu2, sig1, A, target):
    x, step = cp.linspace(-2500,2500, 10000, endpoint=False, retstep=True, dtype=cp.float32)
    cdf = doubleGaussCdf(x, mu1, mu2, sig1, A)
    cdf = cp.asarray(cdf, dtype=cp.float32)
    if target == 'cpu':
        rv = cp.asnumpy(rv_generator(x, cdf, nsamp))
    else:
        rv = rv_generator(x, cdf, nsamp)
        rv = cp.array(rv, dtype=cp.float32)
    return rv


def _wrap_func(func, xdata, ydata, transform):
    if transform is None:
        def func_wrapped(params):
            return func(xdata, *params) - ydata
    else:
        def func_wrapped(params):
            return transform * (func(xdata, *params) - ydata)

    return func_wrapped

def curvefit(func, xdata, ydata, p0=None, sigma=None, bounds=(-np.inf,np.inf), diff_step=None, x_scale=1):
    
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
    jac = '2-point'
    res = least_squares(cost_func, p0, jac=jac, bounds=bounds, method='trf', diff_step=diff_step, x_scale=x_scale, loss='huber', 
                        verbose=2)#, xtol=None, max_nfev=100)
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

    def model(x, a, b):
        return a*x + b

    slope, offset = 2, 5
    x = np.arange(100)
    y = slope * x + offset + np.random.normal(0, 0.1, x.size)
    yerr = 0.1 * np.ones(y.shape)
    
    x0 = [1.,1.]
    
    popt, pcov, res = curvefit(model, x, y, x0, yerr, bounds=([0,0],[20,20]))
    
    chi2 = np.sum((y-model(x, *res.x))**2/yerr**2) * 1/(y.size-res.x.size)
    print('chi2', chi2)
  