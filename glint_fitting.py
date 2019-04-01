# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:44:07 2019

@author: mamartinod

Fit model of pdf over measured Na PDF

Requirements:
    - measured values of Na
    - measured values of photometries
    - measured values of dark noise
    - gaussian distribution of phase
    
To do:
    - generate random values from arbitrary PDF for photometries
    - generate random values from arbitrary PDF for dark current
    - compute a sample of Null values from the previous rv
    - create the histogram to fit to the measured Null PDF
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq
from numba import vectorize
from timeit import default_timer as time
import math

def rv_generator(bins_cent, pdf, nsamp):
    '''
    bins_cent : x-axis of the histogram
    pdf : normalized arbitrary pdf to use to generate rv
    '''
    
    bin_width = np.diff(bins_cent[:2])
    cdf = np.cumsum(pdf) * np.diff(bins_cent[:2])
    cdf, mask = np.unique(cdf, True)
    
    cdf_bins_cent = bins_cent[mask]
    cdf_bins_cent = cdf_bins_cent +  bin_width/2.

    out = np.random.rand(nsamp)
    
    output_samples = np.interp(out, cdf, cdf_bins_cent)
    output_samples = output_samples.astype(np.float32)
    
    return output_samples

def getHistogram(data, bins, density):
    pdf, bin_edges = np.histogram(data, bins=bins, density=density)
    bins_cent = bin_edges[:-1] + np.diff(bin_edges[:2])/2.
    return pdf, bins_cent

def computeNullDepth0(I1, I2, phase, visibility, dark):
    Iminus = I1 + I2 - 2 * np.sqrt(I1 * I2) * visibility * np.cos(phase) + dark
    Iplus = I1 + I2 + 2 * np.sqrt(I1 * I2) * visibility * np.cos(phase) + dark
    null = Iminus / Iplus
    return null

@vectorize(['float32(float32, float32, float32, float32, float32)'], target='cuda')
def computeNullDepth(I1, I2, phase, visibility, dark):
    Iminus = I1 + I2 - 2 * math.sqrt(I1 * I2) * visibility * math.cos(phase) + dark
    Iplus = I1 + I2 + 2 * math.sqrt(I1 * I2) * visibility * math.cos(phase) + dark
    null = Iminus / Iplus
    return null

def MCfunction(null_bins_edges, visibility, mu_phase, sig_phase):
    global n_loops, n_samp, bins_cent_I1, pdf_I1, bins_cent_I2, pdf_I2, pdf_dark, bins_cent_dark
    sig_phase = abs(sig_phase)
     
    accum_pdf = []   
    for i in range(n_loops):
        ''' Generate random values from these pdf '''
        rv_I1 = rv_generator(bins_cent_I1, pdf_I1, n_samp)
        rv_I1[rv_I1<0] = 0
        rv_I1 = rv_I1.astype(np.float32)
        
        rv_I2 = rv_generator(bins_cent_I2, pdf_I2, n_samp)
        rv_I2[rv_I2<0] = 0
        rv_I2 = rv_I2.astype(np.float32)
        
        rv_dark = rv_generator(bins_cent_dark, pdf_dark, n_samp)
        rv_dark = rv_dark.astype(np.float32)
        
        rv_phase = np.random.normal(mu_phase, sig_phase, n_samp)
        rv_phase = rv_phase.astype(np.float32)
        
        rv_null = computeNullDepth(rv_I1, rv_I2, rv_phase, visibility, rv_dark)
        rv_null = rv_null[~np.isnan(rv_null)] # Remove NaNs
        
        pdf_null = np.zeros((null_bins_edges.size-1,), dtype=np.float32)
        accum_pdf.append(pdf_null)
    
    accum_pdf = np.array(accum_pdf)
    accum_pdf = np.sum(accum_pdf, axis=0)
    accum_pdf = accum_pdf / np.sum(accum_pdf * null_bins_width)    
    
    return accum_pdf

def error_function(params, x, y):
    global count
    count += 1
    print(count, params)
    return y - MCfunction(x, *params)
    
n_loops = 100 # number of loops
n_samp = int(1e+6) # number of samples per loop
n_bins = 1000
        

''' Generates mock data '''
mu, sig = 0.1, 0.1
dark_mu, dark_sig = 0., 30.

data_I1 = np.random.normal(2., 1., n_samp)
data_I2 = np.random.normal(2., 1., n_samp)
data_phase = np.random.normal(mu, sig, n_samp)
data_dark = np.random.normal(dark_mu, dark_sig, n_samp)
visibility = 0.5
data_Iminus = data_I1 + data_I2 - 2 * np.sqrt(data_I1 * data_I2) * visibility * np.cos(data_phase) + data_dark
data_Iplus = data_I1 + data_I2 + 2 * np.sqrt(data_I1 * data_I2) * visibility * np.cos(data_phase) + data_dark
data_null = data_Iminus / data_Iplus

''' Get PDF and null depth's bins edges'''
pdf_I1, bins_cent_I1 = getHistogram(data_I1, n_bins, True)
pdf_I2, bins_cent_I2 = getHistogram(data_I2, n_bins, True)
pdf_dark, bins_cent_dark = getHistogram(data_dark, n_bins, True)

del data_I1, data_I2, data_dark

''' Prepare accumulated and real histogram '''
null_bins_edges, null_bins_width = np.linspace(0, 1, n_bins+1, retstep=True)
null_bins_cent = null_bins_edges[:-1] + null_bins_width/2
data_hist = np.histogram(data_null, bins=null_bins_edges, density=True)[0]

''' Model fitting '''
count = 0
initial_guess = [(1-null_bins_edges[np.argmax(data_hist)])/(1+null_bins_edges[np.argmax(data_hist)]), 0., 0.1]
popt = leastsq(error_function, initial_guess, epsfcn = null_bins_width, args=(null_bins_edges, data_hist), full_output=1)

real_params = np.array([visibility, mu, sig])
rel_err = (real_params - popt[0]) / real_params * 100
print(rel_err)
start = time()
out = MCfunction(null_bins_edges, visibility, mu, sig)
stop = time()
print(stop - start)

plt.figure()
plt.plot(null_bins_cent, data_hist, 'o')
plt.plot(null_bins_cent, out, '+')
plt.grid()