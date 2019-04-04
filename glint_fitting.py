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
from scipy.optimize import curve_fit, leastsq, least_squares
from numba import vectorize
from timeit import default_timer as time
import math
import h5py
import os

def load_dark(data):
    data_file = h5py.File(data)
    hist = np.array(data_file['dark/histogram'])
    bin_edges = np.array(data_file['dark/bins_edges'])
    del data
    
    return hist, bin_edges
    
def load_data(data):
    null_data = [[],[],[],[],[],[]]
    null_err_data = [[],[],[],[],[],[]]
    photo_data = [[],[],[],[]]
    photo_err_data = [[],[],[],[]]
    wl_scale = []
    
    for d in data:
        with h5py.File(d) as data_file:
            wl_scale.append(np.array(data_file['null1/wl_scale']))
            
            for i in range(6):
                null_data[i].append(np.array(data_file['null%s/null'%(i+1)]))
                null_err_data[i].append(np.array(data_file['null%s/null_err'%(i+1)]))
                
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
        
        if i < 4:
            photo_data[i] = [selt for elt in photo_data[i] for selt in elt]
            photo_err_data[i] = [selt for elt in photo_err_data[i] for selt in elt]
            
    null_data = np.array(null_data)
    null_err_data = np.array(null_err_data)
    photo_data = np.array(photo_data)
    photo_err_data = np.array(photo_err_data)
    wl_scale = np.array(wl_scale)
    
    return {'null':null_data, 'photo':photo_data, 'wl_scale':wl_scale, 'null_err':null_err_data, 'photo_err':photo_err_data}

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

def computeNullDepth0(I1, I2, phase, visibility, dark_null, dark_antinull):
    Iminus = I1 + I2 - 2 * np.sqrt(I1 * I2) * visibility * np.cos(phase) + dark_null
    Iplus = I1 + I2 + 2 * np.sqrt(I1 * I2) * visibility * np.cos(phase) + dark_antinull
    null = Iminus / Iplus
    return null

#@vectorize(['float32(float32, float32, float32, float32, float32, float32)'], target='cuda')
def computeNullDepth(I1, I2, phase, visibility, dark_null, dark_antinull):
    Iminus = I1 + I2 - 2 * np.sqrt(I1 * I2) * visibility * np.cos(phase) + dark_null
    Iplus = I1 + I2 + 2 * np.sqrt(I1 * I2) * visibility * np.cos(phase) + dark_antinull
    null = Iminus / Iplus
    return null

def MCfunction(null_bins_edges, visibility, mu_phase, sig_phase):
    global n_loops, n_samp, bins_cent_I1, pdf_I1, bins_cent_I2, pdf_I2, pdf_dark, bins_cent_dark
    global count
#    global mu_phase, sig_phase
    sig_phase = abs(sig_phase)

    count += 1
    print(count, visibility, mu_phase, sig_phase)     
    accum_pdf = []   
    for i in range(n_loops):
        ''' Generate random values from these pdf '''
        rv_I1 = rv_generator(bins_cent_I1, pdf_I1, n_samp)
        rv_I1[rv_I1<0] = 0
        rv_I1 = rv_I1.astype(np.float32)
        
        rv_I2 = rv_generator(bins_cent_I2, pdf_I2, n_samp)
        rv_I2[rv_I2<0] = 0
        rv_I2 = rv_I2.astype(np.float32)
        
#        mask = (rv_I1 > 0) & (rv_I2 > 0)
#        rv_I1 = rv_I1[mask]
#        rv_I2 = rv_I2[mask]
        
        rv_dark_null = rv_generator(bins_cent_dark, pdf_dark, n_samp)
        rv_dark_null = rv_dark_null.astype(np.float32)
        rv_dark_antinull = rv_generator(bins_cent_dark, pdf_dark, n_samp)
        rv_dark_antinull = rv_dark_antinull.astype(np.float32)
        
        rv_phase = np.random.normal(mu_phase, sig_phase, n_samp)
        rv_phase = rv_phase.astype(np.float32)
        
        rv_null = computeNullDepth(rv_I1, rv_I2, rv_phase, visibility, rv_dark_null, rv_dark_antinull)
        rv_null = rv_null[~np.isnan(rv_null)] # Remove NaNs
        
        pdf_null = np.histogram(rv_null, null_bins_edges)[0]
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
mu_phase, sig_phase = 0.1, 0.1
dark_mu, dark_sig = 0., 10.

data_I1 = np.random.normal(2., 1., n_samp)
data_I2 = np.random.normal(2., 1., n_samp)
data_phase = np.random.normal(mu_phase, sig_phase, n_samp)
data_dark = np.random.normal(dark_mu, dark_sig, n_samp)
data_dark_null = np.random.normal(dark_mu, dark_sig, n_samp)
data_dark_antinull = np.random.normal(dark_mu, dark_sig, n_samp)
visibility = 0.5

data_Iminus = data_I1 + data_I2 - 2 * np.sqrt(data_I1 * data_I2) * visibility * np.cos(data_phase) + data_dark_null
data_Iplus = data_I1 + data_I2 + 2 * np.sqrt(data_I1 * data_I2) * visibility * np.cos(data_phase) + data_dark_antinull
data_null = data_Iminus / data_Iplus

#''' Import real data '''
#datafolder = '201806_alfBoo/'
#root = "C:/glint/"
#file_path = root+'reduction/'+datafolder
#data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and not 'dark' in f]
#
#data_dark = load_dark(root+'reduction/'+datafolder+'hist_dark.hdf5')
#donnees = load_data(data_list)
#data_null = donnees['null'][0,:,53]
#data_I1 = donnees['photo'][0,:,53]
#data_I2 = donnees['photo'][1,:,53]
#pdf_dark, bins_dark = data_dark
#pdf_dark = pdf_dark / np.sum(pdf_dark * np.diff(bins_dark[:2]))
#bins_cent_dark = bins_dark[:-1] + np.diff(bins_dark[:2])/2

''' Get PDF and null depth's bins edges'''
pdf_I1, bins_cent_I1 = getHistogram(data_I1, n_bins, True)
pdf_I2, bins_cent_I2 = getHistogram(data_I2, n_bins, True)
pdf_dark, bins_cent_dark = getHistogram(data_dark, n_bins, True)

del data_I1, data_I2, data_dark

''' Prepare accumulated and real histogram '''
null_bins_edges, null_bins_width = np.linspace(0, 1, n_bins+1, retstep=True)
null_bins_cent = null_bins_edges[:-1] + null_bins_width/2
data_hist = np.histogram(data_null, bins=null_bins_edges, density=True)[0]

#plt.figure()
#plt.plot(null_bins_cent, data_hist, 'o')
#plt.grid()

''' Model fitting '''
count = 0
initial_guess = [(1-null_bins_edges[np.argmax(data_hist)])/(1+null_bins_edges[np.argmax(data_hist)]), 0.1, 0.15]
#initial_guess = [visibility]
popt = curve_fit(MCfunction, null_bins_edges, data_hist, p0=initial_guess, epsfcn = null_bins_width)
#popt = leastsq(error_function, initial_guess, epsfcn = null_bins_width, args=(null_bins_edges, data_hist), full_output=1)
#popt = least_squares(error_function, initial_guess, diff_step = null_bins_width, args=(null_bins_edges, data_hist), verbose=1, method='lm')
#popt = least_squares(error_function, initial_guess, diff_step = null_bins_width, \
#                     bounds=((0.,-np.pi, -np.pi/2.),(1., np.pi, np.pi/2.)), args=(null_bins_edges, data_hist), verbose=2, method='trf')

#real_params = np.array([visibility, mu_phase, sig_phase])
#rel_err = (real_params - popt[0]) / real_params * 100
#print(rel_err)
#start = time()
out = MCfunction(null_bins_edges, *popt[0])
#stop = time()
#print(stop - start)

z = MCfunction(null_bins_edges, visibility, mu_phase, sig_phase)
plt.figure()
plt.plot(null_bins_cent, data_hist, 'o')
plt.plot(null_bins_cent, out, '+')
plt.plot(null_bins_cent, z, '.')
plt.grid()

chi2 = np.sum((data_hist - out)**2/(data_hist.size-popt[0].size))
khi2 = np.sum((data_hist - z)**2/(data_hist.size-popt[0].size))