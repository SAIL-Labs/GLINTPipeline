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
import cupy as cp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq, least_squares
from timeit import default_timer as time
import h5py
import os
import glint_fitting_functions as gff
  
def MCfunction(null_bins_edges, visibility, mu_phase, sig_phase):
    global n_loops, n_samp, bins_cent_I1, pdf_I1, bins_cent_I2, pdf_I2, pdf_dark, bins_cent_dark
    global count
#    global mu_phase, sig_phase
    sig_phase = abs(sig_phase)

    count += 1
    print(count, visibility, mu_phase, sig_phase)     
    accum_pdf = cp.zeros((null_bins_edges.size-1), dtype=cp.float32)   
    for i in range(n_loops):
        ''' Generate random values from these pdf '''
        rv_I1 = gff.rv_generator(bins_cent_I1, pdf_I1, n_samp)
        rv_I1[rv_I1<0] = 0
        rv_I1 = rv_I1.astype(np.float32)
        
        rv_I2 = gff.rv_generator(bins_cent_I2, pdf_I2, n_samp)
        rv_I2[rv_I2<0] = 0
        rv_I2 = rv_I2.astype(np.float32)
        
#        mask = (rv_I1 > 0) & (rv_I2 > 0)
#        rv_I1 = rv_I1[mask]
#        rv_I2 = rv_I2[mask]
        
        rv_dark_null = gff.rv_generator(bins_cent_dark, pdf_dark, n_samp)
        rv_dark_null = rv_dark_null.astype(np.float32)
        rv_dark_antinull = gff.rv_generator(bins_cent_dark, pdf_dark, n_samp)
        rv_dark_antinull = rv_dark_antinull.astype(np.float32)
        
        rv_phase = cp.random.normal(mu_phase, sig_phase, n_samp)
        rv_phase = rv_phase.astype(np.float32)
        
        rv_null = gff.computeNullDepth(rv_I1, rv_I2, rv_phase, visibility, rv_dark_null, rv_dark_antinull)
        rv_null = rv_null[~np.isnan(rv_null)] # Remove NaNs
        
        pdf_null = cp.histogram(rv_null, cp.asarray(null_bins_edges))[0]
        accum_pdf += pdf_null

    accum_pdf = accum_pdf / cp.sum(accum_pdf * null_bins_width)    
    
    return cp.asnumpy(accum_pdf)

def error_function(params, x, y):
    global count
    count += 1
    print(count, params)
    return y - MCfunction(x, *params)
  
n_loops = 100 # number of loops
n_samp = int(1e+8) # number of samples per loop
n_bins = 1000
  
''' Generates mock data '''
mu_phase, sig_phase = 0.1, 0.1
dark_mu, dark_sig = 0., 100.

data_I1 = np.random.normal(800., 80., n_samp)
data_I2 = np.random.normal(500., 50., n_samp)
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
pdf_I1, bins_cent_I1 = gff.getHistogram(data_I1, n_bins, True, 'gpu')
pdf_I2, bins_cent_I2 = gff.getHistogram(data_I2, n_bins, True, 'gpu')
pdf_dark, bins_cent_dark = gff.getHistogram(data_dark, n_bins, True, 'gpu')

del data_I1, data_I2, data_dark

''' Prepare accumulated and real histogram '''
null_bins_edges, null_bins_width = np.linspace(0, 1, n_bins+1, retstep=True)
null_bins_cent = null_bins_edges[:-1] + null_bins_width/2
data_hist = np.histogram(data_null, bins=null_bins_edges, density=True)[0]
data_hist_gpu = cp.asarray(data_hist, dtype=cp.float32)
null_bins_edges_gpu = cp.asarray(null_bins_edges, dtype=cp.float32)


#plt.figure()
#plt.plot(null_bins_cent, data_hist, 'o')
#plt.grid()

''' Model fitting '''
count = 0
initial_guess = [(1-null_bins_edges[np.argmax(data_hist)])/(1+null_bins_edges[np.argmax(data_hist)]), mu_phase+0.1, sig_phase+0.1]
initial_guess = np.array(initial_guess, dtype=cp.float32)

start = time()
popt = curve_fit(MCfunction, null_bins_edges, data_hist, p0=initial_guess, epsfcn = null_bins_width)
stop = time()
print('Duration:', stop - start)
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