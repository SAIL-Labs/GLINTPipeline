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
import os
import glint_fitting_functions as gff

   
def MCfunction(null_bins_edges, visibility, mu_opd, sig_opd):
    '''
    For now, this function deals with polychromatic for one baseline
    '''
    global bins_cent_I1, pdf_I1, bins_cent_I2, pdf_I2, wl_scale, kap_l
    global n_loops, n_samp, count
    global pdf_dark, bins_cent_dark
    global nonoise

    sig_opd = abs(sig_opd)
    null_bins_width = null_bins_edges[1] - null_bins_edges[0]

    count += 1
    print(count, visibility, mu_opd, sig_opd)     
    accum_pdf = cp.zeros((wl_scale.size, null_bins_edges.size-1), dtype=cp.float32)
    
    for i in range(n_loops):
        rv_opd = cp.random.normal(mu_opd, sig_opd, n_samp)
        rv_opd = rv_opd.astype(cp.float32)
        
        if nonoise:
            rv_dark_null = cp.zeros((n_samp,), dtype=cp.float32)
            rv_dark_antinull = cp.zeros((n_samp,), dtype=cp.float32)
        else:
            rv_dark_null = gff.rv_generator(bins_cent_dark, pdf_dark, n_samp)
            rv_dark_null = rv_dark_null.astype(cp.float32)
            rv_dark_antinull = gff.rv_generator(bins_cent_dark, pdf_dark, n_samp)
            rv_dark_antinull = rv_dark_antinull.astype(cp.float32)

        for k in range(wl_scale.size):
            ''' Generate random values from these pdf '''
            rv_I1 = gff.rv_generator(bins_cent_I1[k], pdf_I1[k], n_samp)
            rv_I1[rv_I1<0] = 0
            
            rv_I2 = gff.rv_generator(bins_cent_I2[k], pdf_I2[k], n_samp)
            rv_I2[rv_I2<0] = 0
            
    #        mask = (rv_I1 > 0) & (rv_I2 > 0)
    #        rv_I1 = rv_I1[mask]
    #        rv_I2 = rv_I2[mask]
            
            rv_null = gff.computeNullDepth(rv_I1, rv_I2, wl_scale[k], rv_opd, visibility, rv_dark_null, rv_dark_antinull, kap_l[k])
            rv_null = rv_null[~np.isnan(rv_null)] # Remove NaNs
            
            pdf_null = cp.histogram(rv_null.astype(cp.float32), cp.asarray(null_bins_edges, dtype=cp.float32))[0]
            accum_pdf[k] += pdf_null
    
    accum_pdf = accum_pdf / cp.sum(accum_pdf * null_bins_width, axis=-1, keepdims=True)
    
    accum_pdf = cp.asnumpy(accum_pdf)
    return accum_pdf.ravel()

def error_function(params, x, y):
    global count
    count += 1
    print(count, params)
    return y - MCfunction(x, *params)

''' Settings '''  
n_loops = 1000 # number of loops
n_samp = int(1e+7) # number of samples per loop
nonoise = True

# =============================================================================
# Real data
# =============================================================================
''' Import real data '''
datafolder = 'simulation_nonoise/'
root = "/mnt/96980F95980F72D3/glint/"
file_path = root+'reduction/'+datafolder
data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and not 'dark' in f][10:]

data_dark = gff.load_dark(root+'reduction/'+datafolder+'hist_dark.hdf5')
data = gff.load_data(data_list, wl_edges=(1550, 1555))
data_null = np.transpose(data['null'], axes=(0,2,1))
data_photo = np.transpose(data['photo'], axes=(0,2,1))
wl_scale = data['wl_scale'].astype(np.float32)
pdf_dark, bins_dark = data_dark
pdf_dark = pdf_dark / np.sum(pdf_dark * np.diff(bins_dark[:2]))
bins_cent_dark = bins_dark[:-1] + np.diff(bins_dark[:2])/2
pdf_dark, bins_cent_dark = cp.asarray(pdf_dark, dtype=cp.float32), cp.asarray(bins_cent_dark, dtype=cp.float32)
n_bins = int(np.size(data_photo, axis=-1)**0.5)

''' Get the contribution of every beams in their interferometric outputs '''
split = gff.get_splitting_coeff('mock', data['wl_idx'])

kappa_l = np.ones((6, wl_scale.size), dtype=np.float32) * np.pi/4.

#data_I_interf = data_photo[:,None,:,:] * split[:,:,:,None] # Getting its histogram let us to check the scqling of the pdf below due to the splitters

''' Get PDF of intensities injected into interferometric paths'''
pdf_I_interf, bins_cent_I_interf = gff.getHistogramOfIntensities(data_photo, n_bins, split, 'gpu')

#test = list(np.histogram(data_photo[1,0], n_bins))
#test[0] = test[0] / np.sum(test[0]*np.diff(test[1]))
#test[1] = test[1][:-1]+np.diff(test[1][:2])/2
#test2 = list(np.histogram(data_I_interf[1,0,0], n_bins))
#test2[0] = test2[0] / np.sum(test2[0]*np.diff(test2[1]))
#test2[1] = test2[1][:-1]+np.diff(test2[1][:2])/2
#
#plt.figure()
#plt.plot(cp.asnumpy(bins_cent_I_interf[1,0,0]), cp.asnumpy(pdf_I_interf[1,0,0]), 'o-')
#plt.plot(test[1], test[0], '+--')
#plt.plot(test2[1], test2[0], 'x')
#plt.grid()

''' Prepare accumulated and real histogram '''
null_bins_edges, null_bins_width = np.linspace(0, 1, n_bins+1, retstep=True, dtype=np.float32)
null_bins_cent = null_bins_edges[:-1] + null_bins_width/2
null_hist = np.array([[np.histogram(selt, bins=null_bins_edges)[0] for selt in elt] for elt in data_null])
null_hist = null_hist / np.sum(null_hist * null_bins_width, axis=-1)[:,:,None]
null_hist_gpu = cp.asarray(null_hist, dtype=cp.float32)
null_bins_edges_gpu = cp.asarray(null_bins_edges, dtype=cp.float32)

pdf_I1, bins_cent_I1 = pdf_I_interf[0,0], bins_cent_I_interf[0,0]
pdf_I2, bins_cent_I2 = pdf_I_interf[1,0], bins_cent_I_interf[1,0]

#rv_dark_null = gff.rv_generator(bins_cent_I1[0], pdf_I1[0], n_samp)
#rv_dark_null = rv_dark_null.astype(np.float32)
#histo = list(np.histogram(cp.asnumpy(rv_dark_null), 1000))
#histo[0] = histo[0] / np.sum(histo[0]*np.diff(histo[1]))
#
#plt.figure()
#plt.plot(cp.asnumpy(bins_cent_I1[0]), cp.asnumpy(pdf_I1[0]), 'o-')
#plt.plot(histo[1][:-1]+np.diff(histo[1])[:1], histo[0], '+-')
#plt.grid()

count = 0.
mu_opd = 1550/2.
sig_opd = 40.
na = 0.1
visibility = np.array([(1-na)/(1+na)], dtype=np.float32)[0]
kap_l = kappa_l[0]

#start = time()
#z = MCfunction(null_bins_edges, visibility, mu_opd, sig_opd)
#z = np.reshape(z, null_hist[0].shape)
#stop = time()
#print('test', stop-start)
#
#rel_diff = abs(null_hist[0,0]-z[0])/null_hist[0,0]*100
#rel_diff[np.isnan(rel_diff)] = 0.
#
#for w in data['wl_idx']:
#    plt.figure()
#    plt.subplot(211)
#    plt.plot(null_bins_cent, null_hist[0,0], 'o')
#    plt.plot(null_bins_cent, z[0], '+')
#    plt.grid()
#    plt.subplot(212)
#    plt.plot(null_bins_cent, rel_diff, 'o')
#    plt.grid()
#
#print(np.mean(rel_diff[~np.isinf(rel_diff)]), np.std(rel_diff[~np.isinf(rel_diff)]), np.max(rel_diff[~np.isinf(rel_diff)]), np.min(rel_diff[~np.isinf(rel_diff)]))    

''' Model fitting '''
count = 0
initial_guess = [(1-null_bins_edges[np.argmax(null_hist[0,0])])/(1+null_bins_edges[np.argmax(null_hist[0,0])]), mu_opd+0.1, sig_opd+0.1]
initial_guess = np.array(initial_guess, dtype=np.float32)

start = time()
popt = curve_fit(MCfunction, null_bins_edges, null_hist[0].ravel(), p0=initial_guess, epsfcn = null_bins_width)
stop = time()
print('Duration:', stop - start)
#popt = leastsq(error_function, initial_guess, epsfcn = null_bins_width, args=(null_bins_edges, data_hist), full_output=1)
#popt = least_squares(error_function, initial_guess, diff_step = null_bins_width, args=(null_bins_edges, data_hist), verbose=1, method='lm')
#popt = least_squares(error_function, initial_guess, diff_step = null_bins_width, \
#                     bounds=((0.,-np.pi, -np.pi/2.),(1., np.pi, np.pi/2.)), args=(null_bins_edges, data_hist), verbose=2, method='trf')

real_params = np.array([visibility, mu_opd, sig_opd])
rel_err = (real_params - popt[0]) / real_params * 100
print('rel_err', rel_err)
out = MCfunction(null_bins_edges, *popt[0])

z = MCfunction(null_bins_edges, *real_params)

na_opt = (1-popt[0][0])/(1+popt[0][0])
f = plt.figure()
ax = f.add_subplot(111)
plt.title('Histogram of the null depth', size=40)
plt.plot(null_bins_cent, null_hist[0,0], 'o', markersize=10, label='Data')
plt.plot(null_bins_cent, out, '-', lw=5, alpha=0.8, label='Fit')
#plt.plot(null_bins_cent, z, '.', label='Expected')
plt.grid()
plt.legend(loc='best', fontsize=35)
plt.xlabel('Null depth', size=40)
plt.ylabel('Frequency', size=40)
plt.xticks(size=35);plt.yticks(size=35)
txt1 = 'Fitted values:(%.3f s)\n'%(stop-start) + 'Na = %.5f (%.3f%%)'%(na_opt, rel_err[0]) + '\n' + r'$\mu_{OPD} = %.3f$ nm (%.3f%%)'%(popt[0][1], rel_err[1]) + '\n' + r'$\sigma_{OPD} = %.3f$ nm (%.3f%%)'%(popt[0][2], rel_err[2])
txt2 = 'Expected Values:\n' + 'Na = %.5f'%(na) + '\n' + r'$\mu_{OPD} = %.3f$ nm'%(mu_opd) + '\n' + r'$\sigma_{OPD} = %.3f$ nm'%(sig_opd)
plt.text(0.3,0.5, txt2, va='center', fontsize=30, transform = ax.transAxes, bbox=dict(boxstyle="square", facecolor='white'))
plt.text(0.6,0.5, txt1, va='center', fontsize=30, transform = ax.transAxes, bbox=dict(boxstyle="square", facecolor='white'))


chi2 = np.sum((null_hist[0,0] - out)**2/(null_hist[0,0].size-popt[0].size))
khi2 = np.sum((null_hist[0,0] - z)**2/(null_hist[0,0].size-popt[0].size))
print('Chi2', chi2, khi2)