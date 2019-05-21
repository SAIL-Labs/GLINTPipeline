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

computeCdfCuda = cp.ElementwiseKernel(
    'float32 x_axis, raw float32 rv, float32 rv_sz',
    'raw float32 cdf',
    '''
    int low = 0;
    int high = rv_sz-1;
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
    cdf[i] = high
    '''
    )

def computeCdf(rv, x_axis, normed=True):
    cdf = np.ones(x_axis.size)*rv.size
    temp = np.sort(rv)
    idx = 0
    for i in range(x_axis.size):
        idx = idx + len(np.where(temp[idx:] <= x_axis[i])[0])

        if len(temp[idx:]) != 0:
            cdf[i] = idx
        else:
            break

    if normed:        
        cdf /= float(rv.size)
        return 1-cdf
    else:
        return rv.size - cdf

def computeCdfCupy(rv, x_axis):
    cdf = cp.ones(x_axis.size, dtype=cp.float32)*rv.size
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
        
def MCfunction(bins, visibility, mu_opd, sig_opd):
    '''
    For now, this function deals with polychromatic for one baseline
    '''
    global bins_cent_I1, pdf_I1, bins_cent_I2, pdf_I2, wl_scale, kap_l
    global n_samp, count
    global pdf_dark, bins_cent_dark
    global mode, nonoise

    sig_opd = abs(sig_opd)

    count += 1
    print(count, visibility, mu_opd, sig_opd)     
    accum_pdf = cp.zeros((wl_scale.size, bins.size), dtype=cp.float32)
    
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
        rv_null = cp.sort(rv_null)
        
        if mode == 'cuda':
            cdf_null = cp.zeros(bins.shape, dtype=cp.float32)
            computeCdfCuda(cp.asarray(bins), rv_null, rv_null.size, cdf_null)
            cdf_null = 1-cdf_null/rv_null.size
        elif mode == 'cupy':
            cdf_null = computeCdfCupy(rv_null, bins)
        else:
            cdf_null = computeCdf(cp.asnumpy(rv_null), cp.asnumpy(bins))
            cdf_null = cp.asarray(cdf_null)
            
        accum_pdf[k] += cdf_null
    
#    accum_pdf = accum_pdf / cp.sum(accum_pdf * null_bins_width, axis=-1, keepdims=True)
    
    accum_pdf = cp.asnumpy(accum_pdf)
    return accum_pdf.ravel()

def error_function(params, x, y):
    global count
    count += 1
    print(count, params)
    return y - MCfunction(x, *params)

''' Settings '''  
n_samp = int(1e+8) # number of samples per loop
mode = 'cuda'
nonoise = True

# =============================================================================
# Real data
# =============================================================================
''' Import real data '''
datafolder = 'simulation/'
root = "/mnt/96980F95980F72D3/glint/"
file_path = root+'reduction/'+datafolder
data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and not 'dark' in f][:]

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
sz = np.size(np.unique(data_null[0,0]))
null_axis, null_axis_width = np.linspace(0., 1., sz, endpoint=False, retstep=True, dtype=np.float32)

data_null_gpu = cp.asarray(data_null, dtype=cp.float32)
data_null_gpu = cp.sort(data_null_gpu, axis=-1)

null_cdf = []
for o in range(data_null.shape[0]):
    temp = []
    for k in range(data_null.shape[1]):
        if mode == 'cuda':
            cdf = cp.zeros(null_axis.shape, dtype=cp.float32)
            computeCdfCuda(cp.asarray(null_axis, dtype=cp.float32), data_null_gpu[o,k], data_null_gpu[o,k].size, cdf)
            cdf = 1-cdf/data_null_gpu[o,k].size
            temp.append(cp.asnumpy(cdf))
        elif mode == 'cupy':
            cdf = computeCdfCupy(data_null_gpu[o,k], cp.asarray(null_axis, dtype=cp.float32))
            temp.append(cp.asnumpy(cdf))
        else:
            cdf = computeCdf(data_null[o,k], null_axis)
            temp.append(cdf)
    null_cdf.append(temp)
null_cdf = np.array(null_cdf)
        
pdf_I1, bins_cent_I1 = pdf_I_interf[0,0], bins_cent_I_interf[0,0]
pdf_I2, bins_cent_I2 = pdf_I_interf[1,0], bins_cent_I_interf[1,0]

plt.figure();plt.plot(cp.asnumpy(bins_cent_I1[0]), cp.asnumpy(pdf_I1[0]));plt.grid()

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
#z = MCfunction(null_bins_cent, visibility, mu_opd, sig_opd)
#z = np.reshape(z, null_hist[0].shape)
#stop = time()
#print('test', stop-start)
#
#rel_diff = abs((null_cdf[0,0]-z[0])/null_cdf[0,0])*100
#rel_diff[np.isnan(rel_diff)] = 0.
#null_cdf2 = 1 - np.cumsum(null_hist[0,0])/null_hist[0,0].sum()
#rel_diff2 = abs((null_cdf2-z[0])/null_cdf2)*100
#rel_diff2[np.isnan(rel_diff2)] = 0.
#
#for w in data['wl_idx']:
#    plt.figure()
#    plt.subplot(211)
#    plt.plot(null_axis, null_cdf2, 'o')
#    plt.plot(null_axis, null_cdf[0,0], '^')
#    plt.plot(null_axis, z[0], '+')
#    plt.grid()
#    plt.subplot(212)
#    plt.plot(null_axis, rel_diff, 'o')
#    plt.plot(null_axis, rel_diff2, 'x')
#    plt.grid()
#    plt.ylim(-5,105)
#
#print(np.mean(rel_diff[~np.isinf(rel_diff)]), np.std(rel_diff[~np.isinf(rel_diff)]), np.max(rel_diff[~np.isinf(rel_diff)]), np.min(rel_diff[~np.isinf(rel_diff)]))    
#print(np.mean(rel_diff2[~np.isinf(rel_diff2)]), np.std(rel_diff2[~np.isinf(rel_diff2)]), np.max(rel_diff2[~np.isinf(rel_diff2)]), np.min(rel_diff2[~np.isinf(rel_diff2)]))    

''' Model fitting '''
np.random.seed()
count = 0
guess_visi = np.where(null_cdf[0].ravel() == np.max(null_cdf[0].ravel()))[0][-1]
guess_visi = null_axis[guess_visi]
initial_guess = [(1-guess_visi)/(1+guess_visi), mu_opd+0.1, sig_opd+0.1]
initial_guess = np.array(initial_guess, dtype=np.float32)

start = time()
popt = curve_fit(MCfunction, null_axis, null_cdf[0].ravel(), p0=initial_guess, epsfcn = null_axis_width)
stop = time()
print('Duration:', stop - start)
#popt = leastsq(error_function, initial_guess, epsfcn = null_bins_width, args=(null_bins_edges, data_hist), full_output=1)
#popt = least_squares(error_function, initial_guess, diff_step = null_bins_width, args=(null_bins_edges, data_hist), verbose=1, method='lm')
#popt = least_squares(error_function, initial_guess, diff_step = null_bins_width, \
#                     bounds=((0.,-np.pi, -np.pi/2.),(1., np.pi, np.pi/2.)), args=(null_bins_edges, data_hist), verbose=2, method='trf')

real_params = np.array([visibility, mu_opd, sig_opd])
rel_err = (real_params - popt[0]) / real_params * 100
print('rel diff', rel_err)
out = MCfunction(null_axis, *popt[0])


z = MCfunction(null_axis, *real_params)

na_opt = (1-popt[0][0])/(1+popt[0][0])
f = plt.figure()
ax = f.add_subplot(111)
plt.title('Survival function of the null depth', size=40)
plt.semilogy(null_axis, null_cdf[0,0], 'o', markersize=10, label='Data')
plt.semilogy(null_axis, out, '-', lw=5, alpha=0.8, label='Fit')
plt.semilogy(null_axis, z, '.', label='Expected')
plt.grid()
plt.legend(loc='best', fontsize=35)
plt.xlabel('Null depth', size=40)
plt.ylabel('Frequency', size=40)
plt.xticks(size=35);plt.yticks(size=35)
txt1 = 'Fitted values:(Last = %.3f s)\n'%(stop-start) + 'Na = %.5f (%.3f%%)'%(na_opt, rel_err[0]) + '\n' + r'$\mu_{OPD} = %.3f$ nm (%.3f%%)'%(popt[0][1], rel_err[1]) + '\n' + r'$\sigma_{OPD} = %.3f$ nm (%.3f%%)'%(popt[0][2], rel_err[2])
txt2 = 'Expected Values:\n' + 'Na = %.5f'%(na) + '\n' + r'$\mu_{OPD} = %.3f$ nm'%(mu_opd) + '\n' + r'$\sigma_{OPD} = %.3f$ nm'%(sig_opd)
plt.text(0.05,0.6, txt2, va='center', fontsize=30, transform = ax.transAxes, bbox=dict(boxstyle="square", facecolor='white'))
plt.text(0.05,0.3, txt1, va='center', fontsize=30, transform = ax.transAxes, bbox=dict(boxstyle="square", facecolor='white'))

chi2 = np.sum((null_cdf[0,0] - out)**2/(null_cdf[0,0].size-popt[0].size))
khi2 = np.sum((null_cdf[0,0] - z)**2/(null_cdf[0,0].size-popt[0].size))
print('chi2', chi2, khi2)