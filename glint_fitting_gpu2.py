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
from scipy.special import erf
from itertools import combinations

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
        
#def MCfunction_old(bins, visibility, mu_opd, sig_opd):
#    '''
#    For now, this function deals with polychromatic for one baseline
#    '''
#    global bins_cent_I1, pdf_I1, bins_cent_I2, pdf_I2, wl_scale, kap_l
#    global n_samp, count
#    global pdf_dark, bins_cent_dark
#    global mode, nonoise
#
#    sig_opd = abs(sig_opd)
#
#    count += 1
#    print(count, visibility, mu_opd, sig_opd)     
#    accum_pdf = cp.zeros((wl_scale.size, bins.size), dtype=cp.float32)
#    
#    rv_opd = cp.random.normal(mu_opd, sig_opd, n_samp)
#    rv_opd = rv_opd.astype(cp.float32)
#    
#    if nonoise:
#        rv_dark_null = cp.zeros((n_samp,), dtype=cp.float32)
#        rv_dark_antinull = cp.zeros((n_samp,), dtype=cp.float32)
#    else:
#        rv_dark_null = gff.rv_generator(bins_cent_dark, pdf_dark, n_samp)
#        rv_dark_null = rv_dark_null.astype(cp.float32)
#        rv_dark_antinull = gff.rv_generator(bins_cent_dark, pdf_dark, n_samp)
#        rv_dark_antinull = rv_dark_antinull.astype(cp.float32)
#
#    for k in range(wl_scale.size):
#        ''' Generate random values from these pdf '''
#        rv_I1 = gff.rv_generator(bins_cent_I1[k], pdf_I1[k], n_samp)
#        rv_I1[rv_I1<0] = 0
#        
#        rv_I2 = gff.rv_generator(bins_cent_I2[k], pdf_I2[k], n_samp)
#        rv_I2[rv_I2<0] = 0
#        
##        mask = (rv_I1 > 0) & (rv_I2 > 0)
##        rv_I1 = rv_I1[mask]
##        rv_I2 = rv_I2[mask]
#        
#        rv_null = gff.computeNullDepth(rv_I1, rv_I2, wl_scale[k], rv_opd, visibility, rv_dark_null, rv_dark_antinull, kap_l[k])
#        rv_null = rv_null[~np.isnan(rv_null)] # Remove NaNs
#        rv_null = cp.sort(rv_null)
#        
#        if mode == 'cuda':
#            cdf_null = cp.zeros(bins.shape, dtype=cp.float32)
#            computeCdfCuda(cp.asarray(bins), rv_null, rv_null.size, cdf_null)
#            cdf_null = 1-cdf_null/rv_null.size
#        elif mode == 'cupy':
#            cdf_null = computeCdfCupy(rv_null, bins)
#        else:
#            cdf_null = computeCdf(cp.asnumpy(rv_null), cp.asnumpy(bins))
#            cdf_null = cp.asarray(cdf_null)
#            
#        accum_pdf[k] += cdf_null
#    
##    accum_pdf = accum_pdf / cp.sum(accum_pdf * null_bins_width, axis=-1, keepdims=True)
#    
#    accum_pdf = cp.asnumpy(accum_pdf)
#    return accum_pdf.ravel()

def MCfunction(bins, visibility, mu_opd, sig_opd):
    '''
    For now, this function deals with polychromatic for one baseline
    '''
    global data_IA_axis, cdf_data_IA, data_IB_axis, cdf_data_IB # On GPU
    global n_samp, count, wl_scale, kap_l
    global dark_Iminus_cdf, dark_Iminus_axis, dark_Iplus_cdf, dark_Iplus_axis # On GPU
    global mode, nonoise

    sig_opd = abs(sig_opd)

    count += 1
    print(count, visibility, mu_opd, sig_opd)     
    accum_pdf = cp.zeros((wl_scale.size, bins.size), dtype=cp.float32)
    
    rv_opd = cp.random.normal(mu_opd, sig_opd, n_samp)
    rv_opd = rv_opd.astype(cp.float32)

    for k in range(wl_scale.size):
        if nonoise:
            rv_dark_Iminus = cp.zeros((n_samp,), dtype=cp.float32)
            rv_dark_Iplus = cp.zeros((n_samp,), dtype=cp.float32)
        else:
            rv_dark_Iminus = gff.rv_generator(dark_Iminus_axis[k], dark_Iminus_cdf[k], n_samp)
            rv_dark_Iminus = rv_dark_Iminus.astype(cp.float32)
            rv_dark_Iplus = gff.rv_generator(dark_Iplus_axis[k], dark_Iplus_cdf[k], n_samp)
            rv_dark_Iplus = rv_dark_Iplus.astype(cp.float32)
            
        ''' Generate random values from these pdf '''
        rv_IA = gff.rv_generator(data_IA_axis[k], cdf_data_IA[k], n_samp)
        rv_IA[rv_IA<0] = 0
        
        rv_IB = gff.rv_generator(data_IB_axis[k], cdf_data_IB[k], n_samp)
        rv_IB[rv_IB<0] = 0
        
#        print('Synth IA', rv_IA.max(), rv_IA.min(), rv_IA.mean(), rv_IA.std())
#        print('Synth IB', rv_IB.max(), rv_IB.min(), rv_IB.mean(), rv_IB.std())
        
        rv_null = gff.computeNullDepth(rv_IA, rv_IB, wl_scale[k], rv_opd, visibility, rv_dark_Iminus, rv_dark_Iplus, kap_l[k])
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
nonoise = False

# =============================================================================
# Real data
# =============================================================================
''' Import real data '''
datafolder = 'simulation_lownull/'
root = "/mnt/96980F95980F72D3/glint/"
file_path = root+'reduction/'+datafolder
data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and not 'dark' in f and '0.0' in f][:]
dark_list = [file_path+'dark_0001.hdf5', file_path+'dark_0002.hdf5']

data = gff.load_data(data_list, wl_edges=(1550, 1555))
dark = gff.load_data(dark_list, wl_edges=(1550, 1555))

if nonoise:
    dark['photo'][:] = 0.
    
wl_scale = data['wl_scale']
label_photo = ['p1', 'p2', 'p3', 'p4']

''' Get the contribution of every beams in their interferometric outputs '''
split_l = gff.get_splitting_coeff('mock', data['wl_idx'])
kappa_l = np.ones((6, wl_scale.size), dtype=np.float32) * np.pi/4.

''' Get histogram of fluctuations and of dark current '''
sample_size = data['null'].shape[-1]
beam_idx = []

maxi = data['photo'].max(axis=-1).astype(np.int) + 1
absc = np.array([[np.linspace(-selt, selt, int(sample_size**0.5), endpoint=False) for selt in elt] for elt in maxi])
step = np.mean(np.diff(absc), axis=-1)
bin_edges = absc-step[:,:,None]/2
right_edge = bin_edges[:,:,-1]+step
bin_edges = np.append(bin_edges, right_edge[:,:,None], axis=-1)


histo = np.array([[np.histogram(data['photo'][k,i], bins=bin_edges[k,i])[0] for i in range(len(wl_scale))] for k in range(4)])
histo = histo/histo.sum(axis=-1)[:,:,None]
mean_data, var_data = np.mean(data['photo'], axis=-1), np.var(data['photo'], axis=-1)
mean_dark, var_dark = np.mean(dark['photo'], axis=-1), np.var(dark['photo'], axis=-1)


    
data_photo = data['photo']
data_photo = (data_photo - mean_data[:,:,None]) * ((var_data[:,:,None]-var_dark[:,:,None])/var_data[:,:,None])**0.5 + mean_data[:,:,None] - mean_dark[:,:,None]

combo_idx = [elt for elt in combinations(np.arange(4), 2)]
ggg
''' Model the 6 null depths '''
for j in range(6)[:1]:
    ''' Get histograms of intensities and dark current in the pair of photomoetry outputs '''
    idx = combo_idx[j]
    kappa = kappa_l[j]
    split_IA = split_l[idx[0], idx[1]]
    split_IB = split_l[idx[1], idx[0]]
    
    data_IA, data_IB = data_photo[idx[0]], data_photo[idx[1]]
    print('IA', data_IA.max(), data_IA.min(), data_IA.mean(), data_IA.std())
    print('IB', data_IB.max(), data_IB.min(), data_IB.mean(), data_IB.std())

    data_IA *= split_IA[:,None]
    data_IB *= split_IB[:,None]
    
    dark_Iminus_axis = cp.array([np.linspace(dark['Iminus'][j,i].min(), dark['Iminus'][j,i].max(), np.size(np.unique(dark['Iminus'][j,i])), endpoint=False) for i in range(len(wl_scale))], dtype=cp.float32)
    dark_Iminus_cdf = cp.array([cp.asnumpy(gff.computeCdf(dark_Iminus_axis[i], dark['Iminus'][j,i], 'cdf', True)) for i in range(len(wl_scale))], dtype=cp.float32)
    dark_Iplus_axis = cp.array([np.linspace(dark['Iplus'][j,i].min(), dark['Iplus'][j,i].max(), np.size(np.unique(dark['Iplus'][j,i])), endpoint=False) for i in range(len(wl_scale))], dtype=cp.float32)
    dark_Iplus_cdf = cp.array([cp.asnumpy(gff.computeCdf(dark_Iplus_axis[i], dark['Iplus'][j,i], 'cdf', True)) for i in range(len(wl_scale))], dtype=cp.float32)

    data_IA_axis = cp.array([np.linspace(data_IA[i].min(), data_IA[i].max()+1, np.size(np.unique(data_IA[i]))) for i in range(len(wl_scale))], dtype=cp.float32)
    cdf_data_IA = cp.array([cp.asnumpy(gff.computeCdf(data_IA_axis[i], data_IA[i], 'cdf', True)) for i in range(len(wl_scale))], dtype=cp.float32)
    data_IB_axis = cp.array([np.linspace(data_IB[i].min(), data_IB[i].max(), np.size(np.unique(data_IB[i])), endpoint=False) for i in range(len(wl_scale))], dtype=cp.float32)
    cdf_data_IB = cp.array([cp.asnumpy(gff.computeCdf(data_IB_axis[i], data_IB[i], 'cdf', True)) for i in range(len(wl_scale))], dtype=cp.float32)

#    plt.figure()
#    plt.plot(cp.asnumpy(dark_Iminus_axis[0]), cp.asnumpy(dark_Iminus_cdf[0]))
#    plt.plot(cp.asnumpy(dark_Iplus_axis[0]), cp.asnumpy(dark_Iplus_cdf[0]))
#    plt.grid()
#    plt.figure()
#    plt.plot(cp.asnumpy(data_IA_axis[0]), cp.asnumpy(cdf_data_IA[0]))
#    plt.plot(cp.asnumpy(data_IB_axis[0]), cp.asnumpy(cdf_data_IB[0]))
#    plt.grid()


    ''' Make the survival function '''
    data_null = data['null'][j]        
    sz = np.size(np.unique(data_null[0]))
    null_axis, null_axis_width = np.linspace(0., 1., sz, endpoint=False, retstep=True, dtype=np.float32)
    
    null_cdf = []
    for wl in range(len(wl_scale)):
        data_null_gpu = cp.array(data_null[wl], dtype=cp.float32)
        data_null_gpu = cp.sort(data_null_gpu, axis=-1)
        if mode == 'cuda':
            cdf = gff.computeCdf(null_axis, data_null_gpu, 'ccdf', True)
            null_cdf.append(cp.asnumpy(cdf))
        elif mode == 'cupy':
            cdf = computeCdfCupy(data_null_gpu, cp.asarray(null_axis, dtype=cp.float32))
            null_cdf.append(cp.asnumpy(cdf))
        else:
            cdf = computeCdf(data_null, null_axis)
            null_cdf.append(cdf)
            
    null_cdf = np.array(null_cdf)

    ''' Model fitting '''
    count = 0.
    mu_opd = 1550/4.
    sig_opd = 40. # In nm
    na = 0.01
    visibility = np.array([(1-na)/(1+na)], dtype=np.float32)[0]
    kap_l = kappa
    
    start = time()
    z = MCfunction(null_axis, visibility, mu_opd, sig_opd)
    stop = time()
    print('test', stop-start)
    
    guess_visi = np.where(null_cdf[0].ravel() == np.max(null_cdf[0].ravel()))[0][-1]
    guess_visi = null_axis[guess_visi%null_axis.size]
    initial_guess = [(1-guess_visi)/(1+guess_visi), mu_opd+0.1, sig_opd-0.1]
    initial_guess = np.array(initial_guess, dtype=np.float32)
    
    start = time()
    popt = curve_fit(MCfunction, null_axis, null_cdf.ravel(), p0=initial_guess, epsfcn = null_axis_width)
    stop = time()
    print('Duration:', stop - start)
    out = MCfunction(null_axis, *popt[0])
    
    real_params = np.array([visibility, mu_opd, sig_opd])
    rel_err = (real_params - popt[0]) / real_params * 100
    print('rel diff', rel_err)
    na_opt = (1-popt[0][0])/(1+popt[0][0])
    
    f = plt.figure()
    ax = f.add_subplot(111)
    plt.semilogy(null_axis, null_cdf[0], 'o', markersize=10, label='Data')
    plt.semilogy(null_axis, out.reshape((wl_scale.size,-1))[0], '-', lw=5, alpha=0.8, label='Fit')
    plt.semilogy(null_axis, z.reshape((wl_scale.size,-1))[0], '--', lw=4, alpha=0.8, label='Expected')
    plt.grid()
    plt.legend(loc='lower left', fontsize=35)
    plt.xlabel('Null depth', size=40)
    plt.ylabel('Frequency', size=40)
    plt.xticks(size=35);plt.yticks(size=35)
    txt1 = 'Fitted values:(Last = %.3f s)\n'%(stop-start) + 'Na = %.5f (%.3f%%)'%(na_opt, rel_err[0]) + '\n' + r'$\mu_{OPD} = %.3f$ nm (%.3f%%)'%(popt[0][1], rel_err[1]) + '\n' + r'$\sigma_{OPD} = %.3f$ nm (%.3f%%)'%(popt[0][2], rel_err[2])
    txt2 = 'Expected Values:\n' + 'Na = %.5f'%(na) + '\n' + r'$\mu_{OPD} = %.3f$ nm'%(mu_opd) + '\n' + r'$\sigma_{OPD} = %.3f$ nm'%(sig_opd)
    plt.text(0.55,0.85, txt2, va='center', fontsize=30, transform = ax.transAxes, bbox=dict(boxstyle="square", facecolor='white'))
    plt.text(0.55,0.55, txt1, va='center', fontsize=30, transform = ax.transAxes, bbox=dict(boxstyle="square", facecolor='white'))
#    plt.tight_layout()


#data_IA_axis = cp.array(np.linspace(data_IA[0].min(), data_IA[0].max(), np.size(np.unique(data_IA[0]))), dtype=cp.float32)
#cdf_data_IA = gff.computeCdf(data_IA_axis, data_IA[0], 'cdf', False)
#
#rv_IA = gff.rv_generator(data_IA_axis, cdf_data_IA/data_IA[0].size, 60000)
#axe = cp.linspace(rv_IA.min(), rv_IA.max()+1, np.size(np.unique(cp.asnumpy(rv_IA))))
#cdf = gff.computeCdf(axe, rv_IA, 'cdf', True)
#cdf2 = gff.computeCdf(data_IA_axis, rv_IA, 'cdf', True)
#plt.figure()
#plt.subplot(211)
#plt.plot(cp.asnumpy(data_IA_axis), cp.asnumpy(cdf_data_IA/data_IA[0].size))
##plt.plot(cp.asnumpy(axe), cp.asnumpy(cdf), '-')
#plt.plot(cp.asnumpy(data_IA_axis), cp.asnumpy(cdf2), '-')
#plt.grid()
#plt.subplot(212)
#plt.plot(cp.asnumpy(data_IA_axis), cp.asnumpy(cdf_data_IA/data_IA[0].size)-cp.asnumpy(cdf))
#plt.plot(cp.asnumpy(data_IA_axis), cp.asnumpy(cdf_data_IA/data_IA[0].size)-cp.asnumpy(cdf2))
#plt.grid()
#
#x = cp.arange(7)
#y = cp.array([1,1,2,2.5,3,9])
#c = gff.computeCdf(x, y, 'cdf', False)
#print(x)
#print(y)
#print(c)