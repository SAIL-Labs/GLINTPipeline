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
from scipy.io import loadmat
from scipy.stats import norm
from astropy.stats import bootstrap

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
        

def MCfunction(bins0, visibility, mu_opd, sig_opd):
    '''
    For now, this function deals with polychromatic for one baseline
    '''
    global data_IA_axis, cdf_data_IA, data_IB_axis, cdf_data_IB # On GPU
    global zeta_minus_A, zeta_minus_B, zeta_plus_A, zeta_plus_B
    global offset_opd, phase_bias
    global n_samp, count, wl_scale
    global dark_Iminus_cdf, dark_Iminus_axis, dark_Iplus_cdf, dark_Iplus_axis # On GPU
    global mode, mode_histo
    global rv_IA, rv_IB, rv_opd, rv_dark_Iminus, rv_dark_Iplus, rv_null # On GPU
    global oversampling_switch, nonoise

#    sig_opd = 100.
#    mu_opd = 0

    sig_opd = abs(sig_opd)
    dphase_bias = 0
    
    nloop = 10
        
#    bins = cp.asarray(bins, dtype=cp.float32)
    count += 1
    print(int(count), visibility, mu_opd, sig_opd, dphase_bias)     
#    accum = cp.zeros((wl_scale.size, bins.size), dtype=cp.float32)
    if not mode_histo:
        accum = cp.zeros(bins0.shape, dtype=cp.float32)
    else:
        accum = cp.zeros((bins0.shape[0], bins0.shape[1]-1), dtype=cp.float32)
#    accum = [np.zeros(elt.shape) for elt in bins0]
    
    rv_opd = cp.random.normal(mu_opd, sig_opd, n_samp)
    rv_opd = rv_opd.astype(cp.float32)
    
    step = np.mean(np.diff(wl_scale))

    for _ in range(nloop):
        for k in range(wl_scale.size):
            bins = cp.asarray(bins0[k], dtype=cp.float32)
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
            rv_IB = gff.rv_generator(data_IB_axis[k], cdf_data_IB[k], n_samp)
            
            rv_null = gff.computeNullDepth(rv_IA, rv_IB, wl_scale[k], offset_opd, rv_opd, phase_bias, dphase_bias, visibility, rv_dark_Iminus, rv_dark_Iplus, \
                                           zeta_minus_A[k], zeta_minus_B[k], zeta_plus_A[k], zeta_plus_B[k], step, oversampling_switch)

            rv_null = rv_null[~np.isnan(rv_null)] # Remove NaNs
            rv_null = cp.sort(rv_null)
            
            if not mode_histo:
                if mode == 'cuda':
#                    cdf_null = cp.zeros(bins.shape, dtype=cp.float32)
#                    computeCdfCuda(bins, rv_null, rv_null.size, cdf_null)
#                    cdf_null = 1-cdf_null/rv_null.size
                    cdf_null = gff.computeCdf(bins, rv_null, 'ccdf', True)
                elif mode == 'cupy':
                    cdf_null = computeCdfCupy(rv_null, bins)
                else:
                    cdf_null = computeCdf(cp.asnumpy(rv_null), cp.asnumpy(bins))
                    cdf_null = cp.asarray(cdf_null)
                accum[k] += cdf_null
            else:
                pdf_null = cp.histogram(rv_null, bins)[0]
                accum[k] += pdf_null / cp.sum(pdf_null)
    
    if not mode_histo:
        accum = accum / nloop
    else:
        accum = accum / cp.sum(accum, axis=-1, keepdims=True)
#    if not mode_histo:
#        accum = [elt / np.max(elt, axis=-1, keepdims=True) for elt in accum]
#    else:
#        accum = [elt / np.sum(elt, axis=-1, keepdims=True) for elt in accum]
        
    accum = cp.asnumpy(accum)
    return accum.ravel()
#    return [selt for elt in accum for selt in elt]

def map_error(params, x, y):
    residuals = y - MCfunction(x, *params[:-1])
    chi2 = np.sum(residuals**2) / (y.size-len(params))
    return chi2

def bootstrap_pdf(rv, axis, bootnum):
    bootstr = bootstrap(rv, bootnum)
    liste = []
    for k in range(bootnum):
        pdf = np.histogram(bootstr[k], axis)[0]
        pdf = pdf / np.sum(pdf)
        liste.append(pdf)
        
    liste = np.array(liste)
    std = liste.std(axis=0)
    std[std==0] = std[std!=0].min()
    return std

def bootstrap_cdf(rv, axis, bootnum):
    bootstr = bootstrap(rv, bootnum)
    liste = []
    for k in range(bootnum):
        cdf = gff.computeCdf(axis, bootstr[k], 'ccdf', True)
        liste.append(cp.asnumpy(cdf))
        
    liste = np.array(liste)
    std = liste.std(axis=0)
    std[std==0] = std[std!=0].min()
    return std
    
plt.ioff()

''' Settings '''  
wl_min = 1550
wl_max = 1600
n_samp = int(1e+7) # number of samples per loop
mode = 'cuda'
nonoise = False
phase_bias_switch = True
opd_bias_switch = True
zeta_switch = True
oversampling_switch = False
skip_fit = False
chi2_map_switch = False
mode_histo = False
nb_blocks = (None, None)

''' Import real data '''
datafolder = '20190718/20190718_turbulence2/'
darkfolder = '20190718/20190718_dark_turbulence/'
root = "/mnt/96980F95980F72D3/glint/"
file_path = root+'reduction/'+datafolder
data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and not 'dark' in f and 'n1n4' in f][nb_blocks[0]:nb_blocks[1]]
dark_list = [root+'reduction/'+darkfolder+f for f in os.listdir(root+'reduction/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_blocks[0]:nb_blocks[1]]
calib_params_path = '/mnt/96980F95980F72D3/glint/reduction/calibration_params/'
zeta_coeff_path = calib_params_path + 'zeta_coeff.hdf5'
instrumental_offsets_path = calib_params_path + '4WG_opd0_and_phase.txt'
segment_positions = loadmat(calib_params_path+'N1N4_opti2.mat')['PTTPositionOn'][:,0]*1000

instrumental_offsets = np.loadtxt(instrumental_offsets_path)
if not opd_bias_switch:
#    opd0 = np.array([ 400.5,  400.5, 1201.5,  400.5,  -801. ,  -801. ]) * (-1)
#    opd0 = np.array([801., 801., 2403., 801., 1602., 1602.])*(-1)
#    opd0 = np.ones(6) * (-1) * 400.5
#    instrumental_offsets[:,0] = opd0
    instrumental_offsets[:,0] = -instrumental_offsets[:,0]
    segment_positions[:] = 0.  
if not phase_bias_switch:
    instrumental_offsets[:,1] = 0.


# List in dictionary: indexes of null, beam A and beam B, zeta label for antinull, segment id
null_table = {'null1':[0,[0,1], 'null7', [28,34]], 'null2':[1,[1,2], 'null8', [34,25]], 'null3':[2,[0,3], 'null9', [28,23]], \
              'null4':[3,[2,3], 'null10', [25,23]], 'null5':[4,[2,0], 'null11',[25,28]], 'null6':[5,[3,1], 'null12', [23,34]]}

mu_opd0 = np.ones(6,)*200
sig_opd0 = np.ones(6,)*200 # In nm
na0 = np.ones(6,)*0
dphase_bias = 0.

results = {}
print('starting loop')
total_time_start = time()
for key in ['null1', 'null2', 'null3', 'null4', 'null5', 'null6'][:1]:
#    plt.ioff()
#    plt.close('all')
   
    if nonoise:
        data = gff.load_data(data_list, (wl_min, wl_max))
    else:   
        dark = gff.load_data(dark_list, (wl_min, wl_max), key)
        data = gff.load_data(data_list, (wl_min, wl_max), key, dark)
        
    wl_scale = data['wl_scale']
    zeta_coeff = gff.get_zeta_coeff(zeta_coeff_path, wl_scale, False)
    if not zeta_switch:
        for key in zeta_coeff.keys():
            if key != 'wl_scale':
                zeta_coeff[key][:] = 1.
    
    data_photo = data['photo'].copy()
    if not nonoise:
        ''' Remove dark contribution to measured intensity fluctuations '''
        # Estimate the mean and variance of the dark and data photometric fluctuations 
        mean_data, var_data = np.mean(data['photo'], axis=-1), np.var(data['photo'], axis=-1)
        mean_dark, var_dark = np.mean(dark['photo'], axis=-1), np.var(dark['photo'], axis=-1)
            
        # Substract variance of dark fluctuations to the variance of the photometric ones
        data_photo = (data_photo - mean_data[:,:,None]) * \
            ((var_data[:,:,None]-var_dark[:,:,None])/var_data[:,:,None])**0.5 + mean_data[:,:,None] - mean_dark[:,:,None]
            

    ''' Get histograms of intensities and dark current in the pair of photomoetry outputs '''
    idx_null = null_table[key][0] 
    idx_photo = null_table[key][1]
    key_antinull = null_table[key][2]
    segment_id_A, segment_id_B = null_table[key][3]
    data_IA, data_IB = data_photo[0], data_photo[1]
    
    zeta_minus_A, zeta_minus_B = zeta_coeff['b%s%s'%(idx_photo[0]+1, key)], zeta_coeff['b%s%s'%(idx_photo[1]+1, key)]
    zeta_plus_A, zeta_plus_B = zeta_coeff['b%s%s'%(idx_photo[0]+1, key_antinull)], zeta_coeff['b%s%s'%(idx_photo[1]+1, key_antinull)]
    offset_opd, phase_bias = instrumental_offsets[idx_null]
    offset_opd = (segment_positions[segment_id_A] - segment_positions[segment_id_B]) - offset_opd
    
    if nonoise:
        dark_Iminus_axis = cp.zeros(np.size(np.unique(data_IA[0])))
        dark_Iminus_cdf = cp.zeros(np.size(np.unique(data_IA[0])))
        dark_Iplus_axis = cp.zeros(np.size(np.unique(data_IA[0])))
        dark_Iplus_cdf = cp.zeros(np.size(np.unique(data_IA[0])))
    else:   
        dark_Iminus_axis = cp.array([np.linspace(dark['Iminus'][i].min(), dark['Iminus'][i].max(), \
                                                 np.size(np.unique(dark['Iminus'][i])), endpoint=False) for i in range(len(wl_scale))], \
                                                 dtype=cp.float32)
    
        dark_Iminus_cdf = cp.array([cp.asnumpy(gff.computeCdf(dark_Iminus_axis[i], dark['Iminus'][i], 'cdf', True)) \
                                    for i in range(len(wl_scale))], dtype=cp.float32)
    
        dark_Iplus_axis = cp.array([np.linspace(dark['Iplus'][i].min(), dark['Iplus'][i].max(), \
                                                np.size(np.unique(dark['Iplus'][i])), endpoint=False) for i in range(len(wl_scale))], \
                                                dtype=cp.float32)
    
        dark_Iplus_cdf = cp.array([cp.asnumpy(gff.computeCdf(dark_Iplus_axis[i], dark['Iplus'][i], 'cdf', True)) \
                                   for i in range(len(wl_scale))], dtype=cp.float32)


    data_IA_axis = cp.array([np.linspace(data_IA[i].min(), data_IA[i].max(), np.size(np.unique(data_IA[i]))) \
                             for i in range(len(wl_scale))], dtype=cp.float32)

    cdf_data_IA = cp.array([cp.asnumpy(gff.computeCdf(data_IA_axis[i], data_IA[i], 'cdf', True)) \
                            for i in range(len(wl_scale))], dtype=cp.float32)

    data_IB_axis = cp.array([np.linspace(data_IB[i].min(), data_IB[i].max(), np.size(np.unique(data_IB[i]))) \
                             for i in range(len(wl_scale))], dtype=cp.float32)

    cdf_data_IB = cp.array([cp.asnumpy(gff.computeCdf(data_IB_axis[i], data_IB[i], 'cdf', True)) \
                            for i in range(len(wl_scale))], dtype=cp.float32)
    
    ''' Make the survival function '''
    print('Compute survival function and error bars')
    data_null = data['null']
#    n_samp = int(data_null.shape[1])
#    data_null = data_null
    data_null_err = data['null_err']
    sz = max([np.size(np.unique(d)) for d in data_null])
    null_axis = np.array([np.linspace(data_null[i].min(), data_null[i].max(), int(sz**0.5), retstep=False, dtype=np.float32) for i in range(data_null.shape[0])])
#    sz = max([np.size(d[(d>=-0.5)&(d<=150)]) for d in data_null])
#    null_axis = np.array([np.linspace(-0.5, 150, int(sz**0.5), retstep=False, dtype=np.float32) for elt in data_null])
    null_axis_width = np.mean(np.diff(null_axis, axis=-1))
    
    null_cdf = []
    null_cdf_err = []
    for wl in range(len(wl_scale)):
        if not mode_histo:
            if mode == 'cuda':
                cdf = gff.computeCdf(null_axis[wl], data_null[wl], 'ccdf', True)
                null_cdf.append(cp.asnumpy(cdf))
            elif mode == 'cupy':
                cdf = computeCdfCupy(data_null[wl], cp.asarray(null_axis[wl], dtype=cp.float32))
                null_cdf.append(cp.asnumpy(cdf))
            else:
                cdf = computeCdf(np.sort(data_null[wl]), null_axis[wl])
                null_cdf.append(cdf)
                    
            start = time()
#            cdf_err = gff.getErrorCDF(data_null[wl], data_null_err[wl], null_axis[wl]) # Barnaby
            cdf_err = gff.getErrorBinomNorm(cp.asnumpy(cdf), data_null[wl].size)
#            cdf_err = gff.getErrorWilson(cp.asnumpy(cdf), data_null[wl].size, norm.cdf(1) - norm.cdf(-1))
            stop = time()
            print('Time CDF error=', stop-start)
            null_cdf_err.append(cdf_err)

        else:
            pdf = np.histogram(data_null[wl], null_axis[wl])[0]
            pdf = pdf / np.sum(pdf)
            null_cdf.append(pdf)
            
            start = time()
            pdf_err = gff.getErrorPDF(data_null[wl], data_null_err[wl], null_axis[wl]) # Barnaby
#            pdf_err = gff.getErrorBinomNorm(pdf, data_null[wl].size)
#            pdf_err = gff.getErrorWilson(pdf, data_null[wl].size, norm.cdf(1) - norm.cdf(-1))
            stop = time()
            print('Time PDF error=', stop-start)
            null_cdf_err.append(pdf_err)
        
#        liste = []
#        niter = 10
#        step = data_null.shape[1] // niter
#        for i in range(niter):
#            lb = i*step
#            ub = (i+1)*step
#            if data_null.shape[1] - ub <= data_null.shape[1] % step:
#                ub = None
#                
#            if not mode_histo:
#                cdf = gff.computeCdf(null_axis[wl], data_null[wl][lb:ub], 'ccdf', True)
#                liste.append(cp.asnumpy(cdf))
#            else:
#                pdf = np.histogram(data_null[wl][i*step:(i+1)*step], null_axis[wl])[0]
#                pdf = pdf / np.sum(pdf)
#                liste.append(pdf)
#        liste = np.array(liste)
#        liste2 = liste.std(axis=0)
#        liste2[liste2==0] = liste2[liste2!=0].min()
#        null_cdf_err.append(liste2)
                                    
    null_cdf = np.array(null_cdf)
    null_cdf_err = np.array(null_cdf_err)

#    f = plt.figure(figsize=(19.20,10.80))
#    count = 0
#    wl_idx = np.arange(wl_scale.size)
#    wl_idx = wl_idx[wl_scale>=1550]
#    for wl in wl_idx[::-1][:10]:
#        if len(wl_idx) > 1:
#            ax = f.add_subplot(5,2,count+1)
#        else:
#            ax = f.add_subplot(1,1,count+1)
#        plt.title('%s nm'%wl_scale[wl])
#        if not mode_histo:
#            plt.errorbar(null_axis[wl], null_cdf[wl], yerr=null_cdf_err[wl], fmt='.')
#        else:
#            plt.errorbar(null_axis[wl][:-1], null_cdf[wl], yerr=null_cdf_err[wl], fmt='.')
#        plt.grid()
#        plt.xlabel('Null depth')
#        plt.ylabel('Frequency')
#        plt.tight_layout()
#        count += 1

    chi2_liste = []
    popt_liste = []
    uncertainties_liste = []
    init_liste = []
    for basin_hopping_count in range(100):
        print('-------------')
        print('Fitting '+key)  
        # model fitting initial guess
        if basin_hopping_count == 0:
            mu_opd = mu_opd0[idx_null]
            sig_opd = sig_opd0[idx_null]
            na = na0[idx_null]
        else:
            mu_opd = abs(np.random.normal(mu_opd0[idx_null], 300))
            sig_opd = abs(np.random.normal(sig_opd0[idx_null], 100))
            na = abs(np.random.normal(na0[idx_null], 0.1))
        
        ''' Model fitting '''
        if not chi2_map_switch:
            if skip_fit:    
                count = 0.
                visibility = np.array([(1-na)/(1+na)], dtype=np.float32)[0]
                
                start = time()
                z = MCfunction(null_axis, visibility, mu_opd, sig_opd, dphase_bias)
                stop = time()
                print('test', stop-start)
                out = z.reshape(null_cdf.shape)
                na_opt = na
                uncertainties = np.zeros(3)
                popt = np.array([[(1-na)/(1+na), mu_opd, sig_opd]])
                chi2 = 1/(null_cdf.size-popt[0].size) * np.sum((null_cdf.ravel() - z)**2)
            
            else:            
                print('Model fitting')    
                count = 0.
                guess_na = na
                initial_guess = [(1-guess_na)/(1+guess_na), mu_opd, sig_opd]
                initial_guess = np.array(initial_guess, dtype=np.float32)
                init_liste.append([guess_na]+list(initial_guess[1:]))
                
                start = time()
                popt = curve_fit(MCfunction, null_axis, null_cdf.ravel(), p0=initial_guess, epsfcn = null_axis_width, sigma=null_cdf_err.ravel(), absolute_sigma=True)
                stop = time()
                print('Duration:', stop - start)
                out = MCfunction(null_axis, *popt[0])
                uncertainties = np.diag(popt[1])**0.5
                uncertainties[0] = 2/(1+popt[0][0])**2 * uncertainties[0]
                chi2 = 1/(null_cdf.size-popt[0].size) * np.sum((null_cdf.ravel() - out)**2/null_cdf_err.ravel()**2)
                print('chi2', chi2)
                
                real_params = np.array([(1-na)/(1+na), mu_opd, sig_opd])
                rel_err = (real_params - popt[0]) / real_params * 100
                print('rel diff', rel_err)
                na_opt = (1-popt[0][0])/(1+popt[0][0])
                print('******')
                print(popt[0])
                print('******')
            
            
            chi2_liste.append(chi2)
            popt_liste.append([na_opt]+[elt for elt in popt[0][1:]])
            uncertainties_liste.append(uncertainties)
                        
            
#            f = plt.figure(figsize=(19.20,10.80))
#            txt3 = '%s '%key+'Fitted values: ' + 'Na$ = %.2E \pm %.2E$, '%(na_opt, uncertainties[0]) + \
#            r'$\mu_{OPD} = %.2E \pm %.2E$ nm, '%(popt[0][1], uncertainties[1]) + \
#            r'$\sigma_{OPD} = %.2E \pm %.2E$ nm,'%(popt[0][2], uncertainties[2])+' Chi2 = %.2E '%(chi2)+'(Last = %.3f s)'%(stop-start)
#            count = 0
#            wl_idx = np.arange(wl_scale.size)
#            wl_idx = wl_idx[wl_scale>=1550]
#            for wl in wl_idx[::-1][:10]:
#                if len(wl_idx) > 1:
#                    ax = f.add_subplot(5,2,count+1)
#                else:
#                    ax = f.add_subplot(1,1,count+1)
#                plt.title('%s nm'%wl_scale[wl])
#                if not mode_histo:
##                    plt.semilogy(null_axis[wl], null_cdf[wl], '.', markersize=5, label='Data')
##                    plt.semilogy(null_axis[wl], out.reshape((wl_scale.size,-1))[wl], '+', markersize=5, lw=5, alpha=0.8, label='Fit')
#                    plt.errorbar(null_axis[wl], null_cdf[wl], yerr=null_cdf_err[wl], fmt='.', markersize=5, label='Data')
#                    plt.errorbar(null_axis[wl], out.reshape((wl_scale.size,-1))[wl], fmt='+', markersize=5, lw=5, alpha=0.8, label='Fit')
#                else:
##                    plt.semilogy(null_axis[wl][:-1], null_cdf[wl], '.', markersize=5, label='Data')
##                    plt.semilogy(null_axis[wl][:-1], out.reshape((wl_scale.size,-1))[wl], '+', markersize=5, lw=5, alpha=0.8, label='Fit')
#                    plt.errorbar(null_axis[wl][:-1], null_cdf[wl], yerr=null_cdf_err[wl], fmt='.', markersize=5, label='Data')
#                    plt.errorbar(null_axis[wl][:-1], out.reshape((wl_scale.size,-1))[wl], fmt='+', markersize=5, lw=5, alpha=0.8, label='Fit')                    
#                plt.grid()
#                plt.legend(loc='best')
#                plt.xlabel('Null depth')
#                plt.ylabel('Frequency')
##                plt.ylim(1e-8, 10)
##                plt.xlim(-0.86, 6)
#                count += 1
#            if len(wl_idx) > 1:
#                ax.text(-0.8, -0.5, txt3, va='center', transform = ax.transAxes, bbox=dict(boxstyle="square", facecolor='white'))
#            else:
#                ax.text(0.025, 0.05, txt3, va='center', transform = ax.transAxes, bbox=dict(boxstyle="square", facecolor='white'))
#            plt.tight_layout()
#            string = key+'_'+str(wl_min)+'-'+str(wl_max)+'_'+os.path.basename(os.path.dirname(datafolder))
#            if nonoise: string = string + '_nodarkinmodel'
#            if not oversampling_switch: string = string + '_nooversamplinginmodel'
#            if not zeta_switch: string = string + '_nozetainmodel'
#            if not skip_fit: 
#                if not mode_histo:
#                    string = 'fit_cdf_' + string
#                else:
#                    string = 'fit_pdf_' + string
##            string = str(np.diff(nb_blocks)[0])+'blocks_'+string
#    #        string = 'range_'+string
##            plt.savefig('/home/mam/Documents/glint/model fitting - labdata/'+string+'.png')
##            print(string)
##            np.save('/home/mam/Documents/glint/model fitting - labdata/'+key+'_%08d'%n_samp+'_%03d'%(supercount)+'.npy', out)
            
        else:
            print('Error mapping')
            count = 0
            map_na, step_na = np.linspace(-0.1,1.1,10, endpoint=False, retstep=True)
            map_visi = (1 - map_na) / (1 + map_na)
            map_mu_opd, step_mu = np.linspace(-100, 100, 10, endpoint=False, retstep=True)
            map_sig_opd, step_sig = np.linspace(1, 201, 10, endpoint=False, retstep=True)
        #    map_sig_opd = np.array([100])
            map_A = np.array([0.5])
            chi2map = []
            start = time()
            for visi in map_visi:
                temp1 = []
                for o in map_mu_opd:
                    temp2 = []
                    for s in map_sig_opd:
                        temp3 = []
                        for jump in map_A:
                            parameters = [visi, o, s, jump]
                            temp3.append(map_error(parameters, null_axis, null_cdf.ravel()))
                        temp2.append(temp3)
                    temp1.append(temp2)
                chi2map.append(temp1)
            stop = time()
            chi2map = np.array(chi2map)
            print('Duration: %.3f s'%(stop-start))
            
        #    chi2map = np.load('/mnt/96980F95980F72D3/glint/GLINTPipeline/chi2map_fit_1wl.npy')
            plt.figure(figsize=(19.20,10.80))
            for i in range(10):
                plt.subplot(5,2,i+1)
                plt.imshow(chi2map[:,:,i,0], interpolation='none', origin='lower', aspect='auto', extent=[map_mu_opd[0]-step_mu/2, map_mu_opd[-1]+step_mu/2, map_na[0]-step_na/2, map_na[-1]+step_na/2])
                plt.colorbar()
                plt.title('sig %.2f'%map_sig_opd[i])
                plt.xlabel('mu opd');plt.ylabel('null depth')
            plt.tight_layout()
            
            plt.figure(figsize=(19.20,10.80))
            for i in range(0,10):
                plt.subplot(5,2,i+1)
                plt.imshow(chi2map[i,:,:,0], interpolation='none', origin='lower', aspect='auto', extent=[map_sig_opd[0]-step_sig/2, map_sig_opd[-1]+step_sig/2, map_mu_opd[0]-step_mu/2, map_mu_opd[-1]+step_mu/2])
                plt.colorbar()
                plt.xlabel('sig opd');plt.ylabel('mu opd')
                plt.title('Na %.2f'%map_na[i])
            plt.tight_layout()
        
            plt.figure(figsize=(19.20,10.80))
            for i in range(10):
                plt.subplot(5,2,i+1)
                plt.imshow(chi2map[:,i,:,0], interpolation='none', origin='lower', aspect='auto', extent=[map_sig_opd[0]-step_sig/2, map_sig_opd[-1]+step_sig/2, map_na[0]-step_na/2, map_na[-1]]+step_na/2)
                plt.colorbar()
                plt.xlabel('sig opd');plt.ylabel('null depth')    
                plt.title('mu %.2f'%map_mu_opd[i])
            plt.tight_layout()
            
    results[key] = [popt_liste, uncertainties_liste, chi2_liste, init_liste]
            
total_time_stop = time()            
print('Total time', total_time_stop-total_time_start)
#np.save('/home/mam/Documents/glint/model fitting - nsamp/'+'chi2_'+'%08d'%(n_samp)+'.npy', out)        
plt.ion()
plt.show()

#plt.figure(figsize=(19.20,10.80))
#plt.plot(results['null1'][2], '.', markersize=20)
#plt.grid()
#plt.ylim(0, 4000)
#plt.xticks(size=35);plt.yticks(size=35)
#plt.xlabel('Trial', size=40)
#plt.ylabel(r'$\chi^2$', size=40)
#plt.tight_layout()
#
#plt.figure(figsize=(19.20,10.80))
#plt.plot(np.array(results['null1'][0])[:,0], results['null1'][2], '.', markersize=20)
#plt.grid()
#plt.ylim(0, 4000)
#plt.xlim(-0.01, 0.025)
#plt.xticks(size=35);plt.yticks(size=35)
#plt.xlabel('Null depth', size=40)
#plt.ylabel(r'$\chi^2$', size=40)
#plt.tight_layout()
#
#plt.figure(figsize=(19.20,10.80))
#plt.plot(np.array(results['null1'][0])[:,1], results['null1'][2], '.', markersize=20)
#plt.grid()
#plt.ylim(1000, 3000)
#plt.xlim(0, 450)
#plt.xticks(size=35);plt.yticks(size=35)
#plt.xlabel(r'$\mu_{OPD}$ (nm)', size=40)
#plt.ylabel(r'$\chi^2$', size=40)
#plt.tight_layout()
#
#plt.figure(figsize=(19.20,10.80))
#plt.plot(np.array(results['null1'][0])[:,2], results['null1'][2], '.', markersize=20)
#plt.grid()
#plt.ylim(1000,3000)
#plt.xlim(95,106)
#plt.xticks(size=35);plt.yticks(size=35)
#plt.xlabel(r'$\sigma_{OPD}$ (nm)', size=40)
#plt.ylabel(r'$\chi^2$', size=40)
#plt.tight_layout()
#
#plt.figure(figsize=(19.20,10.80))
#plt.plot(np.array(results['null1'][-1])[:,2], np.array(results['null1'][0])[:,2], '.', markersize=20)
#plt.grid()
##plt.ylim(-110,150)
##plt.xlim(95,106)
#plt.xticks(size=35);plt.yticks(size=35)
#plt.xlabel(r'Initial $\sigma_{OPD}$ (nm)', size=40)
#plt.ylabel(r'Fitted $\sigma_{OPD}$ (nm)', size=40)
#plt.tight_layout()
#
#plt.figure(figsize=(19.20,10.80))
#plt.plot(np.array(results['null1'][-1])[:,1], np.array(results['null1'][0])[:,1], '.', markersize=20)
#plt.grid()
#plt.ylim(-50,450)
##plt.xlim(95,106)
#plt.xticks(size=35);plt.yticks(size=35)
#plt.xlabel(r'Initial $\mu_{OPD}$ (nm)', size=40)
#plt.ylabel(r'Fitted $\mu_{OPD}$ (nm)', size=40)
#plt.tight_layout()
#
#temp = np.array(results['null1'][-1])[:,0]
#temp = (1-temp)/(1+temp)
#plt.figure(figsize=(19.20,10.80))
#plt.plot(temp, np.array(results['null1'][0])[:,0], '.', markersize=20)
#plt.grid()
#plt.ylim(-0.06,0.06)
#plt.xlim(-0.01,0.33)
#plt.xticks(size=35);plt.yticks(size=35)
#plt.xlabel(r'Initial null depth', size=40)
#plt.ylabel(r'Fitted null depth', size=40)
#plt.tight_layout()