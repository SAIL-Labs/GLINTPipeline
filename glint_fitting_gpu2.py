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
        

def MCfunction(bins0, visibility, mu_opd, sig_opd):#, A):
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
    A = 0
#    mu_opd = 0

    sig_opd = abs(sig_opd)
    A = abs(A)
    
    nloop = 10
        
#    bins = cp.asarray(bins, dtype=cp.float32)
    count += 1
    print(int(count), visibility, mu_opd, sig_opd, A)     
#    accum = cp.zeros((wl_scale.size, bins.size), dtype=cp.float32)
    if not mode_histo:
        accum = cp.zeros(bins0.shape, dtype=cp.float32)
    else:
        accum = cp.zeros((bins0.shape[0], bins0.shape[1]-1), dtype=cp.float32)
#    accum = [np.zeros(elt.shape) for elt in bins0]
    
    rv_opd = cp.random.normal(mu_opd, sig_opd, n_samp)
    rv_opd = rv_opd.astype(cp.float32)
#    rv_opd = gff.rv_gen_doubleGauss(n_samp, mu_opd, mu_opd+1602/2, sig_opd, A, 'gpu')
    
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
            
            rv_null = gff.computeNullDepth(rv_IA, rv_IB, wl_scale[k], offset_opd, rv_opd, phase_bias, visibility, rv_dark_Iminus, rv_dark_Iplus, \
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

plt.ioff()
chi2_list0 = []
#liste = []
for supercount in range(1,2):
#    plt.ioff()
#    plt.close('all')
    ''' Settings '''  
    wl_min = 1550
    wl_max = 1555
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
    
    # model fitting initial guess
    mu_opd0 = np.ones(6,)*100
    sig_opd0 = np.ones(6,)*100 # In nm
    na = 0.
    A = 0.
    
    # =============================================================================
    # Real data
    # =============================================================================
    ''' Import real data '''
    datafolder = '20190718/20190718_turbulence3/'
    darkfolder = '20190718/20190718_dark_turbulence/'
    root = "/mnt/96980F95980F72D3/glint/"
    file_path = root+'reduction/'+datafolder
    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and not 'dark' in f and 'n1n4' in f][nb_blocks[0]:nb_blocks[1]]
    dark_list = [root+'reduction/'+darkfolder+f for f in os.listdir(root+'reduction/'+darkfolder) if '.hdf5' in f and 'dark' in f]
    calib_params_path = '/mnt/96980F95980F72D3/glint/reduction/calibration_params/'
    zeta_coeff_path = calib_params_path + 'zeta_coeff.hdf5'
    instrumental_offsets_path = calib_params_path + '4WG_opd0_and_phase.txt'
    segment_positions = loadmat(calib_params_path+'N1N4_opti2.mat')['PTTPositionOn'][:,0]*1000
    
    if nonoise:
        data = gff.load_data(data_list, (wl_min, wl_max))
    else:    
        dark = gff.load_data(dark_list, (wl_min, wl_max))
        data = gff.load_data(data_list, (wl_min, wl_max), dark)
    
        
    wl_scale = data['wl_scale']
    label_photo = ['p1', 'p2', 'p3', 'p4']
    zeta_coeff = gff.get_zeta_coeff(zeta_coeff_path, wl_scale, False)
    if not zeta_switch:
        for key in zeta_coeff.keys():
            if key != 'wl_scale':
                zeta_coeff[key][:] = 1.
        
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
    
    data_photo = data['photo'].copy()
    if not nonoise:
        ''' Remove dark contribution to measured intensity fluctuations '''
        # Estimate the mean and variance of the dark and data photometric fluctuations 
        mean_data, var_data = np.mean(data['photo'], axis=-1), np.var(data['photo'], axis=-1)
        mean_dark, var_dark = np.mean(dark['photo'], axis=-1), np.var(dark['photo'], axis=-1)
            
        # Substract variance of dark fluctuations to the variance of the photometric ones
        data_photo = (data_photo - mean_data[:,:,None]) * \
            ((var_data[:,:,None]-var_dark[:,:,None])/var_data[:,:,None])**0.5 + mean_data[:,:,None] - mean_dark[:,:,None] 
    
    # List in dictionary: indexes of null, beam A and beam B, zeta label for antinull, segment id
    null_table = {'null1':[0,[0,1], 'null7', [28,34]], 'null2':[1,[1,2], 'null8', [34,25]], 'null3':[2,[0,3], 'null9', [28,23]], \
                  'null4':[3,[2,3], 'null10', [25,23]], 'null5':[4,[2,0], 'null11',[25,28]], 'null6':[5,[3,1], 'null12', [23,34]]}
    
    chi2_list = []
    ''' Model the 6 null depths '''
    for key in ['null1', 'null2', 'null3', 'null4', 'null5', 'null6'][:1]:
        print('-------------')
        print('Fitting '+key)
        ''' Get histograms of intensities and dark current in the pair of photomoetry outputs '''
        idx_null = null_table[key][0] 
        idx_photo = null_table[key][1]
        key_antinull = null_table[key][2]
        segment_id_A, segment_id_B = null_table[key][3]
        data_IA, data_IB = data_photo[idx_photo[0]], data_photo[idx_photo[1]]
        zeta_minus_A, zeta_minus_B = zeta_coeff['b%s%s'%(idx_photo[0]+1, key)], zeta_coeff['b%s%s'%(idx_photo[1]+1, key)]
        zeta_plus_A, zeta_plus_B = zeta_coeff['b%s%s'%(idx_photo[0]+1, key_antinull)], zeta_coeff['b%s%s'%(idx_photo[1]+1, key_antinull)]
        offset_opd, phase_bias = instrumental_offsets[idx_null]
        offset_opd = (segment_positions[segment_id_A] - segment_positions[segment_id_B]) - offset_opd
    #    print('IA', data_IA.max(), data_IA.min(), data_IA.mean(), data_IA.std())
    #    print('IB', data_IB.max(), data_IB.min(), data_IB.mean(), data_IB.std())
        
        mu_opd = mu_opd0[idx_null]
        sig_opd = sig_opd0[idx_null]
        
        if nonoise:
            dark_Iminus_axis = cp.zeros(np.size(np.unique(data_IA[0])))
            dark_Iminus_cdf = cp.zeros(np.size(np.unique(data_IA[0])))
            dark_Iplus_axis = cp.zeros(np.size(np.unique(data_IA[0])))
            dark_Iplus_cdf = cp.zeros(np.size(np.unique(data_IA[0])))
        else:
            dark_Iminus_axis = cp.array([np.linspace(dark['Iminus'][idx_null,i].min(), dark['Iminus'][idx_null,i].max(), \
                                                     np.size(np.unique(dark['Iminus'][idx_null,i])), endpoint=False) for i in range(len(wl_scale))], \
                                                     dtype=cp.float32)
        
            dark_Iminus_cdf = cp.array([cp.asnumpy(gff.computeCdf(dark_Iminus_axis[i], dark['Iminus'][idx_null,i], 'cdf', True)) \
                                        for i in range(len(wl_scale))], dtype=cp.float32)
        
            dark_Iplus_axis = cp.array([np.linspace(dark['Iplus'][idx_null,i].min(), dark['Iplus'][idx_null,i].max(), \
                                                    np.size(np.unique(dark['Iplus'][idx_null,i])), endpoint=False) for i in range(len(wl_scale))], \
                                                    dtype=cp.float32)
        
            dark_Iplus_cdf = cp.array([cp.asnumpy(gff.computeCdf(dark_Iplus_axis[i], dark['Iplus'][idx_null,i], 'cdf', True)) \
                                       for i in range(len(wl_scale))], dtype=cp.float32)
    
    
        data_IA_axis = cp.array([np.linspace(data_IA[i].min(), data_IA[i].max(), np.size(np.unique(data_IA[i]))) \
                                 for i in range(len(wl_scale))], dtype=cp.float32)
    
        cdf_data_IA = cp.array([cp.asnumpy(gff.computeCdf(data_IA_axis[i], data_IA[i], 'cdf', True)) \
                                for i in range(len(wl_scale))], dtype=cp.float32)
    
        data_IB_axis = cp.array([np.linspace(data_IB[i].min(), data_IB[i].max(), np.size(np.unique(data_IB[i]))) \
                                 for i in range(len(wl_scale))], dtype=cp.float32)
    
        cdf_data_IB = cp.array([cp.asnumpy(gff.computeCdf(data_IB_axis[i], data_IB[i], 'cdf', True)) \
                                for i in range(len(wl_scale))], dtype=cp.float32)
    
    #    plt.figure()
    #    plt.plot(cp.asnumpy(dark_Iminus_axis[0]), cp.asnumpy(dark_Iminus_cdf[0]))
    #    plt.plot(cp.asnumpy(dark_Iplus_axis[0]), cp.asnumpy(dark_Iplus_cdf[0]))
    #    plt.grid()
    #    plt.figure()
    #    plt.plot(cp.asnumpy(data_IA_axis[0]), cp.asnumpy(cdf_data_IA[0]))
    #    plt.plot(cp.asnumpy(data_IB_axis[0]), cp.asnumpy(cdf_data_IB[0]))
    #    plt.grid()
    
    
        ''' Make the survival function '''
        data_null = data['null'][idx_null]
    #    n_samp = int(data_null.shape[1])
    #    data_null = data_null
        data_null_err = data['null_err'][idx_null]
        sz = max([np.size(np.unique(d)) for d in data_null])
        null_axis = np.array([np.linspace(data_null[i].min(), data_null[i].max(), int(sz**0.5), retstep=False, dtype=np.float32) for i in range(data_null.shape[0])])
#        sz = max([np.size(d[(d>=-0.5)&(d<=150)]) for d in data_null])
#        null_axis = np.array([np.linspace(-0.5, 150, int(sz**0.5), retstep=False, dtype=np.float32) for elt in data_null])
        null_axis_width = np.mean(np.diff(null_axis, axis=-1))
        
        null_cdf = []
        null_cdf_err = []
        if not mode_histo:
            for wl in range(len(wl_scale)):
#                data_null_gpu = cp.array(data_null[wl], dtype=cp.float32)
#                data_null_gpu = cp.sort(data_null_gpu, axis=-1)
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
                cdf_err = gff.getErrorCDF(data_null[wl], data_null_err[wl], null_axis[wl])
                stop = time()
                print('Time CDF error=', stop-start)
                null_cdf_err.append(cdf_err)
            
        else:
            for wl in range(len(wl_scale)):
                pdf = np.histogram(data_null[wl], null_axis[wl])[0]
                pdf = pdf / np.sum(pdf)
                start = time()
                pdf_err = gff.getErrorPDF(data_null[wl], data_null_err[wl], null_axis[wl])
                stop = time()
                print('Time PDF error=', stop-start)
                null_cdf.append(pdf)
                null_cdf_err.append(pdf_err)
                
        null_cdf = np.array(null_cdf)
        null_cdf_err = np.array(null_cdf_err)
    
        f = plt.figure(figsize=(19.20,10.80))
        count = 0
        wl_idx = np.arange(wl_scale.size)
        wl_idx = wl_idx[wl_scale>=1550]
        for wl in wl_idx[::-1][:10]:
            if len(wl_idx) > 1:
                ax = f.add_subplot(5,2,count+1)
            else:
                ax = f.add_subplot(1,1,count+1)
            plt.title('%s nm'%wl_scale[wl])
            if not mode_histo:
                plt.errorbar(null_axis[wl], null_cdf[wl], yerr=null_cdf_err[wl], fmt='.')
            else:
                plt.errorbar(null_axis[wl][:-1], null_cdf[wl], yerr=null_cdf_err[wl], fmt='.')
            plt.grid()
            plt.xlabel('Null depth')
            plt.ylabel('Frequency')
            plt.tight_layout()
            count += 1
        continue
                
        ''' Model fitting '''
        if not chi2_map_switch:
            if skip_fit:    
                count = 0.
                visibility = np.array([(1-na)/(1+na)], dtype=np.float32)[0]
                
                start = time()
                z = MCfunction(null_axis, visibility, mu_opd, sig_opd)#, A)
                stop = time()
                print('test', stop-start)
                out = z.reshape(null_cdf.shape)
                na_opt = na
                uncertainties = np.zeros(3)
                popt = np.array([[(1-na)/(1+na), mu_opd, sig_opd]])
                chi2 = 1/(null_cdf.size-popt[0].size) * np.sum((null_cdf.ravel() - z)**2)
            
            else:
                print('Computing error of survival function')
                start = time()
#                null_cdf_err = cp.asnumpy(gff.getErrorCDF(cp.asarray(data_null), cp.asarray(data_null_err), cp.asarray(null_axis)))
#                null_cdf_err[null_cdf_err==0] = 1e-32
#                null_cdf_err2 = null_cdf_err.copy()
#                null_cdf_err2[null_cdf_err>=null_cdf] = null_cdf[null_cdf_err>=null_cdf]*0.9999
                null_cdf_err = np.ones(null_cdf.shape)
                stop = time()
                print('Duration: %.3f s'%(stop-start))
            
                print('Model fitting')    
                count = 0.
                guess_na = np.where(null_cdf[0].ravel() == np.max(null_cdf[0].ravel()))[0][-1]
#                guess_na = null_axis[guess_na%null_axis.size]
                guess_na = max(0, guess_na)
                guess_na = na
                initial_guess = [(1-guess_na)/(1+guess_na), mu_opd, sig_opd, A][:-1]
                initial_guess = np.array(initial_guess, dtype=np.float32)
                
                start = time()
                popt = curve_fit(MCfunction, null_axis, null_cdf.ravel(), p0=initial_guess, epsfcn = null_axis_width, sigma=null_cdf_err.ravel(), absolute_sigma=True)
                stop = time()
                print('Duration:', stop - start)
                out = MCfunction(null_axis, *popt[0])
                uncertainties = np.diag(popt[1])**0.5
                chi2 = 1/(null_cdf.size-popt[0].size) * np.sum((null_cdf.ravel() - out)**2/null_cdf_err.ravel()**2)
                print('chi2', chi2)
                
                real_params = np.array([(1-na)/(1+na), mu_opd, sig_opd])#, A])
                rel_err = (real_params - popt[0]) / real_params * 100
                print('rel diff', rel_err)
                na_opt = (1-popt[0][0])/(1+popt[0][0])
                print('******')
                print(popt[0])
                print('******')
            
            
            chi2_list.append(chi2)
            f = plt.figure(figsize=(19.20,10.80))
            txt3 = '%s '%key+'Fitted values: ' + 'Na$ = %.2E \pm %.2E$, '%(na_opt, uncertainties[0]) + \
            r'$\mu_{OPD} = %.2E \pm %.2E$ nm, '%(popt[0][1], uncertainties[1]) + \
            r'$\sigma_{OPD} = %.2E \pm %.2E$ nm,'%(popt[0][2], uncertainties[2])+' Chi2 = %.2E '%(chi2)+'(Last = %.3f s)'%(stop-start)
            count = 0
            wl_idx = np.arange(wl_scale.size)
            wl_idx = wl_idx[wl_scale>=1550]
            for wl in wl_idx[::-1][:10]:
                if len(wl_idx) > 1:
                    ax = f.add_subplot(5,2,count+1)
                else:
                    ax = f.add_subplot(1,1,count+1)
                plt.title('%s nm'%wl_scale[wl])
                if not mode_histo:
                    plt.plot(null_axis[wl], null_cdf[wl], '.', markersize=5, label='Data')
                    plt.plot(null_axis[wl], out.reshape((wl_scale.size,-1))[wl], '+', markersize=5, lw=5, alpha=0.8, label='Fit')
                else:
                    plt.plot(null_axis[wl][:-1], null_cdf[wl], '.', markersize=5, label='Data')
                    plt.plot(null_axis[wl][:-1], out.reshape((wl_scale.size,-1))[wl], '+', markersize=5, lw=5, alpha=0.8, label='Fit')                    
                plt.grid()
                plt.legend(loc='best')
                plt.xlabel('Null depth')
                plt.ylabel('Frequency')
#                plt.ylim(1e-8, 10)
#                plt.xlim(-0.86, 6)
                count += 1
            if len(wl_idx) > 1:
                ax.text(-0.8, -0.5, txt3, va='center', transform = ax.transAxes, bbox=dict(boxstyle="square", facecolor='white'))
            else:
                ax.text(0.025, 0.05, txt3, va='center', transform = ax.transAxes, bbox=dict(boxstyle="square", facecolor='white'))
#            plt.tight_layout()
            string = key+'_'+str(wl_min)+'-'+str(wl_max)+'_'+os.path.basename(os.path.dirname(datafolder))
            if nonoise: string = string + '_nodarkinmodel'
            if not oversampling_switch: string = string + '_nooversamplinginmodel'
            if not zeta_switch: string = string + '_nozetainmodel'
            if not skip_fit: 
                if not mode_histo:
                    string = 'fit_cdf_' + string
                else:
                    string = 'fit_pdf_' + string
#            string = str(np.diff(nb_blocks)[0])+'blocks_'+string
    #        string = 'range_'+string
            plt.savefig('/home/mam/Documents/glint/model fitting - labdata/'+string+'.png')
#            print(string)
#            np.save('/home/mam/Documents/glint/model fitting - labdata/'+key+'_%08d'%n_samp+'_%03d'%(supercount)+'.npy', out)
            
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
            
#        liste.append(out)
    
    chi2_list0.append(chi2_list)
    
chi2_list0 = np.array(chi2_list0)
#np.save('/home/mam/Documents/glint/model fitting - nsamp/'+'chi2_'+'%08d'%(n_samp)+'.npy', out)        
plt.ion()
plt.show()

#liste = np.array(liste)
#if mode_histo:
#    np.save('model_mode_histo_on_range_'+key, liste)
#else:
#    np.save('model_mode_histo_off_range_'+key, liste)
#
#
##null_axis = np.load('null_axis.npy')
#cdf = np.load('model_mode_histo_off_null5.npy')
#pdf = np.load('model_mode_histo_on_null5.npy')
#std_cdf = np.std(cdf, axis=0)
#rel_std_cdf = np.std(cdf, axis=0) / np.mean(cdf, axis=0)
#
#std_pdf = np.std(pdf, axis=0)
#rel_std_pdf = np.std(pdf, axis=0) / np.mean(pdf, axis=0)
#
#f = plt.figure(figsize=(19.20,10.80))
#count = 0
#for wl in wl_idx[::-1][:10]:
#    ax = f.add_subplot(5,2,count+1)
#    plt.title('%s nm'%wl_scale[wl])
#    plt.semilogy(null_axis[wl], std_cdf[wl], '.', markersize=10)
#    plt.semilogy(null_axis[wl][:-1], std_pdf[wl], '+', markersize=10)
#    plt.grid()
#    plt.xlabel('Measured null depth')
##    plt.ylim(-1e-6, 1e-5)
##    plt.xlim(null_axis[wl][0], null_axis[wl][-1])
#    plt.ylabel('Standard deviation')
#    count += 1
#plt.tight_layout()
#
#f = plt.figure(figsize=(19.20,10.80))
#count = 0
#for wl in wl_idx[::-1][:10]:
#    ax = f.add_subplot(5,2,count+1)
#    plt.title('%s nm'%wl_scale[wl])
#    plt.semilogy(null_axis[wl], rel_std_cdf[wl], '.', markersize=10)
#    plt.semilogy(null_axis[wl][:-1], rel_std_pdf[wl], '+', markersize=10)
#    plt.grid()
#    plt.xlabel('Measured null depth')
#    plt.ylim(-1, 1)
##    plt.xlim(null_axis[wl][0], null_axis[wl][-1])
#    if count ==4: plt.ylabel('Relative standard deviation')
#    count += 1
#plt.tight_layout()