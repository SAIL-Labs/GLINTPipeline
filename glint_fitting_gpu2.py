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
from scipy.optimize import curve_fit, least_squares
from timeit import default_timer as time
import h5py
import os
import sys
import glint_fitting_functions as gff
from scipy.special import erf
from scipy.io import loadmat
from scipy.stats import norm
import pickle


def MCfunction(bins0, na, mu_opd, sig_opd):
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
    global fichier

#    sig_opd = 100.
#    mu_opd = 0

    sig_opd = abs(sig_opd)
    dphase_bias = 0
    
    nloop = 10
        
    visibility = (1 - na) / (1 + na)
#    bins = cp.asarray(bins, dtype=cp.float32)
    count += 1
    print(int(count), na, mu_opd, sig_opd)
    try:
        fichier.write('%s\t%s\t%s\t%s\t%s\n'%(int(count), na, mu_opd, sig_opd))
    except:
        pass
    
#    accum = cp.zeros((wl_scale.size, bins.size), dtype=cp.float32)
    if not mode_histo:
        accum = cp.zeros(bins0.shape, dtype=cp.float32)
    else:
        accum = cp.zeros((bins0.shape[0], bins0.shape[1]-1), dtype=cp.float32)
#    accum = [np.zeros(elt.shape) for elt in bins0]
    
    rv_opd = cp.random.normal(mu_opd, sig_opd, n_samp)
    rv_opd = rv_opd.astype(cp.float32)
#    rv_opd = rv_gen_doubleGauss(nsamp, mu1, mu2, sig1, A, target)
    
    if wl_scale.size > 1:
        spec_chan_width = np.mean(np.diff(wl_scale))
    else:
        spec_chan_width = 5.

    for _ in range(nloop):
        for k in range(wl_scale.size):
            bins = cp.asarray(bins0[k], dtype=cp.float32)
#            bin_width = cp.mean(cp.diff(bins), dtype=cp.float32)
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
                                           zeta_minus_A[k], zeta_minus_B[k], zeta_plus_A[k], zeta_plus_B[k], spec_chan_width, oversampling_switch)

            rv_null = rv_null[~np.isnan(rv_null)] # Remove NaNs
            rv_null = cp.sort(rv_null)
            
            if not mode_histo:
                if mode == 'cuda':
                    cdf_null = gff.computeCdf(bins, rv_null, 'ccdf', True)
                elif mode == 'cupy':
                    cdf_null = gff.computeCdfCupy(rv_null, bins)
                else:
                    cdf_null = gff.computeCdfCpu(cp.asnumpy(rv_null), cp.asnumpy(bins))
                    cdf_null = cp.asarray(cdf_null)
                accum[k] += cdf_null
            else:
                pdf_null = cp.histogram(rv_null, bins)[0]
                accum[k] += pdf_null / cp.sum(pdf_null)#*bin_width)
    
    accum = accum / nloop
        
    accum = cp.asnumpy(accum)
    return accum.ravel()
#    return [selt for elt in accum for selt in elt]

def map_error(params, x, y):
    residuals = y - MCfunction(x, *params[:-1])
    chi2 = np.sum(residuals**2) / (y.size-len(params))
    return chi2

class Logger(object):
    def __init__(self, log_path):
        self.orig_stdout = sys.stdout
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
    
    def close(self):
        sys.stdout = self.orig_stdout
        self.log.close()
        print('Stdout closed')

def basin_hoppin_values(mu_opd0, sig_opd0, na0, n_hop, bounds_mu, bounds_sig, bounds_na):
    mu_list = []
    sig_list = []
    null_list = []
    orig_seed = np.random.get_state()
    np.random.seed(1)
    print('Random drawing of init guesses')
    
    for k in range(n_hop[0], n_hop[1]):
        for _ in range(1000):
            mu_opd = np.random.normal(mu_opd0, 20)
            if mu_opd >= bounds_mu[0] and mu_opd <= bounds_mu[1]:
                break
            if _ == 1000-1:
                print('mu_opd: no new guess, take initial one')
                mu_opd = mu_opd0[idx_null]
        mu_list.append(mu_opd)
        
        for _ in range(1000):
            sig_opd = abs(np.random.normal(sig_opd0, 20))
            if sig_opd >= bounds_sig[0] and sig_opd <= bounds_sig[1]:
                break
            if _ == 1000-1:
                print('sig opd: no new guess, take initial one')
                sig_opd = sig_opd0[idx_null]
        sig_list.append(sig_opd)
            
        for _ in range(1000):
            na = np.random.normal(na0, 0.1)
            if na >= bounds_na[0] and na <= bounds_na[1]:
                break
            if _ == 1000-1:
                print('na: no new guess, take initial one')
                na = na0[idx_null]
        null_list.append(na)
            
    print('Random drawing done')
    np.random.set_state(orig_seed)
    return mu_list, sig_list, null_list
        
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
oversampling_switch = True
skip_fit = True
chi2_map_switch = False
mode_histo = True
nb_blocks = (None, None)
bin_bounds = (-0.01, 1) # Boundaries1
#bin_bounds = (0, 10.)
#bin_size = 15000
basin_hopping_nloop = (0, 1)
bounds_mu = (-100, 100)
bounds_sig = (1, 100)
bounds_na = (-0.001, 0.1)

''' Import real data '''
datafolder = '20191015_simulation/'
darkfolder = '20191015_simulation/'
#root = "C:/Users/marc-antoine/glint/"
root = "/mnt/96980F95980F72D3/glint/"
file_path = root+'reduction/'+datafolder
save_path = file_path+'output/'
data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and not 'dark' in f][nb_blocks[0]:nb_blocks[1]]
dark_list = [root+'reduction/'+darkfolder+f for f in os.listdir(root+'reduction/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_blocks[0]:nb_blocks[1]]
calib_params_path = root+'reduction/'+'calibration_params_simu/'
zeta_coeff_path = calib_params_path + 'zeta_coeff_simu.hdf5'
instrumental_offsets_path = calib_params_path + '4WG_opd0_and_phase_simu.txt'
segment_positions = loadmat(calib_params_path+'N1N4_opti.mat')['PTTPositionOn'][:,0]*1000

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
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

mu_opd0 = np.ones(6,)*(0)
sig_opd0 = np.ones(6,)*61 # In nm
na0 = np.ones(6,)*0.
dphase_bias = 0.

''' Generate basin hopping values '''
mu_list, sig_list, null_list = basin_hoppin_values(mu_opd0[0], sig_opd0[0], na0[0], basin_hopping_nloop, bounds_mu, bounds_sig, bounds_na)


results = {}
print('starting loop')
total_time_start = time()
for key in ['null1', 'null2', 'null3', 'null4', 'null5', 'null6'][:1]:
    plt.ioff()
   
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
    offset_opd = (segment_positions[segment_id_A] - segment_positions[segment_id_B]) + offset_opd
    
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
#    sz = max([np.size(np.unique(d)) for d in data_null])
#    null_axis = np.array([np.linspace(data_null[i].min(), data_null[i].max(), int(sz**0.5), retstep=False, dtype=np.float32) for i in range(data_null.shape[0])])
    sz = np.array([np.size(d[(d>=bin_bounds[0])&(d<=bin_bounds[1])]) for d in data_null])
    sz = np.max(sz)
#    sz = bin_size**2
    if not mode_histo:
        null_axis = np.array([np.linspace(bin_bounds[0], bin_bounds[1], int(sz**0.5), retstep=False, dtype=np.float32) for elt in data_null])
        null_axis = np.array([elt[:-1] + np.diff(elt) for elt in null_axis])
    else:
        null_axis = np.array([np.linspace(bin_bounds[0], bin_bounds[1], int(sz**0.5+1), retstep=False, dtype=np.float32) for i in range(data_null.shape[0])])
#        null_axis = np.array([np.linspace(data_null[i].min(), data_null[i].max(), int(sz**0.5+1), retstep=False, dtype=np.float32) for i in range(data_null.shape[0])])
    null_axis_width = np.mean(np.diff(null_axis, axis=-1))
    
    null_cdf = []
    null_cdf_err = []
    for wl in range(len(wl_scale)):
        if not mode_histo:
            if mode == 'cuda':
                cdf = gff.computeCdf(null_axis[wl], data_null[wl], 'ccdf', True)
                null_cdf.append(cp.asnumpy(cdf))
            elif mode == 'cupy':
                cdf = gff.computeCdfCupy(data_null[wl], cp.asarray(null_axis[wl], dtype=cp.float32))
                null_cdf.append(cp.asnumpy(cdf))
            else:
                cdf = gff.computeCdf(np.sort(data_null[wl]), null_axis[wl])
                null_cdf.append(cdf)
                    
            start = time()
#            cdf_err = gff.getErrorCDF(data_null[wl], data_null_err[wl], null_axis[wl]) # Barnaby
            cdf_err = gff.getErrorBinomNorm(cp.asnumpy(cdf), data_null[wl].size, 1.)
#            cdf_err = gff.getErrorWilson(cp.asnumpy(cdf), data_null[wl].size, norm.cdf(1) - norm.cdf(-1))
            stop = time()
            print('Time CDF error=', stop-start)
            null_cdf_err.append(cdf_err)

        else:
            pdf = np.histogram(data_null[wl], null_axis[wl], density=False)[0]
            pdf_size = np.sum(pdf)
            print('Histogram size=', np.sum(pdf), np.sum(pdf)/data_null[wl].size)
            bin_width = null_axis[wl][1]-null_axis[wl][0]
            pdf = pdf / np.sum(pdf)
            null_cdf.append(pdf)
            
            start = time()
#            pdf_err = gff.getErrorPDF(data_null[wl], data_null_err[wl], null_axis[wl]) # Barnaby
            pdf_err = gff.getErrorBinomNorm(pdf, pdf_size, 1.)
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
    pcov_liste = []
    termination_liste = []
    for basin_hopping_count in range(basin_hopping_nloop[0], basin_hopping_nloop[1]):
        sys.stdout = Logger(save_path+'basin_hop_%02d'%(basin_hopping_count)+'.log')
        print('-------------')
        print(basin_hopping_count)
        print('-------------')
        print('Fitting '+key)  
        # model fitting initial guess
        if basin_hopping_count == 0:
            mu_opd = mu_opd0[idx_null]
            sig_opd = sig_opd0[idx_null]
            na = na0[idx_null]
        else:
            mu_opd = mu_list[basin_hopping_count]
            sig_opd = sig_list[basin_hopping_count]
            na = null_list[basin_hopping_count]
                
        ''' Model fitting '''
        if not chi2_map_switch:
            if skip_fit:    
                count = 0.
                start = time()
                z = MCfunction(null_axis, np.array([na], dtype=np.float32)[0], mu_opd, sig_opd)
                stop = time()
                print('test', stop-start)
                out = z.reshape(null_cdf.shape)
                na_opt = na
                uncertainties = np.zeros(3)
                popt = (np.array([na, mu_opd, sig_opd]), np.ones((3,3)))
                chi2 = 1/(null_cdf.size-popt[0].size) * np.sum((null_cdf.ravel() - z)**2)
                term_status = None
            
            else:            
                print('Model fitting')    
                count = 0.
                guess_na = na
                initial_guess = [guess_na, mu_opd, sig_opd]
                initial_guess = np.array(initial_guess, dtype=np.float32)
                init_liste.append([guess_na]+list(initial_guess[1:]))
                
                start = time()
#                popt = curve_fit(MCfunction, null_axis, null_cdf.ravel(), p0=initial_guess, epsfcn = null_axis_width, sigma=null_cdf_err.ravel(), absolute_sigma=True)
#                res = 0
#                term_status = None              
                
                with open(save_path+'callfunc_%02d.txt'%(basin_hopping_count), 'w') as fichier:
                    popt = gff.curvefit(MCfunction, null_axis, null_cdf.ravel(), p0=initial_guess, sigma=null_cdf_err.ravel(), 
                                        bounds=([bounds_na[0], bounds_mu[0], bounds_sig[0]],[bounds_na[1], bounds_mu[1], bounds_sig[1]]), 
                                        diff_step = [0.01, 100., 100.])
                res = popt[2]
                popt = popt[:2]
                print('Termination:', res.message)
                term_status = res.status
                
#                popt = curve_fit(MCfunction, null_axis, null_cdf.ravel(), p0=initial_guess, epsfcn = null_axis_width, sigma=null_cdf_err.ravel(), absolute_sigma=True, 
#                                 full_output=True)
#                res = popt[2:]
#                popt = popt[:2]
#                term_status = res[2]
                stop = time()
                print('Termination', term_status)
                print('Duration:', stop - start)

                out = MCfunction(null_axis, *popt[0])
                uncertainties = np.diag(popt[1])**0.5
                chi2 = 1/(null_cdf.size-popt[0].size) * np.sum((null_cdf.ravel() - out)**2/null_cdf_err.ravel()**2)
                print('chi2', chi2)
                
                real_params = np.array([na, mu_opd, sig_opd])
                rel_err = (real_params - popt[0]) / real_params * 100
                print('rel diff', rel_err)
                na_opt = popt[0][0]
                print('******')
                print(popt[0])
                print(uncertainties*chi2**0.5)
                print(chi2)
                print('******')

                np.savez(save_path+os.path.basename(file_path[:-1])+'_%03d'%(basin_hopping_count),
                         chi2=chi2, popt=[na_opt]+[elt for elt in popt[0][1:]], uncertainties=uncertainties, init=[guess_na]+list(initial_guess[1:]),
                                         termination=np.array([term_status]))
            
            chi2_liste.append(chi2)
            popt_liste.append([na_opt]+[elt for elt in popt[0][1:]])
            uncertainties_liste.append(uncertainties)
            termination_liste.append(term_status)
            pcov_liste.append(popt[1])
            
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
#                    plt.semilogy(null_axis[wl], null_cdf[wl], '.', markersize=5, label='Data')
#                    plt.semilogy(null_axis[wl], out.reshape((wl_scale.size,-1))[wl], '+', markersize=5, lw=5, alpha=0.8, label='Fit')
                    plt.errorbar(null_axis[wl], null_cdf[wl], yerr=null_cdf_err[wl], fmt='.', markersize=5, label='Data')
                    plt.errorbar(null_axis[wl], out.reshape((wl_scale.size,-1))[wl], fmt='+', markersize=5, lw=5, alpha=0.8, label='Fit')
                else:
#                    plt.semilogy(null_axis[wl][:-1], null_cdf[wl], '.', markersize=5, label='Data')
#                    plt.semilogy(null_axis[wl][:-1], out.reshape((wl_scale.size,-1))[wl], '+', markersize=5, lw=5, alpha=0.8, label='Fit')
                    plt.errorbar(null_axis[wl][:-1], null_cdf[wl], yerr=null_cdf_err[wl], fmt='.', markersize=5, label='Data')
                    plt.errorbar(null_axis[wl][:-1], out.reshape((wl_scale.size,-1))[wl], fmt='+', markersize=5, lw=5, alpha=0.8, label='Fit')                    
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
                ax.text(0.3, 0.8, txt3, va='center', transform = ax.transAxes, bbox=dict(boxstyle="square", facecolor='white'))
            plt.tight_layout()
            string = key+'_'+'%03d'%(basin_hopping_count)+'_'+str(wl_min)+'-'+str(wl_max)+'_'+os.path.basename(datafolder[:-1])
#            if nonoise: string = string + '_nodarkinmodel'
#            if not oversampling_switch: string = string + '_nooversamplinginmodel'
#            if not zeta_switch: string = string + '_nozetainmodel'
            if not skip_fit: 
                if not mode_histo:
                    string = string + '_fit_cdf'
                else:
                    string = string + '_fit_pdf'
            plt.savefig(save_path+string+'.png')
            if basin_hopping_nloop[1]-basin_hopping_nloop[0]>5:
                plt.close('all')
##            print(string)
##            np.save('/home/mam/Documents/glint/model fitting - labdata/'+key+'_%08d'%n_samp+'_%03d'%(supercount)+'.npy', out)

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
#                    plt.plot(null_axis[wl], null_cdf_err[wl], fmt='.', markersize=5, label='Data')
#                    plt.plot(null_axis[wl], out.reshape((wl_scale.size,-1))[wl], fmt='+', markersize=5, lw=5, alpha=0.8, label='Fit')
#                else:
##                    plt.semilogy(null_axis[wl][:-1], null_cdf[wl], '.', markersize=5, label='Data')
##                    plt.semilogy(null_axis[wl][:-1], out.reshape((wl_scale.size,-1))[wl], '+', markersize=5, lw=5, alpha=0.8, label='Fit')
#                    plt.plot(null_axis[wl][:-1], null_cdf_err[wl], '.', markersize=5, label='Error on data')
##                    plt.errorbar(null_axis[wl][:-1], out.reshape((wl_scale.size,-1))[wl], fmt='+', markersize=5, lw=5, alpha=0.8, label='Fit')                    
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
            
        else:
            print('Error mapping')
            count = 0
            map_na, step_na = np.linspace(bounds_na[0], bounds_na[1], 10, endpoint=False, retstep=True)
            map_mu_opd, step_mu = np.linspace(bounds_mu[0], bounds_mu[1], 10, endpoint=False, retstep=True)
            map_sig_opd, step_sig = np.linspace(bounds_sig[0], bounds_sig[1], 10, endpoint=False, retstep=True)
        #    map_sig_opd = np.array([100])
            map_A = np.array([0.5])
            chi2map = []
            start = time()
            for visi in map_na:
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
            
            np.save(save_path+'chi2map_%03d'%(basin_hopping_count), chi2map)
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
        
        sys.stdout.close()
    
    results[key] = [popt_liste, uncertainties_liste, chi2_liste, init_liste, termination_liste, pcov_liste]
            
total_time_stop = time()            
print('Total time', total_time_stop-total_time_start)
#np.save('/home/mam/Documents/glint/model fitting - nsamp/'+'chi2_'+'%08d'%(n_samp)+'.npy', out)        
plt.ion()
plt.show()

if mode_histo:
    pickle_name = os.path.basename(file_path[:-1])+'_pdf.pkl'
else:
    pickle_name = os.path.basename(file_path[:-1])+'_cdf.pkl'
                                   
with open(save_path+pickle_name, 'wb') as f:
    pickle.dump(results, f)

print('-- end --')

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