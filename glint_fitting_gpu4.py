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
import glint_fitting_functions4 as gff
from scipy.special import erf
from scipy.io import loadmat
from scipy.stats import norm, skewnorm
import pickle
from datetime import datetime


def MCfunction(bins0, na, mu_opd, sig_opd):
    '''
    For now, this function deals with polychromatic for one baseline
    '''
    global data_IA_axis, cdf_data_IA, data_IB_axis, cdf_data_IB, interfminus_axis, cdf_interfminus # On GPU
    global zeta_minus_A, zeta_minus_B, zeta_plus_A, zeta_plus_B
    global dark_Iminus_cdf, dark_Iminus_axis, dark_Iplus_cdf, dark_Iplus_axis # On GPU
    global rv_IA, rv_IB, rv_opd, rv_dark_Iminus, rv_dark_Iplus, rv_null, rv_interfminus, rv_interfplus # On GPU
    global offset_opd, phase_bias
    global spec_chan_width
    global n_samp, count, wl_scale, nloop 
    global mode, mode_histo
    global oversampling_switch, nonoise, switch_invert_null
    global fichier
    global fullbw, rv_IA_list
    
#    coeffminusA, coeffminusB, coeffplusA, coeffplusB = 9.92102537e-01,  1.07047619e+00,  1.27814825e+00,  1.25404986e+00
#    coeffplusA, coeffplusB = 0.73912532, 1.00733721
#    coeffminusA, coeffminusB = 0.88141557, 1.01240566
#    coeffminusA, coeffminusB = 0.73912532, 1.00733721
#    coeffplusA, coeffplusB = 0.88141557, 1.01240566
#    
#    coeffminusA = [0.91060523, 0.88453292, 0.88463028, 0.85009597, 0.83135682, 0.83185338, 0.82099758, 0.81372649, 0.78178486, 0.76264897]
#    coeffminusB = [1.0266755 , 1.01679614, 1.0107579 , 1.00011394, 0.98552399, 0.97986161, 0.98637908, 0.9795024 , 0.97787673, 0.97589224]
#    coeffplusA = [0.91753561, 0.90109362, 0.90753026, 0.89043818, 0.88715266, 0.89198039, 0.89090354, 0.89308802, 0.88389494, 0.88347878]
#    coeffplusB = [1.02772322, 1.01761948, 1.01190975, 1.00012741, 0.98335892, 0.97544649, 0.98237683, 0.97218489, 0.96874216, 0.96497962]
    
    count += 1
    print(int(count), na, mu_opd, sig_opd)
    try:
        fichier.write('%s\t%s\t%s\t%s\n'%(int(count), na, mu_opd, sig_opd))
    except:
        pass
    
    if not fullbw:
        if not mode_histo:
            accum = cp.zeros(bins0.shape, dtype=cp.float32)
        else:
            accum = cp.zeros((bins0.shape[0], bins0.shape[1]-1), dtype=cp.float32)
    else:
        accum = cp.zeros((1, bins0.shape[1]-1), dtype=cp.float32)
    
    if wl_scale.size > 1:
        spec_chan_width = np.mean(np.diff(wl_scale))
    
        
    ''' Number of samples to simulate is high and the memory is low so we iterate to create an average histogram '''
    for _ in range(nloop):
        rv_opd = cp.random.normal(mu_opd, sig_opd, n_samp)
        rv_opd = rv_opd.astype(cp.float32)
#        rv_opd = skewnorm.rvs(skew, loc=mu_opd, scale=sig_opd, size=n_samp) # Skewd Gaussian distribution
#        rv_opd = cp.asarray(rv_opd, dtype=cp.float32) # Load the random values into the graphic card
        interfminus, interfplus = cp.zeros((wl_scale.size, n_samp), dtype=cp.float32), cp.zeros((wl_scale.size, n_samp), dtype=cp.float32)
        rv_IA_list = cp.zeros((wl_scale.size, n_samp))
        for k in range(wl_scale.size): # Iterate over the wavelength axis
            if fullbw:
                bins = cp.asarray(bins0[0], dtype=cp.float32)
            else:
                bins = cp.asarray(bins0[k], dtype=cp.float32)
#            bin_width = cp.mean(cp.diff(bins), dtype=cp.float32)
            # random values for dark noise
            if nonoise:
                rv_dark_Iminus = cp.zeros((n_samp,), dtype=cp.float32)
                rv_dark_Iplus = cp.zeros((n_samp,), dtype=cp.float32)
            else:
                rv_dark_Iminus = gff.rv_generator(dark_Iminus_axis[k], dark_Iminus_cdf[k], n_samp)
                rv_dark_Iminus = rv_dark_Iminus.astype(cp.float32)
                rv_dark_Iplus = gff.rv_generator(dark_Iplus_axis[k], dark_Iplus_cdf[k], n_samp)
                rv_dark_Iplus = rv_dark_Iplus.astype(cp.float32)
                rv_dark_Iminus -= rv_dark_Iminus.mean()
                rv_dark_Iplus -= rv_dark_Iplus.mean()
                
            ''' Generate random values from these pdf '''
            rv_IA = gff.rv_generator(data_IA_axis[k], cdf_data_IA[k], n_samp)   # random values for photometry A         
            rv_IB = gff.rv_generator(data_IB_axis[k], cdf_data_IB[k], n_samp) # random values for photometry B

            # Random values for synthetic null depths
#            rv_null, rv_interfminus, rv_interfplus = gff.computeNullDepthLinear(na, rv_IA, rv_IB, wl_scale[k], rv_opd, phase_bias, dphase_bias, rv_dark_Iminus, rv_dark_Iplus, 
#                     zeta_minus_A[k], zeta_minus_B[k], zeta_plus_A[k], zeta_plus_B[k],
#                     spec_chan_width, oversampling_switch, switch_invert_null)
            rv_null, rv_interfminus, rv_interfplus = gff.computeNullDepth(na, rv_IA, rv_IB, wl_scale[k], rv_opd, phase_bias, dphase_bias, rv_dark_Iminus, rv_dark_Iplus, 
                     zeta_minus_A[k], zeta_minus_B[k], zeta_plus_A[k], zeta_plus_B[k],
                     spec_chan_width, oversampling_switch, switch_invert_null)
                    
            rv_null = rv_null[~np.isnan(rv_null)] # Remove NaNs
            rv_null = cp.sort(rv_null)
            
            ''' Compute the average histogram over the nloops iterations '''
            if not fullbw:
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
                
            interfminus[k] = rv_interfminus
            interfplus[k] = rv_interfplus
            rv_IA_list[k] = rv_IA
            
        if fullbw:
            interfminus = interfminus.mean(axis=0)
            interfplus = interfplus.mean(axis=0)
            if switch_invert_null:
                rv_null = interfplus / interfminus
            else:
                rv_null = interfminus / interfplus
            rv_null = rv_null[~np.isnan(rv_null)] # Remove NaNs
            rv_null = cp.sort(rv_null)

            pdf_null = cp.histogram(rv_null, bins)[0]
            accum[0] += pdf_null / cp.sum(pdf_null)
            
    accum = accum / nloop
    if cp.all(cp.isnan(accum)):
        accum[:] = 0
    accum = cp.asnumpy(accum)

    return accum.ravel()

class Logger(object):
    ''' Class allowing to save the content of the console inside a txt file '''
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
    ''' Create as many as initial guess as there are basin hopping iterations to do'''
    mu_list = []
    sig_list = []
    null_list = []
#    orig_seed = np.random.get_state()
#    np.random.seed(1)
    print('Random drawing of init guesses')
    
    for k in range(n_hop[0], n_hop[1]):
        for _ in range(1000):
            mu_opd = np.random.normal(mu_opd0, 10)
            if mu_opd > bounds_mu[0] and mu_opd < bounds_mu[1]:
                break
            if _ == 1000-1:
                print('mu_opd: no new guess, take initial one')
                mu_opd = mu_opd0
        mu_list.append(mu_opd)
        
        for _ in range(1000):
            sig_opd = abs(np.random.normal(sig_opd0, 10))
            if sig_opd > bounds_sig[0] and sig_opd < bounds_sig[1]:
                break
            if _ == 1000-1:
                print('sig opd: no new guess, take initial one')
                sig_opd = sig_opd0
        sig_list.append(sig_opd)
            
        for _ in range(1000):
            na = np.random.normal(na0, 0.005)
            if na > bounds_na[0] and na < bounds_na[1]:
                break
            if _ == 1000-1:
                print('na: no new guess, take initial one')
                na = na0
        null_list.append(na)
            
    print('Random drawing done')
#    np.random.set_state(orig_seed)
    return mu_list, sig_list, null_list
        
step = 3000
nbfiles = 30000  # Omi Cet
#nbfiles = 36000 # Eps Peg
#nbfiles = 3300 # Turbulence 1
nb_files_data0 = np.arange(0, nbfiles, step)
#nb_files_data0 = np.tile(nb_files_data0 , 10) # Turbulence 1
basin_hopping_nloop0 = np.arange(nb_files_data0.size)
nb_files_data1 = nb_files_data0[:]
basin_hopping_nloop1 = basin_hopping_nloop0[:]


for a, b in zip(basin_hopping_nloop1, nb_files_data1):
    ''' Settings '''  
    wl_min = 1525 # lower bound of the bandwidth to process
    wl_max = 1575 # Uppber bound of the bandwidth to process
    wl_mid = (wl_max + wl_min)/2 # Centre wavelength of the bandwidth
    spec_chan_width = 50 # Width of the band, useful if you integrate over a bandwidth to compute the measure null. Set the width of the integrated spectral band then
    n_samp = int(1e+7) # number of samples per loop
    nloop = 10
    mode = 'cuda' # Mode for using the MC method, let as 'cuda'
    nonoise = False # If data is noise-free, set True
    phase_bias_switch = True # Implement a non-null achromatic phase in the null model
    opd_bias_switch = True # Implement an offset OPD in the null model
    zeta_switch = True # Use the measured zeta coeff. If False, value are set to 1
    oversampling_switch = True # Include the loss of coherence when OPD is too far from 0, according to a sinc envelop
    skip_fit = False # Do not fit, plot the histograms of data and model given the initial guesses
    chi2_map_switch = False # Map the parameters space over astronull, DeltaPhi mu and sigma
    mode_histo = True # SetTrue to compute the histogram, the CDF otherwise
    nb_files_data = (None, 20781) #(0, 20781) #(0, 20781) #(20782, None) # Which data files to load
    nb_files_dark = (0, None) # Which dark files to load
    basin_hopping_nloop = (a, a+1) # lower and upper bound of the iteration loop for basin hopping method
    fullbw = False
    activate_random_init_guesses = True
    
    if not skip_fit and not chi2_map_switch:
        plt.close('all')

##    # =============================================================================
#    # NullerData_SubaruJuly2019/20190718/20190718_turbulence1/ - 50 nm
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
#    nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
#    bounds_mu0  = [(0, 400),      (2200, 2500), (2200, 2500), (0, 600),      (3000, 3500), (0, 700)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(100, 300),    (200, 300),   (200, 300),   (50,250),      (50, 600),  (100, 400)] # bounds for DeltaPhi sig
#    bounds_na0  = [(-0.1, 0.1),   (-0.2, 0.2),  (-0.2, 0.2),  (-0.01, 0.01), (-0.025, 0.025), (-0.01, 0.01)] # bounds for astronull
##    bounds_sig0 = [(100, 300),    (200, 300),   (200, 300),   (50,300),      (50, 600),  (100, 400)] # 1271
##    bounds_na0  = [(-0.1, 0.1),   (-0.2, 0.2),  (-0.2, 0.2),  (-0.01, 0.02), (-0.025, 0.025), (-0.01, 0.01)] # 1271
##    bounds_mu0[4] = (-6600, -5900)
##    bounds_sig0[4] = (200, 500)
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(-0.02, 0.6), (-0.1, 0.4), (-0.1, 0.4), (-0.02, 0.6), (-0.02, 0.4), (-0.02, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
##    bin_bounds0 = [(-0.2, 0.6), (-0.1, 0.4), (-0.1, 0.4), (-0.2, 0.6), (-0.02, 0.4), (-0.02, 1.)] # 1271 ron
##    bin_bounds0 = [(-0.1, 0.6), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 0.6), (-0.02, 0.4), (-0.02, 1.)] # 459 ron
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([200, 2400, 2400, 300, 3200, 400]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([200, 260, 260, 150, 200, 250]) # initial guess of DeltaPhi sig
#    na0 = np.array([0, 0, 0, 0, 0, 0]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = 'NullerData_SubaruJuly2019/20190718/20190718_turbulence1_50nm_offset/'
#    darkfolder = 'NullerData_SubaruJuly2019/20190718/20190718_dark_turbulence_50nm_offset/'
##    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    root = "C:/Users/marc-antoine/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f][nb_files_data[0]:nb_files_data[1]]
##    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

#    # =============================================================================
#    # NullerData_SubaruJuly2019/20190718/20190718_turbulence2/ - 50 nm
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
#    nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
#    bounds_mu0  = [(0, 400),      (2200, 2500), (2200, 2500), (0, 600), (3000, 4000), (0, 700)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(50, 300),    (200, 300),   (200, 300),   (50,200),      (50, 450),  (50, 400)] # bounds for DeltaPhi sig
#    bounds_na0  = [(-0.1, 0.1),   (-0.2, 0.2),  (-0.2, 0.2),  (-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)] # bounds for astronull
##    bounds_mu0[4] = (-6600, -5900)
##    bounds_sig0[4] = (200, 500)
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(-0.02, 0.6), (-0.1, 0.4), (-0.1, 0.4), (-0.02, 0.6), (-0.02, 0.4), (-0.02, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([200, 2400, 2400, 300, 3700, 400]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([200, 260, 260, 150, 200, 250]) # initial guess of DeltaPhi sig
#    na0 = np.array([0, 0, 0, 0, 0, 0]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = 'NullerData_SubaruJuly2019/20190718/20190718_turbulence2_50nm_offset/'
#    darkfolder = 'NullerData_SubaruJuly2019/20190718/20190718_dark_turbulence_50nm_offset/'
##    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    root = "C:/Users/marc-antoine/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
##    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f][nb_files_data[0]:nb_files_data[1]]
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

#    # =============================================================================m
#    # NullerData_SubaruJuly2019/20190718/20190718_turbulence3/ - 50 nm
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
#    nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
#    bounds_mu0  = [(0, 400),      (2200, 2500), (2200, 2500), (0, 400), (3000, 3500), (0, 700)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(50, 150),    (200, 300),   (200, 300),   (50,150),      (50, 150),  (100, 200)] # bounds for DeltaPhi sig
#    bounds_na0  = [(-0.1, 0.1),   (-0.2, 0.2),  (-0.2, 0.2),  (-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)] # bounds for astronull
##    bounds_mu0[4] = (-6600, -5900)
##    bounds_sig0[4] = (200, 500)
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(-0.02, 0.3), (-0.1, 0.4), (-0.1, 0.4), (-0.02, 0.4), (-0.02, 0.2), (-0.02, 0.4)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([200, 2400, 2400, 300, 3300, 400]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([100, 260, 260, 100, 120, 140]) # initial guess of DeltaPhi sig
#    na0 = np.array([0, 0, 0, 0, 0, 0]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = 'NullerData_SubaruJuly2019/20190718/20190718_turbulence3_50nm_offset/'
#    darkfolder = 'NullerData_SubaruJuly2019/20190718/20190718_dark_turbulence_50nm_offset/'
##    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    root = "C:/Users/marc-antoine/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f][nb_files_data[0]:nb_files_data[1]]
##    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

#    # =============================================================================
#    # NullerData_SubaruJuly2019/20190718/20190718_turbulence4/ - 50 nm
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
#    nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
#    bounds_mu0  = [(0, 400),      (2200, 2500), (2200, 2500), (0, 600), (3000, 4000), (0, 600)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(50, 300),    (200, 300),   (200, 300),   (50,200),      (100, 400),  (50, 200)] # bounds for DeltaPhi sig
#    bounds_na0  = [(-0.1, 0.1),   (-0.2, 0.2),  (-0.2, 0.2),  (-0.01, 0.01), (-0.05, 0.05), (-0.01, 0.01)] # bounds for astronull
##    bounds_mu0[4] = (-6600, -5900)
##    bounds_sig0[4] = (200, 500)
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(-0.02, 0.4), (-0.1, 0.4), (-0.1, 0.4), (-0.02, 0.6), (-0.01, 0.2), (-0.02, 0.4)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([200, 2400, 2400, 300, 3700, 400]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([200, 260, 260, 150, 150, 100]) # initial guess of DeltaPhi sig
#    na0 = np.array([0, 0, 0, 0, 0, 0]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = 'NullerData_SubaruJuly2019/20190718/20190718_turbulence4_50nm_offset/'
#    darkfolder = 'NullerData_SubaruJuly2019/20190718/20190718_dark_turbulence_50nm_offset/'
##    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    root = "C:/Users/marc-antoine/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
##    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f][nb_files_data[0]:nb_files_data[1]]
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

#    # =============================================================================
#    # NullerData_SubaruJuly2019/20190718/20190718_turbulence5/ - 50 nm
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
#    nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
#    bounds_mu0  = [(0, 800),      (2200, 2500), (2200, 2500), (0, 600), (3000, 4000), (0, 600)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(100, 300),    (200, 300),   (200, 300),   (50,200),      (50, 250),  (50, 200)] # bounds for DeltaPhi sig
#    bounds_na0  = [(-0.01, 0.01),   (-0.2, 0.2),  (-0.2, 0.2),  (-0.01, 0.01), (-0.05, 0.05), (-0.01, 0.01)] # bounds for astronull
##    bounds_mu0[4] = (-6600, -5900)
##    bounds_sig0[4] = (200, 500)
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(-0.01, 0.1), (-0.1, 0.4), (-0.1, 0.4), (-0.01, 0.3), (-0.01, 0.2), (-0.01, 0.2)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([200, 2400, 2400, 300, 3600, 400]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([200, 260, 260, 150, 100, 100]) # initial guess of DeltaPhi sig
#    na0 = np.array([0, 0, 0, 0, 0, 0]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = 'NullerData_SubaruJuly2019/20190718/20190718_turbulence5_50nm_offset/'
#    darkfolder = 'NullerData_SubaruJuly2019/20190718/20190718_dark_turbulence_50nm_offset/'
##    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    root = "C:/Users/marc-antoine/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
##    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f][nb_files_data[0]:nb_files_data[1]]
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

    # =============================================================================
#    # NullerData_SubaruJuly2019/20190718/20190718_turbulence1/
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
#    nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
#    bounds_mu0  = [(0, 400),      (2200, 2500), (2200, 2500), (0, 600), (4500, 5000), (0, 700)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(100, 300),    (200, 300),   (200, 300),   (10,200),      (100, 500),  (100, 400)] # bounds for DeltaPhi sig
##    bounds_sig0 = [(100, 400),    (200, 300),   (200, 300),   (50,300),      (100, 500),  (100, 400)] # ron 1271
#    bounds_na0  = [(-0.01, 0.01),   (-0.2, 0.2),  (-0.2, 0.2),  (-0.01, 0.01), (-0.05, 0.05), (-0.01, 0.01)] # bounds for astronull
##    bounds_mu0[4] = (-6600, -5900)
##    bounds_sig0[4] = (200, 500)
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(-0.02, 0.2), (-0.1, 0.4), (-0.1, 0.4), (-0.02, 0.3), (-0.02, 0.1), (-0.02, 0.2)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
##    bin_bounds0 = [(-0.25, 1), (-0.1, 0.4), (-0.1, 0.4), (-0.25, 1), (-0.02, 0.1), (-0.02, 0.2)] # 459
##    bin_bounds0 = [(-0.6, 1.2), (-0.1, 0.4), (-0.1, 0.4), (-0.6, 1.2), (-0.02, 0.1), (-0.02, 0.2)] # 1271
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([200, 2400, 2400, 400, 4700, 400]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([200, 260, 260, 160, 300, 250]) # initial guess of DeltaPhi sig
#    na0 = np.array([0., 0, 0, 0, 0, 0]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = 'NullerData_SubaruJuly2019/20190718/20190718_turbulence1_459ron/'
#    darkfolder = 'NullerData_SubaruJuly2019/20190718/20190718_dark_turbulence_459ron/'
##    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    root = "C:/Users/marc-antoine/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f][nb_files_data[0]:nb_files_data[1]]
##    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

#    # =============================================================================
#    # NullerData_SubaruJuly2019/20190718/20190718_turbulence2/
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
#    nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
#    bounds_mu0  = [(0, 400),      (2200, 2500), (2200, 2500), (0, 600), (4500, 5500), (0, 700)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(100, 300),    (200, 300),   (200, 300),   (10,200),      (100, 500),  (100, 400)] # bounds for DeltaPhi sig
#    bounds_na0  = [(-0.01, 0.01),   (-0.2, 0.2),  (-0.2, 0.2),  (-0.01, 0.01), (-0.05, 0.05), (-0.01, 0.01)] # bounds for astronull
##    bounds_mu0[4] = (-6600, -5900)
##    bounds_sig0[4] = (200, 500)
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(-0.02, 0.2), (-0.1, 0.4), (-0.1, 0.4), (-0.02, 0.3), (-0.02, 0.1), (-0.02, 0.2)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([200, 2400, 2400, 400, 4700, 400]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([200, 260, 260, 160, 300, 250]) # initial guess of DeltaPhi sig
#    na0 = np.array([0, 0, 0, 0, 0, 0]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = 'NullerData_SubaruJuly2019/20190718/20190718_turbulence2/'
#    darkfolder = 'NullerData_SubaruJuly2019/20190718/20190718_dark_turbulence/'
##    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    root = "C:/Users/marc-antoine/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f][nb_files_data[0]:nb_files_data[1]]
##    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

#    # =============================================================================
#    # NullerData_SubaruJuly2019/20190718/20190718_turbulence3/
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
#    nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
#    bounds_mu0  = [(0, 400),      (2200, 2500), (2200, 2500), (0, 600), (4500, 5500), (0, 700)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(100, 300),    (200, 300),   (200, 300),   (10,200),      (100, 500),  (100, 400)] # bounds for DeltaPhi sig
#    bounds_na0  = [(-0.01, 0.01),   (-0.2, 0.2),  (-0.2, 0.2),  (-0.01, 0.01), (-0.05, 0.05), (-0.01, 0.01)] # bounds for astronull
##    bounds_mu0[4] = (-6600, -5900)
##    bounds_sig0[4] = (200, 500)
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(-0.02, 0.2), (-0.1, 0.4), (-0.1, 0.4), (-0.02, 0.3), (-0.02, 0.1), (-0.02, 0.2)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([200, 2400, 2400, 400, 4700, 400]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([200, 260, 260, 160, 300, 250]) # initial guess of DeltaPhi sig
#    na0 = np.array([0, 0, 0, 0, 0, 0]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = 'NullerData_SubaruJuly2019/20190718/20190718_turbulence3/'
#    darkfolder = 'NullerData_SubaruJuly2019/20190718/20190718_dark_turbulence/'
##    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    root = "C:/Users/marc-antoine/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
##    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f][nb_files_data[0]:nb_files_data[1]]
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

#    # =============================================================================
#    # NullerData_SubaruJuly2019/20190718/20190718_turbulence4/
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
#    nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
#    bounds_mu0  = [(0, 400),      (2200, 2500), (2200, 2500), (0, 600), (4500, 5500), (0, 700)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(100, 300),    (200, 300),   (200, 300),   (10,200),      (100, 500),  (100, 400)] # bounds for DeltaPhi sig
#    bounds_na0  = [(-0.01, 0.01),   (-0.2, 0.2),  (-0.2, 0.2),  (-0.01, 0.01), (-0.05, 0.05), (-0.01, 0.01)] # bounds for astronull
##    bounds_mu0[4] = (-6600, -5900)
##    bounds_sig0[4] = (200, 500)
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(-0.02, 0.2), (-0.1, 0.4), (-0.1, 0.4), (-0.02, 0.3), (-0.02, 0.1), (-0.02, 0.2)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([200, 2400, 2400, 400, 4700, 400]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([200, 260, 260, 160, 300, 250]) # initial guess of DeltaPhi sig
#    na0 = np.array([0, 0, 0, 0, 0, 0]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = 'NullerData_SubaruJuly2019/20190718/20190718_turbulence4/'
#    darkfolder = 'NullerData_SubaruJuly2019/20190718/20190718_dark_turbulence/'
##    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    root = "C:/Users/marc-antoine/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f][nb_files_data[0]:nb_files_data[1]]
##    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

#    # =============================================================================
#    # NullerData_SubaruJuly2019/20190718/20190718_turbulence5/
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
#    nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
#    bounds_mu0  = [(0, 400),      (2200, 2500), (2200, 2500), (0, 600), (5000, 5500), (0, 700)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(100, 300),    (200, 300),   (200, 300),   (10,200),      (50, 150),  (100, 400)] # bounds for DeltaPhi sig
#    bounds_na0  = [(-0.01, 0.01),   (-0.2, 0.2),  (-0.2, 0.2),  (-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)] # bounds for astronull
##    bounds_mu0[4] = (-6600, -5900)
##    bounds_sig0[4] = (200, 500)
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(-0.02, 0.2), (-0.1, 0.4), (-0.1, 0.4), (-0.02, 0.3), (-0.02, 0.1), (-0.02, 0.2)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([200, 2400, 2400, 400, 5100, 400]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([200, 260, 260, 160, 100, 250]) # initial guess of DeltaPhi sig
#    na0 = np.array([0, 0, 0, 0, 0, 0]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = 'NullerData_SubaruJuly2019/20190718/20190718_turbulence5/'
#    darkfolder = 'NullerData_SubaruJuly2019/20190718/20190718_dark_turbulence/'
##    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    root = "C:/Users/marc-antoine/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
##    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f][nb_files_data[0]:nb_files_data[1]]
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

#    # =============================================================================
#    # NullerData_SubaruJuly2019/20190718/20190718_turbulence2/
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
#    nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
#    bounds_mu0  = [(-wl_mid, wl_mid),      (2200, 2500), (2200, 2500), (0, 600), (4500, 5000), (0, 700)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(50, 300),    (200, 300),   (200, 300),   (50,200),      (50, 500),  (50, 400)] # bounds for DeltaPhi sig
#    bounds_na0  = [(-0.1, 0.1),   (-0.2, 0.2),  (-0.2, 0.2),  (-0.01, 0.01), (-0.05, 0.05), (-0.01, 0.01)] # bounds for astronull
##    bounds_mu0[4] = (-6600, -5900)
##    bounds_sig0[4] = (200, 500)
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(-0.02, 0.6), (-0.1, 0.4), (-0.1, 0.4), (-0.02, 0.6), (-0.02, 0.4), (-0.02, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([200, 2400, 2400, 300, 4700, 400]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([200, 260, 260, 150, 300, 250]) # initial guess of DeltaPhi sig
#    na0 = np.array([0, 0, 0, 0, 0, 0]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = 'NullerData_SubaruJuly2019/20190718/20190718_turbulence2/'
#    darkfolder = 'NullerData_SubaruJuly2019/20190718/20190718_dark_turbulence/'
##    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    root = "C:/Users/marc-antoine/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f][nb_files_data[0]:nb_files_data[1]]
##    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

    ## =============================================================================
    ## 20191128/turbulence/  - Null1 is inverted, null4 is ok
    ## =============================================================================
    #''' Set the bounds of the parameters to fit '''
    #bounds_mu0 = [(0, 600), (2200, 2500), (2200, 2500), (1554/4-300, 1554/4+300), (2200, 2500), (2200, 2500)] # bounds for DeltaPhi mu, one tuple per null
    #bounds_sig0 = [(50, 500), (200, 300), (200, 300), (50,150), (50, 247), (200, 300)] # bounds for DeltaPhi sig
    #bounds_na0 = [(-0.2, 0.2), (0., 0.05), (0., 0.01), (-0.05, 0.05), (0., 0.05), (0., 0.05)] # bounds for astronull
    #diffstep = [0.02, 50, 50] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    #xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    #
    #''' Set the initial conditions '''
    #mu_opd0 = np.array([100, 2400, 2400, 1554/4, 2300, 2300]) # initial guess of DeltaPhi mu
    #sig_opd0 = np.array([180, 260, 260, 100, 200, 201]) # initial guess of DeltaPhi sig
    #na0 = np.array([0., 0.001, 0.001, 0.04, 0.001, 0.001]) # initial guess of astro null
    #
    #datafolder = '20191128/turbulence/'
    #darkfolder = '20191128/dark_turbulence/'
    #root = "C:/Users/marc-antoine/glint/"
    ##root = "/mnt/96980F95980F72D3/glint/"
    #file_path = root+'GLINTprocessed/'+datafolder
    #save_path = file_path+'output/'
    #data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f][nb_files_data[0]:nb_files_data[1]]
    #dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]
    
#    # =============================================================================
#    # 20191128/turbulence/  - 50nm - Null1 is inverted, null4 is ok
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    bounds_mu0 = [(-400, 1000), (2200, 2500), (2200, 2500), (0, 1000), (2200, 2500), (2200, 2500)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(50, 500), (200, 300), (200, 300), (50,150), (50, 500), (200, 300)] # bounds for DeltaPhi sig
#    bounds_na0 = [(-0.2, 0.2), (0., 0.05), (0., 0.01), (-0.2, 0.2), (0., 0.05), (0., 0.05)] # bounds for astronull
#    diffstep = [0.02, 50, 50] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([100, 2400, 2400, 1554/4, 2300, 2300]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([180, 260, 260, 100, 200, 201]) # initial guess of DeltaPhi sig
#    na0 = np.array([0.01, 0.001, 0.001, 0.04, 0.001, 0.001]) # initial guess of astro null
#    
#    datafolder = '20191128/turbulence1554/'
#    darkfolder = '20191128/dark_turbulence1554/'
#    root = "C:/Users/marc-antoine/glint/"
#    #root = "/mnt/96980F95980F72D3/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]
    
#    # =============================================================================
#    # omi cet - Null1 is inverted, null4 is ok
#    # =============================================================================
#    nulls_to_invert = ['null1'] # If one null and antinull outputs are swapped in the data processing
#    nulls_to_invert_model = ['null1'] # If one null and antinull outputs are swapped in the data processing
#    ''' Set the bounds of the parameters to fit '''
#    bounds_mu0 = [(400, 1000), (2200, 2500), (2200, 2500), (0, 400), (-9100, -8000), (4000, 5300)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(100, 500), (200, 300), (200, 300), (40,140), (100, 500), (100, 400)] # bounds for DeltaPhi sig
#    bounds_na0 = [(0.1, 0.4), (0., 0.05), (0., 0.01), (0., 0.06), (0., 0.1), (0.1, 0.3)] # bounds for astronull
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(-0.5, 1.5), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 1.), (-0.2, 0.5), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([600, 2400, 2400, 200, -8500, 5000]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([250, 260, 260, 50, 300, 250]) # initial guess of DeltaPhi sig
#    na0 = np.array([0.3, 0.001, 0.001, 0.04, 0.06, 0.16]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = '20190907/omi_cet/'
#    darkfolder = '20190907/dark/'
#    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    #root = "/mnt/96980F95980F72D3/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'omi_cet' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]
    
    # =============================================================================
    # omi cet - 50 nm - Null1 is inverted, null4 is ok
    # =============================================================================
    nulls_to_invert = ['null1'] # If one null and antinull outputs are swapped in the data processing
    nulls_to_invert_model = ['null1'] # If one null and antinull outputs are swapped in the data processing
    ''' Set the bounds of the parameters to fit '''
    bounds_mu0 = [(50, 500), (2200, 2500), (2200, 2500), (0, wl_mid/4), (-400, 300), (4280, 5200)] # bounds for DeltaPhi mu, one tuple per null
    bounds_sig0 = [(100, 500), (200, 300), (200, 300), (50,150), (100, 500), (100, 400)] # bounds for DeltaPhi sig
    bounds_na0 = [(0., 0.35), (0., 0.05), (0., 0.01), (0.0, 0.2), (0.0, 0.1), (0.1, 0.3)] # bounds for astronull
#    bounds_mu0[4] = (-6600, -5900)
#    bounds_sig0[4] = (200, 500)
    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    bin_bounds0 = [(-0.5, 1.5), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 0.4), (0, 2.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
    ''' Set the initial conditions '''
    mu_opd0 = np.array([300, 2400, 2400, 300, -150, 4500]) # initial guess of DeltaPhi mu
    sig_opd0 = np.array([150, 260, 260, 100, 300, 250]) # initial guess of DeltaPhi sig
    na0 = np.array([0.3, 0.001, 0.001, 0.1, 0.04, 0.16]) # initial guess of astro null
    
    ''' Import real data '''
    datafolder = '20190907/omi_cet_50nm_offset/'
    darkfolder = '20190907/dark_50nm_offset/'
    root = "//silo.physics.usyd.edu.au/silo4/snert/"
    #root = "/mnt/96980F95980F72D3/glint/"
    file_path = root+'GLINTprocessed/'+datafolder
    save_path = file_path+'output/'
    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'omi_cet' in f][nb_files_data[0]:nb_files_data[1]]
    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

#    # =============================================================================
#    # omi cet stacked 100 frames - Null1 is inverted, null4 is ok
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    bounds_mu0 = [(300, 1000), (2200, 2500), (2200, 2500), (100, 600), (2200, 2500), (2200, 2500)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(100, 500), (200, 300), (200, 300), (1,120), (50, 200), (200, 300)] # bounds for DeltaPhi sig
#    bounds_na0 = [(0.1, 0.4), (0., 0.05), (0., 0.01), (0., 0.1), (-0.01, 0.1), (0., 0.05)] # bounds for astronull
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(0., 1), (-0.1, 0.4), (-0.1, 0.4), (0, 0.5), (-0.1, 0.4), (-0.1, 0.4)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([600, 2400, 2400, 200, 2300, 2300]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([250, 260, 260, 8, 100, 201]) # initial guess of DeltaPhi sig
#    na0 = np.array([0.3, 0.001, 0.001, 0.028, 0.001, 0.001]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = '20190907/omi_cet_stackedframe100/'
#    darkfolder = '20190907/dark_stackedframe100/'
#    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    #root = "/mnt/96980F95980F72D3/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'omi_cet' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

#    # =============================================================================
#    # omi cet stacked 10 frames - Null1 is inverted, null4 is ok
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    bounds_mu0 = [(300, 1000), (2200, 2500), (2200, 2500), (0, 400), (2200, 2500), (2200, 2500)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(100, 500), (200, 300), (200, 300), (20,140), (50, 200), (200, 300)] # bounds for DeltaPhi sig
#    bounds_na0 = [(0.1, 0.4), (0., 0.05), (0., 0.01), (0., 0.1), (-0.01, 0.1), (0., 0.05)] # bounds for astronull
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(0., 1), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 0.4)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([600, 2400, 2400, 200, 2300, 2300]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([250, 260, 260, 50, 100, 201]) # initial guess of DeltaPhi sig
#    na0 = np.array([0.3, 0.001, 0.001, 0.028, 0.001, 0.001]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = '20190907/omi_cet_stackedframe10/'
#    darkfolder = '20190907/dark_stackedframe10/'
#    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    #root = "/mnt/96980F95980F72D3/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'omi_cet' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

#    # =============================================================================
#    #eps peg - 50 nm - Null1 is inverted, null4 is ok
#    # =============================================================================
#    nulls_to_invert = ['null1'] # If one null and antinull outputs are swapped in the data processing
#    nulls_to_invert_model = ['null1'] # If one null and antinull outputs are swapped in the data processing
#    ''' Set the bounds of the parameters to fit '''
#    bounds_mu0  = [(50, 500),   (2200, 2500), (2200, 2500), (0, wl_mid/4), (-600, 400), (5900, 7000)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(100, 500),  (200, 300),   (200, 300),   (20,100),      (100, 500),  (100, 400)] # bounds for DeltaPhi sig
#    bounds_na0  = [(-0.2, 0.2), (-0.2, 0.2),   (-0.2, 0.2),   (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)] # bounds for astronull
##    bounds_mu0[4] = (-6600, -5900)
##    bounds_sig0[4] = (200, 500)
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(-0.5, 1.5), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 1), (-0.5, 1.), (-0.2, 2.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([300, 2400, 2400, 300, -150, 6000]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([150, 260, 260, 80, 300, 250]) # initial guess of DeltaPhi sig
#    na0 = np.array([-0.001, -0.001, -0.001, -0.001, -0.001, -0.001]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = '20190907/eps_peg_50nm_offset/'
#    darkfolder = '20190907/dark_50nm_offset/'
#    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    #root = "/mnt/96980F95980F72D3/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'eps_peg' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]

#    # =============================================================================
#    #eps peg - Null1 is inverted, null4 is ok
#    # =============================================================================
#    ''' Set the bounds of the parameters to fit '''
#    bounds_mu0  = [(0, 500),   (2200, 2500), (2200, 2500), (0, wl_mid/4), (-400, 400), (6250, 7700)] # bounds for DeltaPhi mu, one tuple per null
#    bounds_sig0 = [(50, 450),  (200, 300),   (200, 300),   (50,200),      (100, 500),  (200, 600)] # bounds for DeltaPhi sig
#    bounds_na0  = [(-0.1, 0.1), (-0.2, 0.2),   (-0.2, 0.2),   (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)] # bounds for astronull
##    bounds_mu0[4] = (-6600, -5900)
##    bounds_sig0[4] = (200, 500)
#    diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
#    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
#    bin_bounds0 = [(-1., 2.), (-0.1, 0.4), (-0.1, 0.4), (-1, 1.5), (-0.5, 1.), (-0.5, 2.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
#    
#    ''' Set the initial conditions '''
#    mu_opd0 = np.array([300, 2400, 2400, 300, -150, 7000]) # initial guess of DeltaPhi mu
#    sig_opd0 = np.array([150, 260, 260, 80, 300, 250]) # initial guess of DeltaPhi sig
#    na0 = np.array([-0.001, -0.001, -0.001, -0.001, -0.001, -0.001]) # initial guess of astro null
#    
#    ''' Import real data '''
#    datafolder = '20190907/eps_peg/'
#    darkfolder = '20190907/dark/'
#    root = "//silo.physics.usyd.edu.au/silo4/snert/"
#    #root = "/mnt/96980F95980F72D3/glint/"
#    file_path = root+'GLINTprocessed/'+datafolder
#    save_path = file_path+'output/'
#    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'eps_peg' in f][nb_files_data[0]:nb_files_data[1]]
#    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]
#
    
    ## =============================================================================
    ##  20200201/RLeo2/
    ## =============================================================================
    #''' Set the bounds of the parameters to fit '''
    #bounds_mu0 = [(-2*wl_mid, 2*wl_mid), (2200, 2500), (2200, 2500), (-1600, -600), (2200, 2500), (2200, 2500)] # bounds for DeltaPhi mu, one tuple per null
    #bounds_sig0 = [(50, 150), (200, 300), (200, 300), (80,150), (150, 250), (200, 300)] # bounds for DeltaPhi sig
    #bounds_na0 = [(0., 0.4), (0., 0.05), (0., 0.01), (0., 0.5), (-0.01, 0.1), (0., 0.05)] # bounds for astronull
    #diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    #xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    #
    #''' Set the initial conditions '''
    #mu_opd0 = np.array([389, 2400, 2400, -1200, 2300, 2300]) # initial guess of DeltaPhi mu
    #sig_opd0 = np.array([100, 260, 260, 110, 200, 201]) # initial guess of DeltaPhi sig
    #na0 = np.array([0.25, 0.001, 0.001, 0.2, 0.001, 0.001]) # initial guess of astro null
    #
    #
    #''' Import real data '''
    #datafolder = '20200201/RLeo2/'
    #darkfolder = '20200201/dark2/'
    #root = "//silo.physics.usyd.edu.au/silo4/snert/"
    ##root = "/mnt/96980F95980F72D3/glint/"
    #file_path = root+'GLINTprocessed/'+datafolder
    #save_path = file_path+'output/'
    #data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'rleo' in f][nb_files_data[0]:nb_files_data[1]]
    #dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]
    
    ## =============================================================================
    ##  20200201/AlfBoo/
    ## =============================================================================
    #''' Set the bounds of the parameters to fit '''
    #bounds_mu0 = [(-wl_mid, wl_mid), (2200, 2500), (2200, 2500), (-wl_mid, wl_mid), (2200, 2500), (2200, 2500)] # bounds for DeltaPhi mu, one tuple per null
    #bounds_sig0 = [(50, 150), (200, 300), (200, 300), (80,150), (50, 150), (200, 300)] # bounds for DeltaPhi sig
    #bounds_na0 = [(0., 0.1), (0., 0.05), (0., 0.01), (0., 0.1), (-0.01, 0.1), (0., 0.05)] # bounds for astronull
    #diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    #xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    #bin_bounds0 = [(-0.1, 2.), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 2.), (-0.1, 0.4), (-0.1, 0.4)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    #
    #''' Set the initial conditions '''
    #mu_opd0 = np.array([389, 2400, 2400, -1200, 2300, 2300]) # initial guess of DeltaPhi mu
    #sig_opd0 = np.array([100, 260, 260, 110, 200, 201]) # initial guess of DeltaPhi sig
    #na0 = np.array([0.25, 0.001, 0.001, 0.2, 0.001, 0.001]) # initial guess of astro null
    #
    #
    #''' Import real data '''
    #datafolder = '20200201/AlfBoo_50nm/'
    #darkfolder = '20200201/dark3_50nm/'
    #root = "//silo.physics.usyd.edu.au/silo4/snert/"
    ##root = "/mnt/96980F95980F72D3/glint/"
    #file_path = root+'GLINTprocessed/'+datafolder
    #save_path = file_path+'output/'
    #data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'AlfBoo' in f][nb_files_data[0]:nb_files_data[1]]
    #dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f][nb_files_dark[0]:nb_files_dark[1]]
    
    # =============================================================================
    # Rock 'n roll
    # =============================================================================
    if len(data_list) == 0 or len(dark_list) == 0:
        raise UserWarning('data list or dark list is empty')
        
    calib_params_path = root+'GLINTprocessed/'+'calibration_params/'
    zeta_coeff_path = calib_params_path + 'zeta_coeff.hdf5'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    ''' Some constants/useless variables kept for retrocompatibility '''
    dphase_bias = 0. # constant value for corrective phase term
    phase_bias = 0. # Phase of the fringes
        
    # List in dictionary: indexes of null, beam A and beam B, zeta label for antinull, segment id
    null_table = {'null1':[0,[0,1], 'null7', [28,34]], 'null2':[1,[1,2], 'null8', [34,25]], 'null3':[2,[0,3], 'null9', [28,23]], \
                  'null4':[3,[2,3], 'null10', [25,23]], 'null5':[4,[2,0], 'null11',[25,28]], 'null6':[5,[3,1], 'null12', [23,34]]}
    
    ''' Specific settings for some configurations '''
    if chi2_map_switch:
        n_samp = int(1e+5) # number of samples per loop
        nloop = 1
     
        
    ''' Fool-proof '''
    if not chi2_map_switch and not skip_fit:
        check_mu =  np.any(mu_opd0 <= np.array(bounds_mu0)[:,0]) or np.any(mu_opd0 >= np.array(bounds_mu0)[:,1])
        check_sig =  np.any(sig_opd0 <= np.array(bounds_sig0)[:,0]) or np.any(sig_opd0 >= np.array(bounds_sig0)[:,1])
        check_null = np.any(na0 <= np.array(bounds_na0)[:,0]) or np.any(na0 >= np.array(bounds_na0)[:,1])
        
        if check_mu or check_sig or check_null:
            raise Exception('Check boundaries: the initial guesses (marked as True) are not between the boundaries (null:%s, mu:%s, sig:%s).'%(check_null, check_mu, check_sig))
        
    total_time_start = time()
    for key in ['null1', 'null4', 'null5', 'null6'][:2]: # Iterate over the null to fit
    # =============================================================================
    # Section of import of data and put them into readable format
    # =============================================================================
        print('****************')
        print('Processing %s \n'%key)
        
        plt.ioff()
        
        if key in nulls_to_invert_model:
            switch_invert_null = True
        else:
            switch_invert_null = False    
       
        ''' Load data about the null to fit '''
        if nonoise:
            data = gff.load_data(data_list, (wl_min, wl_max))
        else:   
            dark = gff.load_data(dark_list, (wl_min, wl_max), key, nulls_to_invert)
            data = gff.load_data(data_list, (wl_min, wl_max), key, nulls_to_invert, dark)
            
        wl_scale = data['wl_scale'] # Wavelength axis. One histogrm per value in this array will be created. The set of histogram will be fitted at once.
        data_photo = data['photo'].copy() # Intensities of the beam of the null which are measuredd in the photometric taps. Axes: id beam, wavelength, frames
        
        ''' Remove dark contribution to measured intensity fluctuations '''
        # The variances of the photometries are supposed to be higher than the variances of the dark noise.
        # If not, it means the photometry is readout noise limited. In that case, the wavelength where it happens is flagged and discarded.
        flags = np.ones(wl_scale.shape, dtype=np.bool)
        if not nonoise:
            # Estimate the mean and variance of the dark and data photometric fluctuations 
            mean_data, var_data = np.mean(data['photo'], axis=-1), np.var(data['photo'], axis=-1)
            mean_dark, var_dark = np.mean(dark['photo'], axis=-1), np.var(dark['photo'], axis=-1)
                
            # Substract variance of dark fluctuations to the variance of the photometric ones
            data_photo = (data_photo - mean_data[:,:,None]) * \
                ((var_data[:,:,None]-var_dark[:,:,None])/var_data[:,:,None])**0.5 + mean_data[:,:,None] - mean_dark[:,:,None]
    
        check_nan = np.all(np.isnan(data_photo), axis=-1) # Check if one beam is NaN for some wavelength due to dark variance higher than data one at this wl
        mask = np.where(check_nan==True)[1]
        mask = np.unique(mask)
        flags[mask] = False
        mask = np.where(check_nan==True)
        for k in range(data_photo.shape[2]):
            data_photo[mask[0], mask[1],k] = mean_data[mask[0], mask[1]]
            
        data_photo = data_photo[:,flags] # Keep only wavelength whic are not readout noise linited
        wl_scale = wl_scale[flags] # Keep only wavelength whic are not readout noise linited
        if wl_scale.size == 0:
            print('Read-noise limited for all wavelengths. We discard this dataset.')
            with open(save_path+'too_noisy_data.log', 'a') as log:
                log.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S")+' Read-noise limited, we get rid of the following dataset (%s files, %s, %s):\n'%(len(data_list),a,b))
                for elt in data_list:
                    log.write(elt+'\n')
                log.write('*********************************************\n')
            continue
    
        ''' Reload data considering the flags'''
        if nonoise:
            data = gff.load_data(data_list, (wl_min, wl_max), flag=flags)
        else:   
            dark = gff.load_data(dark_list, (wl_min, wl_max), key, nulls_to_invert, flag=flags)
            data = gff.load_data(data_list, (wl_min, wl_max), key, nulls_to_invert, dark, flag=flags)    
                
        ''' Load the zeta coeff we need. if "wl_bounds" kew is sent, the return zeta coeff are the average over the bandwidth set by the tuple of this key'''
        if wl_scale.size == 1:
            zeta_coeff = gff.get_zeta_coeff(zeta_coeff_path, wl_scale, False, wl_bounds=(wl_min, wl_max))
        else:
            zeta_coeff = gff.get_zeta_coeff(zeta_coeff_path, wl_scale, False)
        
        if not zeta_switch:
            for key in zeta_coeff.keys():
                if key != 'wl_scale':
                    zeta_coeff[key][:] = 1.
        
        ''' Get histograms of intensities and dark current in the pair of photomoetry outputs '''
        idx_null = null_table[key][0] # Select the index of the null output to process
        idx_photo = null_table[key][1] # Select the indexes of the concerned photometries
        key_antinull = null_table[key][2] # Select the index of the antinull output to process
        segment_id_A, segment_id_B = null_table[key][3] # Select the concerned segments to load their position (outdated functionality)
        data_IA, data_IB = data_photo[0], data_photo[1] # Set photometries in dedicated variable into specific variables for clarity. A and B are the generic id of the beams for the processed baseiune
        
        zeta_minus_A, zeta_minus_B = zeta_coeff['b%s%s'%(idx_photo[0]+1, key)], zeta_coeff['b%s%s'%(idx_photo[1]+1, key)] # Set zeta coeff linking null and photometric outputs into dedicated variables for clarity
        zeta_plus_A, zeta_plus_B = zeta_coeff['b%s%s'%(idx_photo[0]+1, key_antinull)], zeta_coeff['b%s%s'%(idx_photo[1]+1, key_antinull)] # Set zeta coeff linking antinull and photometric outputs into dedicated variables for clarity
    
        ''' Average other the bw '''
        if fullbw:
            data['Iminus'] = np.reshape(data['Iminus'].mean(axis=0), (1,-1))
            data['Iplus'] = np.reshape(data['Iplus'].mean(axis=0), (1,-1))
            data['null'] = data['Iminus'] / data['Iplus']
            if key in nulls_to_invert:
                data['null'] = data['Iplus'] / data['Iminus']
            
        ''' Get CDF of photometric outputs for generating random values in the MC function '''
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

    
    # =============================================================================
    # Section where the histogram/CDF is created
    # =============================================================================
        ''' Make the survival function or PDF'''
        print('Compute survival function and error bars')
        bin_bounds = bin_bounds0[idx_null]
        data_null = data['null'] # Save measured null depth into a dedicated variable. Shape: wavelength, number of measurements
        data_null_err = data['null_err'] # Save uncertainties on measured null depth into a dedicated variable. Same shape as above
    #    sz = max([np.size(np.unique(d)) for d in data_null])
    #    null_axis = np.array([np.linspace(data_null[i].min(), data_null[i].max(), int(sz**0.5), retstep=False, dtype=np.float32) for i in range(data_null.shape[0])])
        sz = np.array([np.size(d[(d>=bin_bounds[0])&(d<=bin_bounds[1])]) for d in data_null]) # size of the sample of measured null depth.
        sz = np.max(sz) # size of the sample of measured null depth.
    #    sz = 1000**2
    #    sz = bin_size**2
        ''' Creation of the x-axis of the histogram (one per wavelength)'''
        if not mode_histo:
            null_axis = np.array([np.linspace(bin_bounds[0], bin_bounds[1], int(sz**0.5), retstep=False, dtype=np.float32) for elt in data_null])
            null_axis = np.array([elt[:-1] + np.diff(elt) for elt in null_axis])
        else:
            null_axis = np.array([np.linspace(bin_bounds[0], bin_bounds[1], int(sz**0.5+1), retstep=False, dtype=np.float32) for i in range(data_null.shape[0])])
    #        null_axis = np.array([np.linspace(data_null[i].min(), data_null[i].max(), int(sz**0.5+1), retstep=False, dtype=np.float32) for i in range(data_null.shape[0])])
    #    null_axis_width = np.mean(np.diff(null_axis, axis=-1))
        
        null_cdf = []
        null_cdf_err = []
        for wl in range(len(wl_scale)):
            if not mode_histo:
                ''' Create the CDF (one per wavelegnth) '''
                if mode == 'cuda':
                    cdf = gff.computeCdf(null_axis[wl], data_null[wl], 'ccdf', True)
                    null_cdf.append(cp.asnumpy(cdf))
                elif mode == 'cupy':
                    cdf = gff.computeCdfCupy(data_null[wl], cp.asarray(null_axis[wl], dtype=cp.float32))
                    null_cdf.append(cp.asnumpy(cdf))
                else:
                    cdf = gff.computeCdf(np.sort(data_null[wl]), null_axis[wl])
                    null_cdf.append(cdf)
                        
                ''' Create the error bars of the histograms '''
                start = time()
    #            cdf_err = gff.getErrorCDF(data_null[wl], data_null_err[wl], null_axis[wl]) # Barnaby's method, tend to underestimate the error
                cdf_err = gff.getErrorBinomNorm(cp.asnumpy(cdf), data_null[wl].size, 1.) # Classic method
                stop = time()
                print('Time CDF error=', stop-start)
                null_cdf_err.append(cdf_err)
    
            else:
                ''' Create the histogram (one per wavelegnth) '''
                pdf = np.histogram(data_null[wl], null_axis[wl], density=False)[0]
                pdf_size = np.sum(pdf)
                print('Histogram size=', np.sum(pdf), np.sum(pdf)/data_null[wl].size)
                bin_width = null_axis[wl][1]-null_axis[wl][0]
                pdf = pdf / np.sum(pdf)
                null_cdf.append(pdf)
                
                start = time()
    #            pdf_err = gff.getErrorPDF(data_null[wl], data_null_err[wl], null_axis[wl]) # Barnaby's method, tend to underestimate the error
                pdf_err = gff.getErrorBinomNorm(pdf, pdf_size, 1.) # Classic method
                stop = time()
                print('Time PDF error=', stop-start)
                null_cdf_err.append(pdf_err)
            if fullbw:
                break
                                        
        null_cdf = np.array(null_cdf)
        null_cdf_err = np.array(null_cdf_err)
    
        ''' Select the bounds of the baseline (null) to process '''
        bounds_mu = bounds_mu0[idx_null]
        bounds_sig = bounds_sig0[idx_null]
        bounds_na = bounds_na0[idx_null]
        # Compile them into a readable tuple called by the TRF algorithm
        bounds_fit = ([bounds_na[0], bounds_mu[0], bounds_sig[0]], 
                      [bounds_na[1], bounds_mu[1], bounds_sig[1]])
    
        
        ''' Generate basin hopping values '''
        if activate_random_init_guesses:
            mu_list, sig_list, null_list = basin_hoppin_values(mu_opd0[idx_null], sig_opd0[idx_null], na0[idx_null], 
                                                               basin_hopping_nloop, bounds_mu, bounds_sig, bounds_na)
            mu_list = mu_list * basin_hopping_nloop0.size
            sig_list = sig_list * basin_hopping_nloop0.size
            null_list = null_list * basin_hopping_nloop0.size
    
    # =============================================================================
    # Section where the fit is done.        
    # =============================================================================
        chi2_liste = [] # Save the reduced Chi2 of the different basin hop
        popt_liste = [] # Save the optimal parameters of the different basin hop
        uncertainties_liste = [] # Save the errors on fitted parameters of the different basin hop
        init_liste = [] # Save the initial guesses of the different basin hop
        pcov_liste = [] # Save the covariance matrix given by the fitting algorithm of the different basin hop
        termination_liste = [] # Save the termination condition of the different basin hop
        wl_scale_saved = wl_scale.copy()
        for basin_hopping_count in range(basin_hopping_nloop[0], basin_hopping_nloop[1]):
            if chi2_map_switch:
                sys.stdout = Logger(save_path+'mapping_%s_%02d'%(key, basin_hopping_count)+'.log') # Save the content written in the console into a txt file
            else:
                sys.stdout = Logger(save_path+'basin_hop_%s_%02d'%(key, basin_hopping_count)+'.log') # Save the content written in the console into a txt file
                
            print('-------------')
            print(basin_hopping_count)
            print('-------------')
            print('Fitting '+key)  
            # model fitting initial guess
            ''' Create the set of initial guess for each hop '''
            if basin_hopping_count == 0 or not activate_random_init_guesses:
                mu_opd = mu_opd0[idx_null]
                sig_opd = sig_opd0[idx_null]
                na = na0[idx_null]
            else:
                mu_opd = mu_list[basin_hopping_count]
                sig_opd = sig_list[basin_hopping_count]
                na = null_list[basin_hopping_count]
                    
            ''' Model fitting '''
            if not chi2_map_switch:
                guess_na = na
                initial_guess = [guess_na, mu_opd, sig_opd]
                initial_guess = np.array(initial_guess, dtype=np.float32)
                if skip_fit:
                    ''' No fit is perforend here, just load the values in the initial guess and compute the histogram'''
                    print('Direct display')
                    count = 0.
                    start = time()
                    out = MCfunction(null_axis, *initial_guess)
                    stop = time()
                    print('Duration:', stop-start)
    #                out = z.reshape(null_cdf.shape)
                    na_opt = na
                    uncertainties = np.zeros(3)
                    popt = (np.array([na, mu_opd, sig_opd]), np.ones((3,3)))
                    chi2 = 1/(null_cdf.size-popt[0].size) * np.sum((null_cdf.ravel() - out)**2/null_cdf_err.ravel()**2)
                    term_status = None
                    print('chi2', chi2)
                
                else:            
                    ''' Fit is done here '''
                    print('Model fitting')    
                    count = 0.
                    init_liste.append(initial_guess)
                    
                    start = time()
    #                popt = curve_fit(MCfunction, null_axis, null_cdf.ravel(), p0=initial_guess, epsfcn = null_axis_width, sigma=null_cdf_err.ravel(), absolute_sigma=True)
    #                res = 0
    #                term_status = None              
                    
                    ''' Save the content of the console generated by this function into a txt file'''
                    with open(save_path+'callfunc_%02d.txt'%(basin_hopping_count), 'w') as fichier:
    #                    popt = gff.curvefit(MCfunction, null_axis, null_cdf.ravel(), p0=initial_guess, sigma=null_cdf_err.ravel(), 
    #                                        bounds=([bounds_na[0], bounds_mu[0], bounds_sig[0], bounds_skew[0]],[bounds_na[1], bounds_mu[1], bounds_sig[1], bounds_skew[1]]), 
    #                                        diff_step = diffstep)
                        popt = gff.curvefit(MCfunction, null_axis, null_cdf.ravel(), p0=initial_guess, sigma=null_cdf_err.ravel(), 
                                            bounds=bounds_fit, diff_step = diffstep, x_scale=xscale)
                        
                    res = popt[2] # all outputs of the fitting function but optimal parameters and covariance matrix (see scipy.optimize.minimize doc)
                    popt = popt[:2] # Optimal parameters found
                    print('Termination:', res.message) # Termination condition
                    term_status = res.status # Value associated by the termination condition
                    
    #                popt = curve_fit(MCfunction, null_axis, null_cdf.ravel(), p0=initial_guess, epsfcn = null_axis_width, sigma=null_cdf_err.ravel(), absolute_sigma=True, 
    #                                 full_output=True)
    #                res = popt[2:]
    #                popt = popt[:2]
    #                term_status = res[2]
                    stop = time()
                    print('Termination', term_status)
                    print('Duration:', stop - start)
    
                    out = MCfunction(null_axis, *popt[0]) # Hsigogram computed according to the optimal parameters
                    uncertainties = np.diag(popt[1])**0.5 # Errors on the optimal parameters
                    chi2 = 1/(null_cdf.size-popt[0].size) * np.sum((null_cdf.ravel() - out)**2/null_cdf_err.ravel()**2) # Reduced Chi2
                    print('chi2', chi2)
                    
                    ''' Display in an easy-to-read way this key information (optimal parameters, error and reduced Chi2) '''
                    na_opt = popt[0][0]
                    print('******')
                    print(popt[0])
                    print(uncertainties*chi2**0.5)
                    print(chi2)
                    print('******')
    
                    ''' Save input and the outputs of the fit into a npz file. One per basin hop '''
                    np.savez(save_path+os.path.basename(file_path[:-1])+'_%s_%03d'%(key, basin_hopping_count),
                             chi2=chi2, popt=[na_opt]+[elt for elt in popt[0][1:]], uncertainties=uncertainties, init=[guess_na]+list(initial_guess[1:]),
                                             termination=np.array([term_status]), nsamp=np.array([n_samp]), wl=wl_scale)
                
                chi2_liste.append(chi2)
                popt_liste.append([na_opt]+[elt for elt in popt[0][1:]])
                uncertainties_liste.append(uncertainties)
                termination_liste.append(term_status)
                pcov_liste.append(popt[1])
                
                ''' Display the results of the fit'''
                # Each figure display the histogram of 10 wavelength, if more than 10 are fitted, extra figures are created
                if fullbw:
                    wl_scale = np.array([wl_scale.mean()])
                    
                wl_idx0 = np.arange(wl_scale.size)
                wl_idx0 = list(gff.divide_chunks(wl_idx0, 10)) # Subset of wavelength displayed in one figure
                flags2 = list(gff.divide_chunks(flags, 10))
                
                for wl_idx in wl_idx0:
                    f = plt.figure(figsize=(19.20,10.80))
                    txt3 = '%s '%key+'Fitted values: ' + 'Na$ = %.2E \pm %.2E$, '%(na_opt, uncertainties[0]) + \
                    r'$\mu_{OPD} = %.2E \pm %.2E$ nm, '%(popt[0][1], uncertainties[1]) + \
                    r'$\sigma_{OPD} = %.2E \pm %.2E$ nm,'%(popt[0][2], uncertainties[2])+' Chi2 = %.2E '%(chi2)+'(Last = %.3f s)'%(stop-start)
    #                txt3 = '%s '%key+'Fitted values: ' + 'Na$ = %.2E \pm %.2E$, '%(na_opt, uncertainties[0]) +' Chi2 = %.2E '%(chi2)+'(Last = %.3f s)'%(stop-start)
                    count = 0
                    flags3 = flags2[wl_idx0.index(wl_idx)]
                    axs = []
        #            wl_idx = wl_idx[wl_scale>=1550]
                    for wl in wl_idx[::-1]:
                        if flags3[list(wl_idx[::-1]).index(wl)]:
                            if len(wl_idx) > 1:
                                ax = f.add_subplot(5,2,count+1)
                            else:
                                ax = f.add_subplot(1,1,count+1)
                            axs.append(ax)
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
    #                        plt.xlim(-0.01, 0.5)
                        count += 1
                    plt.tight_layout(rect=[0., 0.05, 1, 1])
                    if len(wl_idx) > 1:
                        axs[0].text(-0.4, -0.7, txt3, va='center', transform = ax.transAxes, bbox=dict(boxstyle="square", facecolor='white'))
                    else:
                        axs[0].text(0.3, -0.1, txt3, va='center', transform = ax.transAxes, bbox=dict(boxstyle="square", facecolor='white'))
                    string = key+'_'+'%03d'%(basin_hopping_count)+'_'+str(wl_min)+'-'+str(wl_max)+'_'+os.path.basename(datafolder[:-1])+'_%s'%int(wl_scale[wl_idx[-1]])
                    if nonoise: string = string + '_nodarkinmodel'
                    if not oversampling_switch: string = string + '_nooversamplinginmodel'
                    if not zeta_switch: string = string + '_nozetainmodel'
                    if not skip_fit: 
                        if not mode_histo:
                            string = string + '_fit_cdf'
                        else:
                            string = string + '_fit_pdf'
                    plt.savefig(save_path+string+'.png')
                    if basin_hopping_nloop[1]-basin_hopping_nloop[0]>5:
                        plt.close('all')
    
                ''' Plot the histogram of the photometries '''
                ''' Photo A '''
                for wl_idx in wl_idx0:
                    f = plt.figure(figsize=(19.20,10.80))
                    flags3 = flags2[wl_idx0.index(wl_idx)]
                    axs = []
                    count = 0
                    for wl in wl_idx[::-1]:
                        histo_IA = np.histogram(data_IA[wl], int(data_IA[wl].size**0.5), density=True)
                        histo_dIA = np.histogram(dark['photo'][0][wl], int(np.size(dark['photo'][0][wl])**0.5), density=True)
                        if flags3[list(wl_idx[::-1]).index(wl)]:
                            if len(wl_idx) > 1:
                                ax = f.add_subplot(5,2,count+1)
                            else:
                                ax = f.add_subplot(1,1,count+1)
                            axs.append(ax)
                            plt.title('%s nm'%wl_scale[wl])
                            plt.plot(histo_IA[1][:-1], histo_IA[0], '.', markersize=5, label='P%s'%(null_table[key][1][0]+1))
                            plt.plot(histo_dIA[1][:-1], histo_dIA[0], '.', markersize=5, label='Dark')
                            plt.grid()
                            plt.legend(loc='best')
                            plt.xlabel('Flux')
                            plt.ylabel('Frequency')
                        count += 1
                    plt.tight_layout(rect=[0., 0.05, 1, 1])
                    string = 'P%s'%(null_table[key][1][0]+1)+'_'+key+'_'+'%03d'%(basin_hopping_count)+'_'+str(wl_min)+'-'+str(wl_max)+'_'+os.path.basename(datafolder[:-1])+'_%s'%int(wl_scale[wl_idx[-1]])
                    plt.savefig(save_path+string+'.png')
                    if basin_hopping_nloop[1]-basin_hopping_nloop[0]>1:
                        plt.close('all')
    
                ''' Photo B '''
                for wl_idx in wl_idx0:
                    f = plt.figure(figsize=(19.20,10.80))
                    flags3 = flags2[wl_idx0.index(wl_idx)]
                    axs = []
                    count = 0
                    for wl in wl_idx[::-1]:
                        histo_IB = np.histogram(data_IB[wl], int(data_IB[wl].size**0.5), density=True)
                        histo_dIB = np.histogram(dark['photo'][1][wl], int(np.size(dark['photo'][1][wl])**0.5), density=True)
                        if flags3[list(wl_idx[::-1]).index(wl)]:
                            if len(wl_idx) > 1:
                                ax = f.add_subplot(5,2,count+1)
                            else:
                                ax = f.add_subplot(1,1,count+1)
                            axs.append(ax)
                            plt.title('%s nm'%wl_scale[wl])
                            plt.plot(histo_IB[1][:-1], histo_IB[0], '.', markersize=5, label='P%s'%(null_table[key][1][1]+1))
                            plt.plot(histo_dIB[1][:-1], histo_dIB[0], '.', markersize=5, label='Dark')
                            plt.grid()
                            plt.legend(loc='best')
                            plt.xlabel('Flux')
                            plt.ylabel('Frequency')
                        count += 1
                    plt.tight_layout(rect=[0., 0.05, 1, 1])
                    string = 'P%s'%(null_table[key][1][1]+1)+'_'+key+'_'+'%03d'%(basin_hopping_count)+'_'+str(wl_min)+'-'+str(wl_max)+'_'+os.path.basename(datafolder[:-1])+'_%s'%int(wl_scale[wl_idx[-1]])
                    plt.savefig(save_path+string+'.png')
                    if basin_hopping_nloop[1]-basin_hopping_nloop[0]>1:
                        plt.close('all')            
            else:
                ''' Map the parameters space '''
                print('Mapping parameters space')
                count = 0
                map_na, step_na = np.linspace(bounds_na[0], bounds_na[1], 10, endpoint=False, retstep=True)
                map_mu_opd, step_mu = np.linspace(bounds_mu[0], bounds_mu[1], 250, endpoint=False, retstep=True)
                map_sig_opd, step_sig = np.linspace(bounds_sig[0], bounds_sig[1], 10, endpoint=False, retstep=True)
                chi2map = []
                start = time()
                for visi in map_na:
                    temp1 = []
                    for o in map_mu_opd:
                        temp2 = []
                        for s in map_sig_opd:
                            parameters = np.array([visi, o, s])
                            out = MCfunction(null_axis, *parameters)
                            value = 1/(null_cdf.size-parameters.size) * np.sum((null_cdf.ravel() - out)**2/null_cdf_err.ravel()**2)                        
                            temp2.append([value, visi, o, s])
                        temp1.append(temp2)
                    chi2map.append(temp1)
                stop = time()
                chi2map = np.array(chi2map)
                print('Duration: %.3f s'%(stop-start))
                
                np.savez(save_path+'chi2map_%s_%03d'%(key, basin_hopping_count), 
                         value=chi2map, na=map_na, mu=map_mu_opd, sig=map_sig_opd, wl=wl_scale)
                
                chi2map2 = chi2map[:,:,:,0]
                chi2map2[np.isnan(chi2map2)] = np.nanmax(chi2map[:,:,:,0])
                argmin = np.unravel_index(np.argmin(chi2map2), chi2map2.shape)
                print('Min in param space', chi2map2.min(), map_na[argmin[0]], map_mu_opd[argmin[1]], map_sig_opd[argmin[2]])
                print('Indexes are:,', argmin)
                fake = chi2map2.copy()
                fake[argmin] = chi2map2.max()
                argmin2 = np.unravel_index(np.argmin(fake), chi2map2.shape)
                print('2nd min in param space', chi2map2[argmin2], map_na[argmin2[0]], map_mu_opd[argmin2[1]], map_sig_opd[argmin2[2]])
                print('Indexes are:,', argmin2)
                
                valmin = np.nanmin(chi2map[:,:,:,0])
                valmax = np.nanmax(chi2map[:,:,:,0])
    
                ''' plot the 3D parameters space '''
                # WARNING: they are in log scale
                plt.figure(figsize=(19.20,10.80))
                for i in range(map_na.size):
                    if i < 10:
                        plt.subplot(5,2,i+1)
                        plt.imshow(np.log10(chi2map[i,:,:,0].T), interpolation='none', origin='lower', aspect='auto', 
                                   extent=[map_mu_opd[0]-step_mu/2, map_mu_opd[-1]+step_mu/2, map_sig_opd[0]-step_sig/2, map_sig_opd[-1]+step_sig/2],
                                   vmin=np.log10(valmin), vmax=np.log10(valmax))
                        plt.colorbar()
                        plt.ylabel('sig opd');plt.xlabel('mu opd')
                        plt.title('Na %s'%map_na[i])
                plt.tight_layout(rect=[0,0,1,0.95])
                plt.suptitle(key)
                plt.savefig(save_path+'chi2map_%s_%03d_mu_vs_sig.png'%(key, basin_hopping_count))
                
                plt.figure(figsize=(19.20,10.80))
                for i in range(map_sig_opd.size):
                    if i < 10:
                        plt.subplot(5,2,i+1)
                        plt.imshow(np.log10(chi2map[:,:,i,0]), interpolation='none', origin='lower', aspect='auto', 
                                   extent=[map_mu_opd[0]-step_mu/2, map_mu_opd[-1]+step_mu/2, map_na[0]-step_na/2, map_na[-1]+step_na/2],
                                   vmin=np.log10(valmin), vmax=np.log10(valmax))
                        plt.colorbar()
                        plt.title('sig %s'%map_sig_opd[i])
                        plt.xlabel('mu opd');plt.ylabel('null depth')
                plt.tight_layout(rect=[0,0,1,0.95])
                plt.suptitle(key)
                plt.savefig(save_path+'chi2map_%s_%03d_null_vs_mu.png'%(key, basin_hopping_count))
                        
                plt.figure(figsize=(19.20,10.80))
                for i, it in zip(range(argmin[1]-5,argmin[1]+5), range(10)):
                    plt.subplot(5,2,it+1)
                    plt.imshow(np.log10(chi2map[:,i,:,0]), interpolation='none', origin='lower', aspect='auto', 
                               extent=[map_sig_opd[0]-step_sig/2, map_sig_opd[-1]+step_sig/2, map_na[0]-step_na/2, map_na[-1]]+step_na/2,
                               vmin=np.log10(valmin), vmax=np.log10(valmax))
                    plt.colorbar()
                    plt.xlabel('sig opd');plt.ylabel('null depth')    
                    plt.title('mu %s'%map_mu_opd[i])
                plt.tight_layout(rect=[0,0,1,0.95])
                plt.suptitle(key)
                plt.savefig(save_path+'chi2map_%s_%03d_null_vs_sig.png'%(key, basin_hopping_count))
                
            sys.stdout.close()
        
        results = {key:[popt_liste, uncertainties_liste, chi2_liste, init_liste, termination_liste, pcov_liste, wl_scale, n_samp]}
        wl_scale = wl_scale_saved

        if not skip_fit and not chi2_map_switch:
            ''' Save the optimal parameters, inputs, fit information of all basin hop in one run '''
            pickle_name = key+'_'+'%03d'%(basin_hopping_count)+'_'+str(wl_min)+'-'+str(wl_max)+'_'+os.path.basename(datafolder[:-1])
            if mode_histo:
                pickle_name = pickle_name+'_pdf'
            else:
                pickle_name = pickle_name+'_cdf'
            
            pickle_name = pickle_name+'.pkl'
                                               
            with open(save_path+pickle_name, 'wb') as f:
                pickle.dump(results, f)
                
    total_time_stop = time()            
    print('Total time', total_time_stop-total_time_start)
    plt.ion()
    plt.show()
    

    
    print('-- End --')

