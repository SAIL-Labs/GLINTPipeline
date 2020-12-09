#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a configuration file to load different data to fit from GLINT
This file is called by ``glint_fitting_gpu6.py`` which reads the dictionary ``config``.

See the example in the code to see how to design a configuration.

NOTE: the configuration is encapsulated into a function to make Sphinx happy.
"""
import numpy as np
import os

def prepareConfig():
    # # =============================================================================
    # # omi cet - Null1 is inverted, null4 is ok v2 measured null depth
    # # =============================================================================
    # starname = 'Omi Cet'
    # date = '2019-09-07'
    # nulls_to_invert = ['null1', 'null6'] # If one null and antinull outputs are swapped in the data processing
    # nulls_to_invert_model = ['null1', 'null6'] # If one null and antinull outputs are swapped in the data processing
    # ''' Set the bounds of the parameters to fit '''
    # bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 700), (4700, 5700), (5500, 6500)] # bounds for DeltaPhi mu, one tuple per null
    # bounds_sig0 = [(50, 250), (200, 300), (200, 300), (40,200), (50, 150), (100, 400)] # bounds for DeltaPhi sig
    # bounds_na0 = [(0., 0.4), (0., 0.05), (0., 0.01), (0., 0.06), (0.02, 0.12), (0.1, 0.2)] # bounds for astronull
    # diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    # xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    # bin_bounds0 = [(-0.5, 1.2), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 1.2), (-0.5, 1.2), (-0.5, 1.2)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
    # ''' Set the initial conditions '''
    # mu_opd0 = np.array([380, 2400, 2400, 200, 5100, 6000]) # initial guess of DeltaPhi mu
    # sig_opd0 = np.array([130, 260, 260, 100, 130, 250]) # initial guess of DeltaPhi sig
    # na0 = np.array([0.17, 0.001, 0.001, 0.02, 0.04, 0.17]) # initial guess of astro null
    
    # ''' Import real data '''
    # datafolder = '20190907/omi_cet/'
    # darkfolder = '20190907/dark/'
    # # root = "//silo.physics.usyd.edu.au/silo4/snert/"
    # root = "/mnt/96980F95980F72D3/glint/"
    # #root = "C:/Users/marc-antoine/glint/"
    # file_path = root+'GLINTprocessed/'+datafolder
    # save_path = file_path+'output/'
    # data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'omi_cet' in f]
    # dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f]
    # calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
    # zeta_coeff_path = calib_params_path + '20191015_zeta_coeff_raw.hdf5'
    
    ## =============================================================================
    ##  20200201/AlfBoo/
    ## =============================================================================
    #starname = 'Alf Boo'
    #date = '2020-02-01'
    #''' Set the bounds of the parameters to fit '''
    #nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
    #nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
    #bounds_mu0 = [(0, 700), (2200, 2500), (2200, 2500), (0, 400), (2200, 2500), (2200, 2500)] # bounds for DeltaPhi mu, one tuple per null
    #bounds_sig0 = [(150, 600), (200, 300), (200, 300), (100,300), (80, 250), (200, 300)] # bounds for DeltaPhi sig
    #bounds_na0 = [(0.0, 0.1), (0., 0.05), (0., 0.01), (0.0, 0.05), (-0.05, 0.15), (0., 0.05)] # bounds for astronull
    #diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    #xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    #bin_bounds0 = [(-0.1, 1), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 1.), (-0.1, 0.4), (-0.1, 0.4)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    #
    #bounds_mu0[0] = (0, 1000)
    #bounds_sig0[0] = (150, 350)
    #
    #''' Set the initial conditions '''
    #mu_opd0 = np.array([155, 2400, 2400, 200, 2300, 2300], dtype=np.float64) # initial guess of DeltaPhi mu
    #sig_opd0 = np.array([250, 260, 260, 215, 200, 201], dtype=np.float64) # initial guess of DeltaPhi sig
    #na0 = np.array([0.08, 0.001, 0.001, 0.011, 0.001, 0.001], dtype=np.float64) # initial guess of astro null
    #
    #mu_opd0[0] = 152
    #sig_opd0[0] = 350
    #na0[0] = 0.06
    #''' Import real data '''
    #datafolder = '20200201/AlfBoo/'
    #darkfolder = '20200201/dark3/'
    #root = "//silo.physics.usyd.edu.au/silo4/snert/"
    ##root = "C:/Users/marc-antoine/glint/"
    ##root = "/mnt/96980F95980F72D3/glint/"
    #file_path = root+'GLINTprocessed/'+datafolder
    #save_path = file_path+'output/'
    #data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'AlfBoo' in f]
    #dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f]
    #calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
    #zeta_coeff_path = calib_params_path + '20200131_zeta_coeff_raw.hdf5'
    
    ## =============================================================================
    ##  20200201/AlfBoo2/
    ## =============================================================================
    #''' Set the bounds of the parameters to fit '''
    #starname = 'Alf Boo'
    #date = '2020-02-01'
    #nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
    #nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
    #bounds_mu0 = [(480, 780), (2200, 2500), (2200, 2500), (450, 750), (4500, 5500),(4500, 5500)] #(1950, 2300) , 2100 for N6 bounds for DeltaPhi mu, one tuple per null
    #bounds_sig0 = [(100, 300), (200, 300), (200, 300), (80,250), (100, 300), (100, 500)] # bounds for DeltaPhi sig
    #bounds_na0 = [(0.0, 0.1), (0., 0.05), (0., 0.01), (0.0, 0.05), (0, 0.1), (0.0, 0.1)] # bounds for astronull
    #diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    #xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    #bin_bounds0 = [(-0.1, 1.), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 1.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    #
    #''' Set the initial conditions '''
    #mu_opd0 = np.array([655, 2400, 2400, 570, 5100, 5100]) # initial guess of DeltaPhi mu
    #sig_opd0 = np.array([220, 260, 260, 148, 200, 200]) # initial guess of DeltaPhi sig
    #na0 = np.array([0.07, 0.001, 0.001, 0.014, 0.03, 0.08]) # initial guess of astro null
    #bounds_mu0[-2] = (5090, 5200)
    ##bounds_na0[-2] = (0.024, 0.025)
    ##mu_opd0[-2] = 5200
    ##sig_opd0[-2] = 250
    ##na0[-2] = 0.024
    #
    #''' Import real data '''
    #datafolder = '20200201/AlfBoo2/'
    #darkfolder = '20200201/dark3/'
    ##root = "//silo.physics.usyd.edu.au/silo4/snert/"
    ##root = "/mnt/96980F95980F72D3/glint/"
    #root = "C:/Users/marc-antoine/glint/"
    #file_path = root+'GLINTprocessed/'+datafolder
    #save_path = file_path+'output/'
    #data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'AlfBoo' in f]
    #dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f]
    #
    #calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
    #zeta_coeff_path = calib_params_path + '20200131_zeta_coeff_raw.hdf5'
    
    ## =============================================================================
    ## NullerData_SubaruJuly2019/20190718/20190718_turbulence1/
    ## =============================================================================
    #''' Set the bounds of the parameters to fit '''
    #starname = 'superK'
    #date = '2019-07-18'
    #nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
    #nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
    #bounds_mu0  = [(0, 400),      (2200, 2500), (2200, 2500), (100, 600), (4500, 5000), (0, 700)] # bounds for DeltaPhi mu, one tuple per null
    #bounds_mu0  = [(0, 1000),      (2200, 2500), (2200, 2500), (0, 1000), (1500, 2500), (9200, 10200)] # bounds for DeltaPhi mu, one tuple per null
    #bounds_sig0 = [(10, 1000),    (200, 300),   (200, 300),   (10,1000),  (10, 1000),  (10, 1000)] # bounds for DeltaPhi sig
    ##    bounds_sig0 = [(100, 400),    (200, 300),   (200, 300),   (50,300),      (100, 500),  (100, 400)] # ron 1271
    #bounds_na0  = [(-0.01, 0.01),   (-0.2, 0.2),  (-0.2, 0.2),  (-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01)] # bounds for astronull
    ##    bounds_mu0[4] = (-6600, -5900)
    ##    bounds_sig0[4] = (200, 500)
    #diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    #xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    #bin_bounds0 = [(-0.02, 1.), (-0.1, 0.4), (-0.1, 0.4), (-0.02, 1.), (-0.02, 1.), (-0.02, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    ##bin_bounds0 = [(-0.25, 1), (-0.1, 0.4), (-0.1, 0.4), (-0.25, 1), (-0.02, 0.1), (-0.02, 0.2)] # 459
    ##    bin_bounds0 = [(-0.6, 1.2), (-0.1, 0.4), (-0.1, 0.4), (-0.6, 1.2), (-0.02, 0.1), (-0.02, 0.2)] # 1271
    #
    #''' Set the initial conditions '''
    #mu_opd0 = np.array([150, 2400, 2400, 370, 2120, 9700]) # initial guess of DeltaPhi mu
    #sig_opd0 = np.array([210, 260, 260, 195, 220, 220]) # initial guess of DeltaPhi sig
    #na0 = np.array([0., 0, 0, 0, 0, 0.005]) # initial guess of astro null
    #
    #factor_minus0 = [0.5, 1, 1, 1.5, 4.5, 2.5] 
    #factor_plus0 = [0.5, 1, 1, 1.5, 2.5, 2] 
    #
    #''' Import real data '''
    #datafolder = 'NullerData_SubaruJuly2019/20190718/20190718_turbulence1/'
    #darkfolder = 'NullerData_SubaruJuly2019/20190718/20190718_dark_turbulence/'
    #root = "//tintagel.physics.usyd.edu.au/snert/"
    ## root = "C:/Users/marc-antoine/glint/"
    ##root = "/mnt/96980F95980F72D3/glint/"
    #file_path = root+'GLINTprocessed/'+datafolder
    #save_path = file_path+'output/'
    ##data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f]
    #data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f]
    #dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f]
    #calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
    #zeta_coeff_path = calib_params_path + '20190715_zeta_coeff_raw.hdf5'
    
    
    ## =============================================================================
    ##  data202006/AlfBoo/ mew wl calib
    ## =============================================================================
    #starname = 'Alf Boo'
    #date = '2020-06-01'
    #''' Set the bounds of the parameters to fit '''
    #nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
    #nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
    #bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 1000), (9700, 10100), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
    #bounds_sig0 = [(100, 300), (200, 300), (200, 300), (10,200), (100, 200), (100, 200)] # bounds for DeltaPhi sig
    #bounds_na0 = [(0.0, 0.1), (0., 0.05), (0., 0.01), (0.0, 0.05), (0., 0.05), (0., 0.1)] # bounds for astronull
    #diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    #xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    #bin_bounds0 = [(-0.1, 1.), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 1.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    #
    #''' Set the initial conditions '''
    #mu_opd0 = np.array([300, 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
    #sig_opd0 = np.array([200, 260, 260, 110, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
    #na0 = np.array([0.08, 0.001, 0.001, 0.011, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    #
    ##mu_opd0[0] = 3.02383619e+02
    ##mu_opd0[3] = 4.05482071e+02
    ##mu_opd0[4] = 9.85782767e+03
    ##mu_opd0[5] = 1.29190582e+04
    ##sig_opd0[0] = 1.63288461e+02
    ##sig_opd0[3] = 1.14174033e+02
    ##sig_opd0[4] = 1.26271045e+02
    ##sig_opd0[5] = 1.70226744e+02
    ##na0[0] = 7.04645603e-02
    ##na0[3] = 1.10478859e-02
    ##na0[4] = 2.32493033e-02
    ##na0[5] = 7.80002694e-02
    #
    #factor_minus0 = [1., 1, 1, 1.5, 4.5, 2.5] 
    #factor_plus0 = [1., 1, 1, 1.5, 2.5, 2] 
    #
    #''' Import real data '''
    #datafolder = 'data202006/AlfBoo/'
    #darkfolder = 'data202006/AlfBoo/'
    #root = "//tintagel.physics.usyd.edu.au/snert/"
    ##root = "C:/Users/marc-antoine/glint/"
    ##root = "/mnt/96980F95980F72D3/glint/"
    #file_path = root+'GLINTprocessed/'+datafolder
    #save_path = file_path+'output/'
    #data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f]
    #dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark1' in f]
    #calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
    #zeta_coeff_path = calib_params_path + '20200604_zeta_coeff_raw.hdf5'
    ##
    ### With fitted amplitude instead of raw
    ##factor_minus0 = [1., 1, 1, 1.5, 4.5, 2.5] 
    ##factor_plus0 = [1., 1, 1, 1.5, 2.5, 2] 
    ##zeta_coeff_path = calib_params_path + '20200604_zeta_coeff_fit.hdf5'
    
    ## =============================================================================
    ##  data202006/AlfBoo/ mew wl calib - binned
    ## =============================================================================
    #starname = 'Alf Boo'
    #date = '2020-06-01'
    #''' Set the bounds of the parameters to fit '''
    #nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
    #nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
    #bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 1000), (20000, 26000), (12500, 13500)] # bounds for DeltaPhi mu, one tuple per null
    #bounds_sig0 = [(10, 210), (200, 300), (200, 300), (10,200), (100, 200), (100, 200)] # bounds for DeltaPhi sig
    #bounds_na0 = [(0.0, 0.1), (0., 0.05), (0., 0.01), (0.0, 0.05), (0., 0.2), (0., 0.2)] # bounds for astronull
    #diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    #xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    #bin_bounds0 = [(-0.1, 1.), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 1.), (-0.1, 1.), (-0.1, 1.)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    #
    ##bounds_mu0[4] = (8000, 10500)
    ##bounds_na0[4] = (0, 0.2)
    ##bounds_sig0[4] = (130, 200)
    #
    #''' Set the initial conditions '''
    #mu_opd0 = np.array([300, 2400, 2400, 400, 9800, 12900], dtype=np.float64) # initial guess of DeltaPhi mu
    #sig_opd0 = np.array([200, 260, 260, 110, 130, 160], dtype=np.float64) # initial guess of DeltaPhi sig
    #na0 = np.array([0.08, 0.001, 0.001, 0.011, 0.025, 0.08], dtype=np.float64) # initial guess of astro null
    #
    ##mu_opd0[0] = 3.02383619e+02
    ##mu_opd0[3] = 4.05482071e+02
    ##mu_opd0[4] = 9.85782767e+03
    ##mu_opd0[5] = 1.29190582e+04
    ##sig_opd0[0] = 1.63288461e+02
    ##sig_opd0[3] = 1.14174033e+02
    ##sig_opd0[4] = 1.26271045e+02
    ##sig_opd0[5] = 1.70226744e+02
    ##na0[0] = 7.04645603e-02
    ##na0[3] = 1.10478859e-02
    ##na0[4] = 2.32493033e-02
    ##na0[5] = 7.80002694e-02
    #
    #factor_minus0 = [1., 1, 1, 1.5, 4.5, 2.5] 
    #factor_plus0 = [1., 1, 1, 1.5, 2.5, 2] 
    #
    #factor_minus0 = [1., 1, 1, 1.5, 4.5, 2] 
    #factor_plus0 = [1., 1, 1, 1.5, 1.5, 1.5] 
    #
    #''' Import real data '''
    #datafolder = 'data202006/AlfBoo/'
    #darkfolder = 'data202006/AlfBoo/'
    #root = "//tintagel.physics.usyd.edu.au/snert/"
    ##root = "C:/Users/marc-antoine/glint/"
    ##root = "/mnt/96980F95980F72D3/glint/"
    #file_path = root+'GLINTprocessed/'+datafolder
    #save_path = file_path+'output/'
    #data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f]
    #dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark1' in f]
    #calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
    #zeta_coeff_path = calib_params_path + '20200604_zeta_coeff_raw.hdf5'
    #
    ### With fitted amplitude instead of raw
    ##factor_minus0 = [1., 1, 1, 1.5, 4.5, 2.5] 
    ##factor_plus0 = [1., 1, 1, 1.5, 2.5, 2] 
    ##zeta_coeff_path = calib_params_path + '20200604_zeta_coeff_fit.hdf5'
    
    ## =============================================================================
    ##  data202006/20200605/AlfBoo/
    ## =============================================================================
    #starname = 'Alf Boo'
    #date = '2020-06-05'
    #''' Set the bounds of the parameters to fit '''
    #nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
    #nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
    #bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 1000), (2200, 2500), (2200, 2500)] # bounds for DeltaPhi mu, one tuple per null
    #bounds_sig0 = [(100, 300), (200, 300), (200, 300), (100,200), (80, 250), (200, 300)] # bounds for DeltaPhi sig
    #bounds_na0 = [(0., 0.1), (0., 0.05), (0., 0.01), (0.0, 0.05), (-0.05, 0.15), (0., 0.05)] # bounds for astronull
    #diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    #xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    #bin_bounds0 = [(-0.3, 1.5), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 1.), (-0.1, 0.4), (-0.1, 0.4)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    #
    #''' Set the initial conditions '''
    #mu_opd0 = np.array([80, 2400, 2400, 400, 2300, 2300], dtype=np.float64) # initial guess of DeltaPhi mu
    #sig_opd0 = np.array([300, 260, 260, 110, 200, 201], dtype=np.float64) # initial guess of DeltaPhi sig
    #na0 = np.array([0.08, 0.001, 0.001, 0.011, 0.001, 0.001], dtype=np.float64) # initial guess of astro null
    #
    #''' Import real data '''
    #datafolder = 'data202006/20200605/AlfBoo/'
    #darkfolder = 'data202006/20200605/AlfBoo/'
    #root = "//silo.physics.usyd.edu.au/silo4/snert/"
    ##root = "C:/Users/marc-antoine/glint/"
    ##root = "/mnt/96980F95980F72D3/glint/"
    #file_path = root+'GLINTprocessed/'+datafolder
    #save_path = file_path+'output/'
    #data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f]
    #dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f]
    #calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
    #zeta_coeff_path = calib_params_path + '20200604_zeta_coeff_raw.hdf5'
    
    ## =============================================================================
    ##  data202006/20200605/Del Vir/
    ## =============================================================================
    #starname = 'Del Vir'
    #date = '2020-06-05'
    #''' Set the bounds of the parameters to fit '''
    #nulls_to_invert = ['null6'] # If one null and antinull outputs are swapped in the data processing
    #nulls_to_invert_model = ['null6'] # If one null and antinull outputs are swapped in the data processing
    #bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 1000), (9250, 10250), (12500, 14000)] # bounds for DeltaPhi mu, one tuple per null
    #bounds_sig0 = [(100, 300), (200, 300), (200, 300), (100,200), (50, 150), (100, 300)] # bounds for DeltaPhi sig
    #bounds_na0 = [(0.0, 0.05), (0., 0.05), (0., 0.01), (0.0, 0.05), (0., 0.01), (0., 0.1)] # bounds for astronull
    #diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    #xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    #bin_bounds0 = [(-0.5, 1.5), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    #
    #''' Set the initial conditions '''
    #mu_opd0 = np.array([300, 2400, 2400, 400, 9800, 13000], dtype=np.float64) # initial guess of DeltaPhi mu
    #sig_opd0 = np.array([200, 260, 260, 110, 120, 240], dtype=np.float64) # initial guess of DeltaPhi sig
    #na0 = np.array([0.016, 0.001, 0.001, 0.002, 0.007, 0.02], dtype=np.float64) # initial guess of astro null
    #
    #factor_minus0 = [1.5, 1, 1, 1.6, 4.5, 0] 
    #factor_plus0 = [1.5, 1, 1, 1.6, 2.5, 0] 
    #
    #''' Import real data '''
    #datafolder = 'data202006/20200605/delVir/'
    #darkfolder = 'data202006/20200605/delVir/'
    #root = "//tintagel.physics.usyd.edu.au/snert/"
    ##root = "C:/Users/marc-antoine/glint/"
    ##root = "/mnt/96980F95980F72D3/glint/"
    #file_path = root+'GLINTprocessed/'+datafolder
    #save_path = file_path+'output/'
    ##data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f]
    #data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f]
    #dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f]
    #calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
    #zeta_coeff_path = calib_params_path + '20200604_zeta_coeff_raw.hdf5'
    
    ## =============================================================================
    ##  data202006/20200609/Del Vir/
    ## =============================================================================
    #starname = 'Del Vir'
    #date = '2020-06-05'
    #''' Set the bounds of the parameters to fit '''
    #nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
    #nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
    #bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 1000), (2200, 2500), (2200, 2500)] # bounds for DeltaPhi mu, one tuple per null
    #bounds_sig0 = [(80, 180), (200, 300), (200, 300), (100, 150), (80, 250), (200, 300)] # bounds for DeltaPhi sig
    #bounds_na0 = [(0., 0.1), (0., 0.1), (0., 0.01), (0.0, 0.1), (-0.05, 0.15), (0., 0.05)] # bounds for astronull
    #diffstep = [0.0001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    #xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    #bin_bounds0 = [(-0.1, 1.), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 1.), (-0.1, 0.4), (-0.1, 0.4)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    #
    #''' Set the initial conditions '''
    #mu_opd0 = np.array([400, 2400, 2400, 400, 2300, 2300], dtype=np.float64) # initial guess of DeltaPhi mu
    #sig_opd0 = np.array([100, 260, 260, 110, 200, 201], dtype=np.float64) # initial guess of DeltaPhi sig
    #na0 = np.array([0.025, 0.001, 0.001, 0.011, 0.001, 0.001], dtype=np.float64) # initial guess of astro null
    #
    #factor_minus0 = [1.3, 1, 1, 1., 1., 2.5] 
    #factor_plus0 = [1.3, 1, 1, 1., 1., 2] 
    #
    #''' Import real data '''
    #datafolder = 'data202006/20200609/delVir/'
    #darkfolder = 'data202006/20200609/delVir/'
    ##root = "//silo.physics.usyd.edu.au/silo4/snert/"
    #root = "C:/Users/marc-antoine/glint/"
    ##root = "/mnt/96980F95980F72D3/glint/"
    #file_path = root+'GLINTprocessed/'+datafolder
    #save_path = file_path+'output/'
    #data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f]
    #dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f]
    #calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
    #zeta_coeff_path = calib_params_path + '20200604_zeta_coeff_raw.hdf5'
    
    ## =============================================================================
    ##  data202006/20200602/turbulence/
    ## =============================================================================
    #starname = 'superK'
    #date = '2020-06-02'
    #''' Set the bounds of the parameters to fit '''
    #nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
    #nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
    #bounds_mu0 = [(0, 1000), (2200, 2500), (2200, 2500), (0, 1000), (2200, 2500), (2200, 2500)] # bounds for DeltaPhi mu, one tuple per null
    #bounds_sig0 = [(1., 100), (200, 300), (200, 300), (100, 150), (80, 250), (200, 300)] # bounds for DeltaPhi sig
    #bounds_na0  = [(-0.01, 0.1),   (-0.2, 0.2),  (-0.2, 0.2),  (-0.01, 0.01), (-0.05, 0.05), (-0.01, 0.01)] # bounds for astronull
    #diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    #xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    #bin_bounds0 = [(-0.1, 1.), (-0.1, 0.4), (-0.1, 0.4), (-0.1, 1.), (-0.1, 0.4), (-0.1, 0.4)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    #
    #''' Set the initial conditions '''
    #mu_opd0 = np.array([300, 2400, 2400, 400, 2300, 2300], dtype=np.float64) # initial guess of DeltaPhi mu
    #sig_opd0 = np.array([50, 260, 260, 110, 200, 201], dtype=np.float64) # initial guess of DeltaPhi sig
    #na0 = np.array([0.001, 0.001, 0.001, 0., 0.001, 0.001], dtype=np.float64) # initial guess of astro null
    #
    #factor_minus0 = [2., 1, 1, 1., 1., 2.5] 
    #factor_plus0 = [2., 1, 1, 1., 1., 2] 
    #
    #''' Import real data '''
    #datafolder = 'data202006/20200602/turbulence/'
    #darkfolder = datafolder
    #root = "//tintagel.physics.usyd.edu.au/snert/"
    ##root = "C:/Users/marc-antoine/glint/"
    ##root = "/mnt/96980F95980F72D3/glint/"
    #file_path = root+'GLINTprocessed/'+datafolder
    #save_path = file_path+'output/'
    #data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f]
    #dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f]
    #calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
    #zeta_coeff_path = calib_params_path + '20200604_zeta_coeff_raw.hdf5'
    
    ## =============================================================================
    ##  data202007/20200705/Del Vir/
    ## =============================================================================
    #starname = 'Del Vir'
    #date = '2020-07-05'
    #''' Set the bounds of the parameters to fit '''
    #nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
    #nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
    #bounds_mu0 = [(400, 800), (2200, 2500), (2200, 2500), (0, 800), (7800, 8600), (10800, 11600)] # bounds for DeltaPhi mu, one tuple per null
    #bounds_sig0 = [(50, 200), (200, 300), (200, 300), (50, 150), (50, 150), (50, 250)] # bounds for DeltaPhi sig
    #bounds_na0 = [(0, 0.1), (0., 0.1), (0., 0.01), (0.0, 0.01), (0, 0.01), (0., 0.03)] # bounds for astronull
    #diffstep = [0.001, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    #xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    #bin_bounds0 = [(-0.5, 1.5), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    #
    #''' Set the initial conditions '''
    #mu_opd0 = np.array([500, 2400, 2400, 540, 8000, 11000], dtype=np.float64) # initial guess of DeltaPhi mu
    #sig_opd0 = np.array([100, 260, 260, 80, 80, 170], dtype=np.float64) # initial guess of DeltaPhi sig
    #na0 = np.array([0.022, 0.001, 0.001, 0.001, 0.007, 0.02], dtype=np.float64) # initial guess of astro null
    #
    ##mu_opd0[0] = 5.28582142e+02
    ##mu_opd0[3] = 5.23691132e+02
    ##mu_opd0[4] = 8.21416419e+03
    ##mu_opd0[5] = 1.12587525e+04
    ##sig_opd0[0] = 1.15588898e+02
    ##sig_opd0[3] = 8.75419645e+01
    ##sig_opd0[4] = 1.11547871e+02
    ##sig_opd0[5] = 1.66979803e+02
    ##na0[0] = 2.14666129e-02
    ##na0[3] = 3.30480150e-03
    ##na0[4] = 6.85718013e-03
    ##na0[5] = 2.46298820e-02
    #
    #factor_minus0 = [1.3, 1, 1, 1.2, 2, 1.] 
    #factor_plus0 = [1.3, 1, 1, 1.2, 1.3, 1.] 
    #
    #''' Import real data '''
    #datafolder = 'data202007/20200705/DelVir/'
    #darkfolder = datafolder
    #root = "//tintagel.physics.usyd.edu.au/snert/"
    ##root = "C:/Users/marc-antoine/glint/"
    ##root = "/mnt/96980F95980F72D3/glint/"
    #file_path = root+'GLINTprocessed/'+datafolder
    #save_path = file_path+'output2/'
    #data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f]
    ##data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n5n6' in f]
    #dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark' in f]
    #calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
    #zeta_coeff_path = calib_params_path + '20200703_zeta_coeff_raw.hdf5'
    
    # =============================================================================
    # 20200917/Capella
    # =============================================================================
    starname = 'Capella'
    date = '2020-09-17'
    ''' Set the bounds of the parameters to fit '''
    nulls_to_invert = [''] # If one null and antinull outputs are swapped in the data processing
    nulls_to_invert_model = [''] # If one null and antinull outputs are swapped in the data processing
    bounds_mu0 = [(0, 500), (2200, 2500), (2200, 2500), (-4000, 0), (7800, 8600), (10800, 11600)] # bounds for DeltaPhi mu, one tuple per null
    bounds_sig0 = [(20, 140), (200, 300), (200, 300), (40, 140), (50, 150), (50, 250)] # bounds for DeltaPhi sig
    bounds_na0 = [(0, 0.6), (0., 0.1), (0., 0.01), (0.0, 0.3), (0, 0.01), (0., 0.03)] # bounds for astronull
    diffstep = [0.01, 10, 10] # differential step to apply to the TRF fitting algorithm, used for computing the finite difference
    xscale = np.ones(len(diffstep)) # scale factor of the parameters to fit, see least_squares doc for more details
    bin_bounds0 = [(-0.5, 2), (-0.1, 0.4), (-0.1, 0.4), (-0.5, 1.25), (-0.5, 1.5), (-0.5, 1.5)] # Boundaries of the histogram, to be set manually after checking the histogram sphape with "skip_fit = True"
    
    ''' Set the initial conditions '''
    mu_opd0 = np.array([200, 2400, 2400, 560, 8000, 11000], dtype=np.float64) # initial guess of DeltaPhi mu
    sig_opd0 = np.array([100, 260, 260, 80, 80, 170], dtype=np.float64) # initial guess of DeltaPhi sig
    na0 = np.array([0.2, 0.001, 0.001, 0.2, 0.007, 0.02], dtype=np.float64) # initial guess of astro null
    
    factor_minus0 = [2., 1, 1, 1.2, 2, 1.] 
    factor_plus0 = [1.3, 1, 1, 1.2, 2, 1.] 
    
    ''' Import real data '''
    datafolder = 'data202009/20200917/Capella/'
    darkfolder = datafolder
    root = "//tintagel.physics.usyd.edu.au/snert/"
    file_path = root+'GLINTprocessed/'+datafolder
    save_path = file_path+'output/'
    data_list = [file_path+f for f in os.listdir(file_path) if '.hdf5' in f and 'n1n4' in f]
    dark_list = [root+'GLINTprocessed/'+darkfolder+f for f in os.listdir(root+'GLINTprocessed/'+darkfolder) if '.hdf5' in f and 'dark2' in f]
    calib_params_path = file_path#root+'GLINTprocessed/calibration_params/'
    zeta_coeff_path = calib_params_path + '20200916_zeta_coeff_raw.hdf5'
    
    # =============================================================================
    # Set the configuration into a dictionay
    # =============================================================================
    config = {'nulls_to_invert': nulls_to_invert,
              'nulls_to_invert_model': nulls_to_invert_model,
              'bounds_mu0': bounds_mu0,
              'bounds_sig0': bounds_sig0,
              'bounds_na0': bounds_na0,
              'diffstep': diffstep,
              'xscale': xscale,
              'bin_bounds0': bin_bounds0,
              'mu_opd0': mu_opd0,
              'sig_opd0': sig_opd0,
              'na0': na0,
              'datafolder': datafolder,
              'darkfolder': darkfolder,
              'root': root,
              'file_path': file_path,
              'save_path': save_path,
              'data_list': data_list,
              'dark_list': dark_list,
              'calib_params_path': calib_params_path,
              'zeta_coeff_path': zeta_coeff_path,
              'starname': starname,
              'date': date,
              'factor_minus0':factor_minus0,
              'factor_plus0':factor_plus0}
    
    return config
