#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:00:02 2019

@author: mam
"""

import numpy as np

def getPhotoCoeff(fluxes, fluxes_err, output):
    null, null_err = fluxes[1], fluxes_err[1]
    antinull, antinull_err = fluxes[2], fluxes_err[2]
    if output == '1':
        photo, photo_err = fluxes[0], fluxes_err[0]
    elif output == '2':
        photo, photo_err = fluxes[3], fluxes_err[3]
    else:
        raise("Output must be '1' or '2'")
        
    ratio = (null + antinull) / photo
    ratio_err = np.sqrt(1/photo**2 * (null_err**2 + antinull_err**2 + (null+antinull)**2/photo**2 * photo_err**2))
    
    coeff = ratio / (1 + ratio)
    coeff_err = ratio_err / (1 + ratio)**2
    
    return coeff, coeff_err

def getCouplingRatio(fluxes, fluxes_err, output):
    null, null_err = fluxes[1], fluxes_err[1]
    antinull, antinull_err = fluxes[2], fluxes_err[2]
    ech = [np.random.normal(null, null_err, 1000), np.random.normal(antinull, antinull_err, 1000)]
    
    if output == '1': 
        coupling = np.arctan2(null**0.5, antinull**0.5)
        coupling_err = np.std(np.arctan2(ech[0]**0.5, ech[1]**0.5))
    elif output == '2':
        coupling = np.arctan2(antinull**0.5, null**0.5)
        coupling_err = np.std(np.arctan2(ech[1]**0.5, ech[0]**0.5))
    else:
        raise("Output must be '1' or '2'")
        
    
    return coupling, coupling_err
    
#[Phot. output 1, null, antinull, Phot. output 2]
#Output 2:
# labMeas02 (Seg28 turned off with MEMS)
rawFluxes = [1.608598e-03, 1.377138e+00, 1.307289e+00, 1.218056e+00]
rawFluxerr = [6.016353e-07, 3.043493e-05, 2.486730e-05, 2.226069e-05]
coeff, coeff_err = getPhotoCoeff(rawFluxes, rawFluxerr, '2')
print('Photo coeff (to coupler) %0.2f +/- %s' %(coeff, coeff_err))
coupling, coupling_err = getCouplingRatio(rawFluxes, rawFluxerr, '2')
print('Coupling %0.2f +/- %s' %(coupling, coupling_err))
print("Rel diff to pi/4 %0.2f percent" %((coupling-np.pi/4)/(np.pi/4)*100))
print('')


# labMeas07 (Seg36 only illuminated with 450 mask, but pupil edge visible)
rawFluxes = [1.612414e-03, 9.479209e-01, 9.552715e-01, 8.622424e-01]
rawFluxerr = [5.769084e-07, 2.066169e-05, 2.163612e-05, 1.893089e-05]
coeff, coeff_err = getPhotoCoeff(rawFluxes, rawFluxerr, '2')
print('Photo coeff (to coupler) %0.2f +/- %s' %(coeff, coeff_err))
coupling, coupling_err = getCouplingRatio(rawFluxes, rawFluxerr, '2')
print('Coupling %0.2f +/- %s' %(coupling, coupling_err))
print("Rel diff to pi/4 %0.2f percent" %((coupling-np.pi/4)/(np.pi/4)*100))
print('')

#labMeas09 (Seg36 only illuminated with 500 mask)
rawFluxes = [2.137741e-04, 9.650394e-01, 9.810367e-01, 8.828618e-01]
rawFluxerr = [4.630632e-07, 1.640207e-05, 1.675187e-05, 1.485789e-05]
coeff, coeff_err = getPhotoCoeff(rawFluxes, rawFluxerr, '2')
print('Photo coeff (to coupler) %0.2f +/- %s' %(coeff, coeff_err))
coupling, coupling_err = getCouplingRatio(rawFluxes, rawFluxerr, '2')
print('Coupling %0.2f +/- %s' %(coupling, coupling_err))
print("Rel diff to pi/4 %0.2f percent" %((coupling-np.pi/4)/(np.pi/4)*100))
print('')

#Output 1
#Only left hand waveguide (with 550 mask) - right hand blocked by mask
rawFluxes = [2.206061e+00, 1.756225e+00, 2.023874e+00, 3.576269e-04]
rawFluxerr = [3.972499e-05, 3.191172e-05, 3.651780e-05, 8.291792e-07]
coeff, coeff_err = getPhotoCoeff(rawFluxes, rawFluxerr, '1')
print('Photo coeff (to coupler) %0.2f +/- %s' %(coeff, coeff_err))
coupling, coupling_err = getCouplingRatio(rawFluxes, rawFluxerr, '1')
print('Coupling %0.2f +/- %s' %(coupling, coupling_err))
print("Rel diff to pi/4 %0.2f percent" %((coupling-np.pi/4)/(np.pi/4)*100))
print('')

