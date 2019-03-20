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
import matplotlib.pyplot as plt

def rv_generator(absc, pdf):
    '''
    absc : realization of the rv
    pdf : normalized arbitrary pdf to use to generate rv
    '''
    sz = len(pdf)
    cdf = np.zeros((sz,))
    for i in range(1, sz):
        cdf[i] = cdf[i-1] + pdf[i]
        
    ppf = np.interp