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
import math
from numba import vectorize, cuda
from timeit import default_timer as time

def rv_generator(bins_cent, pdf, nsamp):
    '''
    bins_cent : x-axis of the histogram
    pdf : normalized arbitrary pdf to use to generate rv
    '''
    
    bin_width = np.diff(bins_cent[:2])
    cdf = np.cumsum(pdf) * np.diff(bins_cent[:2])
    cdf, mask = np.unique(cdf, True)
    cdf_bins_cent = bins_cent[mask]
    cdf_bins_cent = cdf_bins_cent +  bin_width/2.

    out = np.random.rand(nsamp)
    
    output_samples = np.interp(out, cdf, cdf_bins_cent)
    output_samples = output_samples.astype(np.float32)
    
    return output_samples

def getHistogram(data, nbins):
    pdf, bins = np.histogram(data, bins=int(nbins**0.5), density=True)
    bins_cent = bins[:-1] + np.diff(bins[:2])/2.
    return pdf, bins_cent

@vectorize(['float32(float32, float32, float32, float32)'], target='cuda')
def computeNullDepth(I1, I2, phase, visibility):
    Iminus = I1 + I2 - 2 * math.sqrt(I1 * I2) * visibility * math.cos(phase)
    Iplus = I1 + I2 - 2 * math.sqrt(I1 * I2) * visibility * math.cos(phase)
    null = Iminus / Iplus
    return null

@vectorize(['float32(float32, float32, float32, float32)'], target='parallel')
def computeNullDepth_parallel(I1, I2, phase, visibility):
    Iminus = I1 + I2 - 2 * math.sqrt(I1 * I2) * visibility * math.cos(phase)
    Iplus = I1 + I2 - 2 * math.sqrt(I1 * I2) * visibility * math.cos(phase)
    null = Iminus / Iplus
    return null

def computeNullDepth_normal(I1, I2, phase, visibility):
    Iminus = I1 + I2 - 2 * np.sqrt(I1 * I2) * visibility * np.cos(phase)
    Iplus = I1 + I2 - 2 * np.sqrt(I1 * I2) * visibility * np.cos(phase)
    null = Iminus / Iplus
    return null

@cuda.jit
def computeNullDepth_cuda(I1, I2, phase, visibility, out):
    i = cuda.grid(1)
    if i < I1.size:
        Iminus = I1[i] + I2[i] - 2 * math.sqrt(I1[i] * I2[i]) * visibility[0] * math.cos(phase[i])
        Iplus = I1[i] + I2[i] - 2 * math.sqrt(I1[i] * I2[i]) * visibility[0] * math.cos(phase[i])
        out[i] = Iminus / Iplus

@vectorize(['float32(float32,float32)'], target='cuda')
def add_kernel(x,y):
    return x+y

@cuda.jit
def add_kernel_cuda(x,y, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = x[i] + y[i]

        
''' Generates mock data '''
n = int(1e+8)
data_I1 = np.random.normal(0., 1., n)
data_I2 = np.random.normal(0., 1., n)
data_phase = np.random.normal(0., 0.1, n)
visibility = 1.

''' Get PDF '''
pdf_I1, bins_cent_I1 = getHistogram(data_I1, int(n**0.5))
pdf_I2, bins_cent_I2 = getHistogram(data_I2, int(n**0.5))
pdf_phase, bins_cent_phase = getHistogram(data_phase, int(n**0.5))

del data_I1, data_I2, data_phase

''' Generate random values from these pdf '''
start = time()
rv_I1 = rv_generator(bins_cent_I1, pdf_I1, n)
rv_I1 = rv_I1.astype(np.float32)
stop = time()
print('rv', stop - start)

rv_I2 = rv_generator(bins_cent_I2, pdf_I2, n)
rv_I2 = rv_I2.astype(np.float32)

rv_phase = rv_generator(bins_cent_phase, pdf_phase, n)
rv_phase = rv_phase.astype(np.float32)

print('Let\'s rock')
start = time()
rv_I1_d = cuda.to_device(rv_I1)
rv_I2_d = cuda.to_device(rv_I2)
rv_phase_d = cuda.to_device(rv_phase)
visibility_d = cuda.to_device(np.array([visibility]).astype(np.float32))
rv_null_d = cuda.device_array(rv_I1.shape, dtype=np.float32)


computeNullDepth(rv_I1_d, rv_I2_d, rv_phase_d, visibility_d, out=rv_null_d)
rv_null = rv_null_d.copy_to_host()
stop = time()
print('gpu', stop - start)

start = time()
rv_null_parallel = computeNullDepth_parallel(rv_I1, rv_I2, rv_phase, visibility)
stop = time()
print('parallel', stop - start)

start = time()
rv_null2 = computeNullDepth_normal(rv_I1, rv_I2, rv_phase, visibility)
stop = time()
print('normal', stop - start)

start = time()
rv_I1_d = cuda.to_device(rv_I1)
rv_I2_d = cuda.to_device(rv_I2)
rv_phase_d = cuda.to_device(rv_phase)
visibility_d = cuda.to_device(np.array([visibility]).astype(np.float32))
rv_null_d = cuda.device_array(rv_I1.shape, dtype=np.float32)

start2 = time()
computeNullDepth_cuda(rv_I1_d, rv_I2_d, rv_phase_d, visibility_d, rv_null_d)
rv_null = rv_null_d.copy_to_host()
stop = time()
print(stop-start2)
print('cuda', stop - start)