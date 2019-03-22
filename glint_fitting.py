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
from scipy.interpolate import interp1d
from timeit import default_timer as time
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64
from numba import jit

@cuda.jit
def rv_generator_cuda(rng_states, out):
    thread_id = cuda.grid(1)

    x = xoroshiro128p_uniform_float32(rng_states, thread_id)
    out[thread_id] = x


def rv_generator(bins_cent, pdf, threads_per_block, blocks, rng_states):
    '''
    bins_cent : x-axis of the histogram
    pdf : normalized arbitrary pdf to use to generate rv
    '''
    
    bin_width = np.diff(bins_cent[:2])
    cdf = np.cumsum(pdf) * np.diff(bins_cent[:2])
    cdf, mask = np.unique(cdf, True)
    cdf_bins_cent = bins_cent[mask]
    cdf_bins_cent = cdf_bins_cent +  bin_width/2.
    
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
    out = np.zeros(threads_per_block * blocks, dtype=np.float32)
    rv_generator_cuda[blocks, threads_per_block](rng_states, out)

    output_samples = interp1d(cdf, cdf_bins_cent, bounds_error=False, fill_value=(0,1))(out)
    
    return output_samples, out

def rv_generator2(bins_cent, pdf, threads_per_block, blocks):
    '''
    bins_cent : x-axis of the histogram
    pdf : normalized arbitrary pdf to use to generate rv
    '''
    
    bin_width = np.diff(bins_cent[:2])
    cdf = np.cumsum(pdf) * np.diff(bins_cent[:2])
    cdf, mask = np.unique(cdf, True)
    cdf_bins_cent = bins_cent[mask]
    cdf_bins_cent = cdf_bins_cent +  bin_width/2.
    

    out = np.random.rand(threads_per_block*blocks)
    
    output_samples = interp1d(cdf, cdf_bins_cent, bounds_error=False, fill_value=(0,1))(out)
    
    return output_samples, mask


threads_per_block = 32*8
blocks = 16*10000
rng_states = 0. #create_xoroshiro128p_states(threads_per_block * blocks, seed=1)

n = int(threads_per_block * blocks)
rvn = np.random.randn(n)
x, step = np.linspace(-5, 5, int(n**0.5)+1, retstep=True)
hist, bin_edges = np.histogram(rvn, bins=x, density=True)
bins_cent = bin_edges[:-1] + np.diff(bin_edges[:2])/2.

start = time()
rv, out = rv_generator(bins_cent, hist, threads_per_block, blocks, rng_states)
stop = time()
print(stop-start)

hist2, bins2 = np.histogram(rv, bins=int(n**0.5), density=True)
bins2_cent = bins2[:-1] + np.diff(bins2[:2])/2.

x, step = np.linspace(-5, 5, int(n**0.5)+1, retstep=True)
y = 1./np.sqrt(2*np.pi) * np.exp(-(x[:-1]+step/2)**2/2)
y /= np.sum(y*step)

start = time()
rv2, mask = rv_generator2(x[:-1]+step/2, y, threads_per_block, blocks)
stop = time()
print(stop-start)

hist3, bins3 = np.histogram(rv2, bins=int(n**0.5), density=True)
bins3_cent = bins3[:-1] + np.diff(bins3[:2])/2.

plt.figure()
plt.plot(bins_cent, hist, 'o', label='model')
plt.plot(bins2_cent, hist2, label='histo')
plt.plot(bins3_cent, hist3, label='histo2')
plt.grid()
plt.legend(loc='best')

print(rv.mean(), rv2.mean())