#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:56:58 2019

@author: mam
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:47:53 2019
@author: mamartinod
Get the spectral flux ratio between null/antinull outputs and photometric tracks
3 steps :
    1) Get the shape (position and width) of all tracks
    2) Spectral calibration of them
    3) Get the flux ratios
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glint_classes
from scipy.optimize import curve_fit
#import skimage.measure as sk

def gaussian(x, A, x0, sig):
    return A * np.exp(-(x-x0)**2/(2*sig**2))# + offset

def polynom(x, *args):
    p = np.poly1d([*args])
    return p(x)

''' Settings '''
save = False


''' Inputs '''
print("-----------------------------\nSpectral calibration")
datafolder = '201806_wavecal/'
root = "/mnt/96980F95980F72D3/glint/"
data_path = '/mnt/96980F95980F72D3/glint_data/'+datafolder
output_path = root+'reduction/'+datafolder

coeff_position_poly = np.load(output_path+'coeff_position_poly.npy')
coeff_width_poly = np.load(output_path+'coeff_width_poly.npy')

''' Output '''
output_path = root+'reduction/'+datafolder
if not os.path.exists(output_path):
    os.makedirs(output_path)

''' Iterate on wavelength '''
wavelength = [1350, 1450, 1550, 1650, 1700][1:]
data_list = [[data_path+f for f in os.listdir(data_path) if not 'dark' in f and str(wl) in f] for wl in wavelength]

''' Remove dark from the frames and average them to increase SNR '''
dark = np.load(output_path+'superdark.npy')

''' Run '''
nb_tracks = 16
y_ends = [33, 329] # row of top and bottom-most Track
sep = (y_ends[1] - y_ends[0])/(nb_tracks-1)
channel_pos = np.around(np.arange(y_ends[0], y_ends[1]+sep, sep))

position_poly = [np.poly1d(coeff_position_poly[i]) for i in range(nb_tracks)]
width_poly = [np.poly1d(coeff_width_poly[i]) for i in range(nb_tracks)]

calib_pos = []

for elt in data_list[:]:
    super_img = np.zeros(dark.shape)
    superNbImg = 0.
     
    for f in elt:
        img = glint_classes.File(f, transpose=True)        
        img.data = img.data - dark
        super_img = super_img + img.data.sum(axis=0)
        superNbImg = superNbImg + img.nbimg
    
    super_img = super_img / superNbImg
    super_img -= super_img[:25].mean()
    spatial_axis = np.arange(super_img.shape[0])
    spectral_axis = np.arange(super_img.shape[1])

    plt.figure()
    plt.imshow(super_img, interpolation='none', aspect='auto', vmin=-100, vmax=100)
    plt.colorbar()
    
    slices = np.array([super_img[np.int(np.around(pos-sep/2)):np.int(np.around(pos+sep/2)),:] for pos in channel_pos])
    slices_axes = np.array([spatial_axis[np.int(np.around(pos-sep/2)):np.int(np.around(pos+sep/2))] for pos in channel_pos])

#    sz_boxes = 20
#    maxi = np.array([np.unravel_index(np.argmax(s), s.shape) for s in slices])
#    boxes = [slices[i,:,int(maxi[i,1]-sz_boxes/2):int(maxi[i,1]+sz_boxes/2)+1] for i in range(slices.shape[0])]
#    moment = np.array([sk.moments(b, order=1) for b in boxes])
#    photocenter_spatial = np.array([m[1, 0] / m[0, 0] for m in moment]) - sep/2. + channel_pos
#    photocenter_wl = np.array([m[0, 1] / m[0, 0] for m in moment]) - sz_boxes/2 + maxi[:,1]
#    calib_pos.append(np.array([photocenter_wl, photocenter_spatial]))
#    plt.scatter(photocenter_wl, photocenter_spatial, marker='+', color='r')
    
    ygrid = np.array([np.around(p(spectral_axis)) for p in position_poly], dtype=np.int)
    xgrid = np.meshgrid(spectral_axis, spectral_axis)[0][:nb_tracks]
    tracks = super_img[ygrid, xgrid]
    wl_pos = []
    for i in range(nb_tracks):
        popt, pcov = curve_fit(gaussian, spectral_axis, tracks[i], p0=[100., spectral_axis[np.argmax(tracks[i])], 1.])
        wl_pos.append(popt[1:])
    calib_pos.append(wl_pos)

calib_pos = np.array(calib_pos)

coeff_poly_wl_to_px = np.array([np.polyfit(wavelength, calib_pos[:,i,0], deg=1) for i in range(nb_tracks)]) # detector ersolution is around 5 nm/px
coeff_poly_px_to_wl = np.array([np.polyfit(calib_pos[:,i,0], wavelength, deg=1) for i in range(nb_tracks)])
poly_wl = [np.poly1d(coeff_poly_wl_to_px[i]) for i in range(nb_tracks)]

if save:
    np.save(output_path+'wl_to_px', coeff_poly_wl_to_px)
    np.save(output_path+'px_to_wl', coeff_poly_px_to_wl)

fwhm = 2 * np.sqrt(2*np.log(2)) * calib_pos[:,:,1] * abs(coeff_poly_px_to_wl[None,:,0])
print('Spectral resolution for')
for wl in wavelength:
    print(str(wl)+' nm -> '+str(wl/fwhm.mean(axis=1)[wavelength.index(wl)]))

plt.figure()
for i in range(nb_tracks):
    plt.subplot(4,4,i+1)
    plt.plot(wavelength, calib_pos[:,i,0], 'o')
    plt.plot(wavelength, poly_wl[i](wavelength))
    plt.grid()
    plt.title('Track %s'%(i+1))
#plt.figure()
#plt.grid()
#ax = plt.gca()
#for i in range(nb_tracks):
#    marker1, marker2 = 'o', '-'
#    if i > 9 : marker1, marker2 = 'd', '--'
#    color = next(ax._get_lines.prop_cycler)['color']
#    plt.plot(spectral_axis, position_poly[i](spectral_axis), marker2, c=color)
#    plt.plot(calib_pos[:,0,i], calib_pos[:,1,i], marker1, c=color)
#    plt.xlim(-5,100)

   

#for i in range(nb_tracks):
#    plt.figure(i) 
#    plt.grid()
#    ax = plt.gca()
#    marker1, marker2 = 'o', '-'
#    if i > 9 : marker1, marker2 = 'd', '--'
#    color = next(ax._get_lines.prop_cycler)['color']
#    plt.plot(calib_pos[:,i,0], wavelength, marker1, c=color)
#    plt.plot(calib_pos[:,i,0], poly_wl[i](calib_pos[:,i,0]), marker2)
#    plt.ylim(1340, 1750)