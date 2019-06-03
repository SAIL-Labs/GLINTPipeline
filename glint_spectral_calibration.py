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

# =============================================================================
# Get the shape (position and width) of all tracks
# =============================================================================
print("Getting the shape (position and width) of all tracks")
''' Inputs '''
datafolder = '201806_alfBoo/'
root = "/mnt/96980F95980F72D3/glint/"
data_path = '/mnt/96980F95980F72D3/glint_data/'+datafolder
data_list = [data_path+f for f in os.listdir(data_path) if not 'dark' in f]

''' Output '''
output_path = root+'reduction/'+datafolder
if not os.path.exists(output_path):
    os.makedirs(output_path)

''' Remove dark from the frames and average them to increase SNR '''
#dark = np.load(output_path+'superdark.npy')
#super_img = np.zeros(dark.shape)
#superNbImg = 0.
# 
#for f in data_list[:]:
#    img = glint_classes.File(f)
#    
#    img.data = img.data - dark
#    super_img = super_img + img.data.sum(axis=0)
#    superNbImg = superNbImg + img.nbimg
#
#super_img = super_img / superNbImg
#super_img -= super_img[:,:20].mean()
#np.save(output_path+'avg_alfboo', super_img)

#dark = np.load(output_path+'superdark.npy')
#super_img2 = super_img - super_img[:,:20].mean()
#super_img3 = super_img - dark
#super_img4 = super_img3 - super_img3[:,:20].mean()

#
#plt.figure()
#plt.imshow(super_img4[72-10:72+11])
#plt.colorbar()
#
#plt.figure()
#plt.subplot(211)
#plt.plot(super_img2[72], label='Row 72 - unprocessed')
#plt.plot(super_img3[72], label='Row 72 - dk removed')
#plt.plot(super_img4[72], label='Row 72 - processed')
#plt.plot(dark[72]-dark.mean(), label='Row 72 - dark')
#plt.grid()
#plt.legend(loc='best')
#plt.ylabel('Amplitude')
#plt.xlabel('Column of pixels')
#plt.subplot(212)
#plt.plot(super_img2[71], label='Row 71 - unprocessed')
#plt.plot(super_img3[71], label='Row 71 - dk removed')
#plt.plot(super_img4[71], label='Row 71 - processed')
#plt.plot(dark[71]-dark.mean(), label='Row 71 - dark')
#plt.grid()
#plt.legend(loc='best')
#plt.ylabel('Amplitude')
#plt.xlabel('Column of pixels')


super_img = np.load(output_path+'avg_alfboo.npy')
sh_img = super_img.shape

labels = []
spatial_axis = np.arange(super_img.shape[0])
spectral_axis = np.arange(super_img.shape[1])
nb_tracks = 16 # Number of tracks

y_ends = [33, 329] # row of top and bottom-most Track
sep = (y_ends[1] - y_ends[0])/(nb_tracks-1)

plt.figure(0);plt.clf();plt.imshow(super_img[72-10:72+11], interpolation='none', aspect='auto');plt.colorbar()

''' Fit a gaussian on every track and spectral channel
    to get their positions, widths and amplitude '''

amplitudes = 100*np.ones((nb_tracks,))
channel_pos = np.around(np.arange(y_ends[0], y_ends[1]+sep, sep))
sigmas = np.ones((nb_tracks,))
offsets = np.ones((nb_tracks,)) * 0.25

slices = np.array([super_img[np.int(np.around(pos-sep/2)):np.int(np.around(pos+sep/2)),:] for pos in channel_pos])
slices = np.transpose(slices, (2,0,1))
slices_axes = np.array([spatial_axis[np.int(np.around(pos-sep/2)):np.int(np.around(pos+sep/2))] for pos in channel_pos])
params2 = []
cov2 = []
residuals2 = []

for i in range(slices.shape[0]): # Loop over columns of pixel
    for j in range(slices.shape[1]): # Loop over tracks
        p_init2 = np.array([100, channel_pos[j], 1.])
        try:
            popt, pcov = curve_fit(gaussian, slices_axes[j], slices[i,j], p0=p_init2)
            params2.append(popt)
            cov2.append(np.diag(pcov))
            residuals2.append(slices[i,j] - gaussian(slices_axes[j], *popt))
        except RuntimeError:
            params2.append(np.zeros(p_init2.shape))
            cov2.append(np.zeros(p_init2.shape))
            residuals2.append(np.zeros(slices_axes[j].shape))
            print("Error fit at "+str(i)+"/"+str(j))

params2 = np.array(params2).reshape((sh_img[1],nb_tracks,-1))
cov2 = np.array(cov2).reshape((sh_img[1],nb_tracks,-1))
residuals2 = np.array(residuals2).reshape((sh_img[1],nb_tracks,-1))

params2[:,:,2] = abs(params2[:,:,2]) # convert negative widths into positive ones

''' Fit functions on positions and widths vector to extrapolate on any point 
    along the spectral axis '''
    
bounds = [33, len(spectral_axis)] # Cut the noisy part of the spectral
pos = params2[bounds[0]:bounds[1],:,1]
wi = params2[bounds[0]:bounds[1],:,2]

weight_pos = np.ones(pos.shape)
weight_pos[52-bounds[0]:57-bounds[0],2] = 1.e-36
weight_width = np.ones(wi.shape)
coeff_position_poly = np.array([np.polyfit(spectral_axis[bounds[0]:bounds[1]], pos[:,i], deg=4, w=weight_pos[:,i]) for i in range(nb_tracks)])
coeff_width_poly = np.array([np.polyfit(spectral_axis[bounds[0]:bounds[1]], wi[:,i], deg=4, w=weight_width[:,i]) for i in range(nb_tracks)])
position_poly = [np.poly1d(coeff_position_poly[i]) for i in range(nb_tracks)]
width_poly = [np.poly1d(coeff_width_poly[i]) for i in range(nb_tracks)]

#gaus_mask = np.array([[gaussian(slices_axes[i],1., position_poly[i](spectr), width_poly[i](spectr)) for i in range(nb_tracks)] for spectr in spectral_axis])
#integ = gaus_mask * slices
#integ = np.sum(integ, axis=-1)
#
#gaus_mask2  = np.array([[gaussian(slices_axes[i], params2[j,i,0], position_poly[i](spectral_axis[j]), width_poly[i](spectral_axis[j])) for i in range(nb_tracks)] for j in range(len(spectral_axis))])
#integ2 = np.sum(gaus_mask2, axis=-1)
#
#integ3 = np.sum(slices, axis=-1)
#
#labelsz = 12
#plt.figure()
#plt.grid()
#ax = plt.gca()
#for i in range(nb_tracks):
#    marker, marker2, marker3 = '-', '-', '-'
##    if i > 9: marker, marker2, marker3 = '--', 'd', 'x'
##    color = next(ax._get_lines.prop_cycler)['color']
#    plt.subplot(4,4,i+1)
##    plt.title('Track '+str(i+1), size=labelsz)
#    plt.plot(spectral_axis, integ[:,i], marker, label='Gaussian window')
#    plt.plot(spectral_axis, integ2[:,i], marker2, label='Gaussian model')
#    plt.plot(spectral_axis, integ3[:,i], marker3, label='Raw data')
#    plt.ylim(-10)
#    plt.xticks(size=labelsz);plt.yticks(size=labelsz)
#    plt.ylabel('Track '+str(i+1), size=labelsz)
#    if i >= 12: plt.xlabel('Column of pixels', size=labelsz)
#plt.legend(loc='lower center', ncol=3, bbox_to_anchor = (0.5,0), bbox_transform = plt.gcf().transFigure, fontsize=labelsz)
#plt.figure()
#ax = plt.gca()
#for i in range(nb_tracks):
#    color = next(ax._get_lines.prop_cycler)['color']
#    marker = '-'
#    if i > 9: marker = '--'
#    plt.plot(spectral_axis, params2[:,i,0], marker, lw=3, c=color, label='Track '+str(i+1))
#plt.grid()
#plt.legend(loc='best', ncol=2)
#plt.title('amplitude / track')
#plt.xlim(30)
#plt.ylim(0,500)


#for i in range(nb_tracks):
#    plt.figure()
#    ax = plt.gca()
#    color = next(ax._get_lines.prop_cycler)['color']
#    marker, marker2 = '+-', '.'
##    if i > 9: marker, marker2 = '+--', 'd'
#    plt.subplot(221)
#    plt.plot(spectral_axis[30:], params2[30:,i,1], marker, lw=3, label='Track '+str(i+1))
#    plt.plot(spectral_axis[30:], position_poly[i](spectral_axis[30:]), marker2, label='Poly '+str(i+1))
#    plt.grid()
#    plt.legend(loc='best', ncol=4)
#    plt.ylim(params2[30:,i,1].mean()-0.5,params2[30:,i,1].mean()+0.5)
#    plt.title('x0 / track')
#    plt.ylabel('x0')
#    plt.xlabel('Wavelength')
#    plt.subplot(223)
#    plt.plot(spectral_axis[30:], (params2[30:,i,1]-position_poly[i](spectral_axis[30:]))/params2[30:,i,1]*100)
#    plt.grid()
#    plt.xlabel('Wavelength')
#    plt.ylabel('Residual (%)')
#    plt.ylim(-0.1,0.1)
#    plt.subplot(222)
#    plt.plot(spectral_axis, params2[:,i,2], marker, lw=3, label='Track '+str(i+1))
#    plt.plot(spectral_axis, width_poly[i](spectral_axis), marker2, label='Poly '+str(i+1))
#    plt.grid()
#    plt.legend(loc='best', ncol=4)
#    plt.title('sig / Track')
#    plt.ylabel('Sig')
#    plt.xlabel('Wavelength')
#    plt.xlim(0)
#    plt.ylim(0,2)
#    plt.subplot(224)
#    plt.plot(spectral_axis, (params2[:,i,2]-width_poly[i](spectral_axis))/params2[:,i,2]*100)
#    plt.grid()
#    plt.xlabel('Wavelength')
#    plt.ylabel('Residual (%)')
#    plt.ylim(-10,10)

#for i in range(nb_tracks):
#    plt.figure()
#    ax = plt.gca()
#    color = next(ax._get_lines.prop_cycler)['color']
#    marker, marker2 = '-', 'o'
#    if i > 9: marker, marker2 = '--', 'd'
#    plt.subplot(211)
#    plt.plot(spectral_axis, params2[:,i,2], marker, lw=3, c=color, label='Track '+str(i+1))
#    plt.plot(spectral_axis, width_poly[i](spectral_axis), marker2, c=color, label='Poly '+str(i+1))
#    plt.grid()
#    plt.legend(loc='best', ncol=4)
#    plt.title('sig / Track')
#    plt.ylabel('Sig')
#    plt.xlabel('Wavelength')
#    plt.xlim(0)
#    plt.ylim(0,2)
#    plt.subplot(212)
#    plt.plot(spectral_axis, (params2[:,i,2]-width_poly[i](spectral_axis))/params2[:,i,2]*100)
#    plt.grid()
#    plt.xlabel('Wavelength')
#    plt.ylabel('Residual (%)')
#    plt.ylim(-20,20)

if save:
    np.save(output_path+'coeff_position_poly', coeff_position_poly)
    np.save(output_path+'coeff_width_poly', coeff_width_poly)
plt.close('all')
# =============================================================================
# Spectral calibration
# =============================================================================
print("-----------------------------\nSpectral calibration")
datafolder = '201806_wavecal/'
root = "/mnt/96980F95980F72D3/glint/"
data_path = '/mnt/96980F95980F72D3/glint_data/'+datafolder

''' Output '''
output_path = root+'reduction/'+datafolder
if not os.path.exists(output_path):
    os.makedirs(output_path)

''' Iterate on wavelength '''
wavelength = [1350, 1450, 1550, 1650, 1700][1:]
data_list = [[data_path+f for f in os.listdir(data_path) if not 'dark' in f and str(wl) in f] for wl in wavelength]

''' Remove dark from the frames and average them to increase SNR '''
dark = np.load(output_path+'superdark.npy')

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