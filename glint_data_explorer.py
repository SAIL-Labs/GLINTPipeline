#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:58:33 2019
@author: Marc-Antoine Martinod
Just fill the settings and run the script.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from skimage.measure import moments
from scipy.optimize import curve_fit

nonoise_switch = False
''' Settings '''
nb_files = (0,10) # Number of data files to read. None = all files
root = "/mnt/96980F95980F72D3/glint/" # Root path containing the reduced data
path_to_data = '/mnt/96980F95980F72D3/glint_data/'
datafolder = '20200201/AlfBoo/' # Folder of the data to explore
darkfolder = '20200201/dark3/'
#datafolder = '20191128/turbulence/' # Folder of the data to explore
#darkfolder = '20191128/turbulence/'
wl_path = root+'GLINTprocessed/calibration_params/px_to_wl.npy'
#wl_path = root+'GLINTprocessed/201806_wavecal/px_to_wl.npy'
output_path = root+'GLINTprocessed/'+datafolder # Path to reduced data
dark_path = output_path+'superdark.npy'
wl_min, wl_max = 1400,1650
fps = 10

''' Running script '''
data_path = path_to_data+datafolder # Full path to the data
data_list = sorted([data_path+f for f in os.listdir(data_path)])
data_list = data_list[nb_files[0]:nb_files[1]]
if not nonoise_switch:
    switch_dark = False
    try:
        dark = np.load(dark_path)
    except FileNotFoundError:
        print('No dark found. Create a new one.')
        switch_dark = True
    
    if switch_dark:
        dark_path = path_to_data + darkfolder
        dark_list = [dark_path+f for f in os.listdir(dark_path)][:10]
        with h5py.File(dark_list[0], 'r') as dataFile:
            dark = np.array(dataFile['imagedata'])
            dark = np.transpose(dark, axes=(0,2,1))
            dark = dark.mean(axis=0)    
else:
    print('No-noise data')
    dark = np.zeros((344,96))
    
print('Load data.')
for f in data_list:
    with h5py.File(f) as dataFile:
        try:
            data = np.array(dataFile['imagedata'])
            print('%s / %s \t Number of frames='%(data_list.index(f)+1, len(data_list)), data.shape[0])
        except KeyError as e:
            print(e)
            print(f)
            continue
        data = np.transpose(data, axes=(0,2,1))
        data = data - dark
        
        try:
            stack = np.vstack((stack, data))
        except NameError:
            stack = data

switch_wl = False            
try:
    wl_coeff = np.load(wl_path)
    poly = [np.poly1d(elt) for elt in wl_coeff]
    wl_scale = np.array([p(np.arange(stack.shape[2])) for p in poly])
    # Rearrange wavelegnth scale
    wl_scale = np.array([wl_scale[15], wl_scale[11], wl_scale[3], wl_scale[1],
                         wl_scale[13], wl_scale[9], wl_scale[12], wl_scale[14],
                         wl_scale[2], wl_scale[6], wl_scale[5], wl_scale[8],
                         wl_scale[0], wl_scale[4], wl_scale[7], wl_scale[10]])
#    wl_scale = np.tile(np.arange(96), (16, 1))
except:
    print('No wavelength calibration found. Keep pixel scale.')
    wl_scale = np.tile(np.arange(96), (16, 1))*(-1)
    switch_wl = True
    
print('Let\'s go!')
positions_tracks = [33, 53, 72, 92, 111, 131, 151, 170, 190, 210, 229, 249, 269, 288, 308, 328]
#positions_tracks = [34, 53, 73, 93, 112, 132, 152, 171, 191, 211, 230, 250, 270, 289, 309, 328]
p1, p2, p3, p4 = stack[:,positions_tracks[15]], stack[:,positions_tracks[13]], stack[:,positions_tracks[2]], stack[:,positions_tracks[0]]
n1, n2, n3, n4, n5, n6 = stack[:,positions_tracks[11]], stack[:,positions_tracks[3]], stack[:,positions_tracks[1]], stack[:,positions_tracks[6]], stack[:,positions_tracks[5]], stack[:,positions_tracks[8]]
an1, an2, an3, an4, an5, an6 = stack[:,positions_tracks[9]], stack[:,positions_tracks[12]], stack[:,positions_tracks[14]], stack[:,positions_tracks[4]], stack[:,positions_tracks[7]], stack[:,positions_tracks[10]]
spectrum = p1 / p1.sum(axis=1)[:,None]

#data = np.array([p1, n1, n2, n3,\
#        p2, an1, an2, an3,\
#        p3, n4, n5, n6,\
#        p4, an4, an5, an6])
data = np.array([p1, p2, p3, p4,\
        n1, an1, n2, an2,\
        n3, an3, n4, an4,\
        n5, an5, n6, an6])    
data = np.transpose(data, axes=(1,0,2))
#data /= spectrum[:,None,:]
#data = np.array([np.mean(data, axis=0)])

wl_px = np.tile(np.arange(96), (16, 1))

center_of_mass = np.array([[moments(data[k,i][(wl_scale[i]>wl_min)&(wl_scale[i]<wl_max)], 1)[1] / moments(data[k,i][(wl_scale[i]>wl_min)&(wl_scale[i]<wl_max)], 1)[0] for i in range(16)] for k in range(data.shape[0])])
com_wl = np.array([[poly[i](center_of_mass[j,i] + wl_px[i][wl_scale[i]<wl_max][0]) for i in range(16)] for j in range(center_of_mass.shape[0])])
com_wl = np.array([[np.sum(wl_scale[i]*data[k,i])/np.sum(data[k,i]) for i in range(16)] for k in range(data.shape[0])])
    
#titles_photo = ['P1', 'N1 (12)' ,'N2 (23)', 'N3 (14)',\
#                'P2', 'AN1 (12)', 'AN2 (23)', 'AN3 (14)',\
#                'P3', 'N4 (34)', 'N5 (13)', 'N6 (24)',\
#                'P4', 'AN4 (34)', 'AN5 (13)', 'AN6 (24)']
titles_photo = ['P1', 'P2' ,'P3', 'P4',\
                'N1 (12)', 'AN1 (12)', 'N2 (23)', 'AN2 (23)',\
                'N3 (14)', 'AN3 (14)', 'N4 (34)', 'AN4 (34)',\
                'N5 (13)', 'AN5 (13)', 'N6 (24)', 'AN6 (24)']

fig = plt.figure(figsize=(19.20,10.80))
grid = plt.GridSpec(4, 5, wspace=0.2, hspace=0.55)

frame_ax = fig.add_subplot(grid[:, :1])
axs = []
axs.append(frame_ax)
for i in range(4):
    for j in range(1,5):
        axs.append(fig.add_subplot(grid[i, j]))

lines = [elt.plot([], [], lw=2)[0] for elt in axs[1:]] + \
    [axs[0].imshow(np.zeros(stack[0].shape), interpolation='none', vmin=stack.min(), vmax=1000)]#, extent=[abs(wl_scale.max()), abs(wl_scale.min()), 344, 0], aspect='auto')]
lines2 = [elt.plot([], [], 'o')[0] for elt in axs[1:]]
lines3 = [elt.plot([], [])[0] for elt in axs[1:]]

axs[0].set_xlabel('Wavelength (nm)')
if switch_wl:
    axs[0].set_xlabel('Wavelength (px)')

for i in range(1,17):
    axs[i].set_title(titles_photo[i-1])
    axs[i].set_xlim(abs(wl_scale.max())*1.01, abs(wl_scale.min())*0.99)
#    axs[i].set_xlim(1550-25,1550+25)
    axs[i].set_ylim(data.min()*1.0, data.max()*1.0)
#    axs[i].set_ylim(0, 30000)
    axs[i].grid()
    axs[i].set_xlabel('Wavelength (nm)')
    if switch_wl:
        axs[i].set_xlabel('Wavelength (px)')

wl_scale = abs(wl_scale)
time_text = axs[0].text(0.05, 0.97, '', transform=axs[0].transAxes, color='w')
text_p1 = axs[0].text(0.05, 0.04, 'P1', transform=axs[0].transAxes, color='w')
text_p2 = axs[0].text(0.05, 0.15, 'P2', transform=axs[0].transAxes, color='w')
text_p3 = axs[0].text(0.05, 0.789, 'P3', transform=axs[0].transAxes, color='w')
text_p4 = axs[0].text(0.05, 0.9, 'P4', transform=axs[0].transAxes, color='w')
text_null1 = axs[0].text(0.05, 0.265, 'N1', transform=axs[0].transAxes, color='w')
text_null2 = axs[0].text(0.05, 0.725, 'N2', transform=axs[0].transAxes, color='w')
text_null3 = axs[0].text(0.05, 0.84, 'N3', transform=axs[0].transAxes, color='w')
text_null4 = axs[0].text(0.05, 0.5525, 'N4', transform=axs[0].transAxes, color='w')
text_null5 = axs[0].text(0.05, 0.61, 'N5', transform=axs[0].transAxes, color='w')
text_null6 = axs[0].text(0.05, 0.4375, 'N6', transform=axs[0].transAxes, color='w')
text_antinull1 = axs[0].text(0.05, 0.38, 'AN1', transform=axs[0].transAxes, color='w')
text_antinull2 = axs[0].text(0.05, 0.2075, 'AN2', transform=axs[0].transAxes, color='w')
text_antinull3 = axs[0].text(0.05, 0.0975, 'AN3', transform=axs[0].transAxes, color='w')
text_antinull4 = axs[0].text(0.05, 0.6675, 'AN4', transform=axs[0].transAxes, color='w')
text_antinull5 = axs[0].text(0.05, 0.495, 'AN5', transform=axs[0].transAxes, color='w')
text_antinull6 = axs[0].text(0.05, 0.3225, 'AN6', transform=axs[0].transAxes, color='w')
        
def init3():
    global stack, wl_scale
    lines[-1].set_data(np.zeros(stack[0].shape))
    time_text.set_text('')
    for i in range(16):
            lines[i].set_data(wl_scale[i], np.zeros(wl_scale[i].size))
            lines2[i].set_data(0, 0)
#            lines3[i].set_data(wl_scale[i], np.zeros(wl_scale[i].size))
    return lines + lines2 + [time_text, text_p1, text_p2, text_p3, text_p4,\
            text_null1, text_null2, text_null3, text_null4, text_null5, text_null6,\
            text_antinull1, text_antinull2, text_antinull3, text_antinull4, text_antinull5, text_antinull6]
        
def run2(k):
    global data, stack, wl_scale, com_wl, params
    lines[-1].set_data(stack[k])
    time_text.set_text('Frame %s/%s (%.5f s)'%(k+1, stack.shape[0], (k+1)/fps))
    for i in range(16):
        lines[i].set_data(wl_scale[i], data[k,i,:])
        lines2[i].set_data(com_wl[k,i], 0)
#        if i == 1:
#            lines3[i].set_data(wl_scale[i], func(wl_scale[i], *params[k]))
#        if i == 5:
#            lines3[i].set_data(wl_scale[i], func2(wl_scale[i], *params2[k]))
            
    return lines + lines2 + [time_text, text_p1, text_p2, text_p3, text_p4,\
            text_null1, text_null2, text_null3, text_null4, text_null5, text_null6,\
            text_antinull1, text_antinull2, text_antinull3, text_antinull4, text_antinull5, text_antinull6]

anim = animation.FuncAnimation(fig, run2, init_func=init3, frames=data.shape[0], interval=100)#, blit=True)

plt.figure(figsize=(19.20,10.80))
for i in range(16):
    if i<4: 
        plt.subplot(4,4,i+1)
        plt.plot(data[:,i,45:57].mean(axis=-1))
        plt.grid()
        plt.title('Flux in '+titles_photo[i])
    elif i%2==0:
        plt.subplot(4,4,i+1)
        plt.plot(data[:,i,45:57].mean(axis=-1))
        plt.plot(data[:,i+1,45:57].mean(axis=-1))
#        plt.plot(data[:,i,33:34].sum(axis=-1)/data[:,i+1,33:34].sum(axis=-1))
        plt.grid()
        plt.title('Flux in '+titles_photo[i]+' and '+titles_photo[i+1])
    if i == 1 or i == 3 or i == 12 or i == 14:
        plt.xlabel('Frame')
plt.tight_layout()

plt.figure(figsize=(19.20,10.80))
for i in range(16):
    if i<4: 
        plt.subplot(4,4,i+1)
        plt.plot(data[:,i,45:57].mean(axis=-1))
        plt.grid()
        plt.title('Flux in '+titles_photo[i])
    elif i%2==0:
        plt.subplot(4,4,i+1)
        plt.plot(data[:,i,45:57].mean(axis=-1)/data[:,i+1,45:57].mean(axis=-1))
        plt.grid()
        plt.title('Null '+titles_photo[i])
    if i == 1 or i == 3 or i == 12 or i == 14:
        plt.xlabel('Frame')
plt.tight_layout()  

plt.figure(figsize=(19.20,10.80))
for i in range(16):
    if i<4: 
        histo = np.histogram(data[:,i,45:57].mean(axis=-1), bins=int(np.size(data[:,i,45:57].mean(axis=-1))**0.5), density=True)
        plt.subplot(4,4,i+1)
        plt.plot(histo[1][:-1], histo[0], '.')
        plt.grid()
        plt.title('Histogram of flux in '+titles_photo[i])
    elif i%2==0:
        null = data[:,i,45:57].mean(axis=-1)/data[:,i+1,45:57].mean(axis=-1)
        axis = np.linspace(-1, 3, int(np.size(null)**0.5)+1)
        histo = np.histogram(null, bins=axis, density=True)
        plt.subplot(4,4,i+1)
        plt.plot(histo[1][:-1], histo[0], '.')
        plt.grid()
        plt.title('Histogram of null '+titles_photo[i])
    if i == 1 or i == 3 or i == 12 or i == 14:
        plt.xlabel('Frame')
plt.tight_layout() 
#plt.figure(figsize=(19.20,10.80))
#for i in range(16):
#    plt.subplot(4,4,i+1)
#    plt.plot(com_wl[:,i], 'o-')
#    plt.grid()
#    plt.title('Center of mass of '+titles_photo[i])
#plt.tight_layout()
#
#plt.figure(figsize=(19.20,10.80))
#for i in range(16):
#    plt.subplot(4,4,i+1)
#    b = data[:,i,40:75].sum(axis=-1)
#    histo = np.histogram(b, int(b.size**0.5))
#    plt.plot(histo[1][:-1], histo[0], 'o-')
#    plt.grid()
#    plt.title('Histogram of flux of '+titles_photo[i])
#plt.tight_layout()


#plt.figure(figsize=(19.20,10.80))
#plt.plot(wl_scale[4], data[0,4], lw=3, label='Frame 0')
#plt.plot(wl_scale[4], data[300,4], lw=3, label='Frame 300')
#plt.grid()
#plt.xticks(size=35);plt.yticks(size=35)
#plt.legend(loc='best', fontsize=35)
#plt.xlabel('Wavelength (nm)', size=40)
#plt.ylabel('Intensity (AU)', size=40)
#plt.xlim(1400, 1700)
#plt.tight_layout()

plt.figure(figsize=(19.20,10.80))
plt.plot(wl_scale[0], data[300,0], lw=3, label='P1')
plt.plot(wl_scale[1], data[300,1], lw=3, label='P2')
plt.plot(wl_scale[4], data[300,4], lw=3, label='N1')
plt.plot(wl_scale[5], data[300,5], lw=3, label='AN1')
plt.grid()
plt.xticks(size=35);plt.yticks(size=35)
plt.legend(loc='best', fontsize=35)
plt.xlabel('Wavelength (nm)', size=40)
plt.ylabel('Intensity (AU)', size=40)
plt.xlim(1400, 1700)
plt.tight_layout()

#plt.figure(figsize=(19.20,10.80))
#plt.plot(wl_scale[0], data[300,0], lw=3, label='P1')
#plt.plot(wl_scale[1], data[300,1], lw=3, label='P2')
#plt.plot(wl_scale[4], data[0,2], lw=3, label='P3')
#plt.plot(wl_scale[5], data[0,3], lw=3, label='P4')
#plt.grid()
#plt.xticks(size=35);plt.yticks(size=35)
#plt.legend(loc='best', fontsize=35)
#plt.xlabel('Wavelength (nm)', size=40)
#plt.ylabel('Intensity (AU)', size=40)
#plt.xlim(1400, 1700)
#plt.tight_layout()

photo1 = data[:,0].copy()
photo2 = data[:,1].copy()
photo3 = data[:,2].copy()
photo4 = data[:,3].copy()
photo1 /= photo1.max(axis=1)[:,None]
photo2 /= photo2.max(axis=1)[:,None]
photo3 /= photo3.max(axis=1)[:,None]
photo4 /= photo4.max(axis=1)[:,None]

plt.figure(figsize=(19.20,10.80))
plt.plot(wl_scale[0], photo1.mean(axis=0), lw=3, label='P1')
plt.plot(wl_scale[1], photo2.mean(axis=0), lw=3, label='P2')
plt.plot(wl_scale[4], photo3.mean(axis=0), lw=3, label='P3')
plt.plot(wl_scale[5], photo4.mean(axis=0), lw=3, label='P4')
plt.grid()
plt.xticks(size=30);plt.yticks(size=30)
plt.legend(loc='best', fontsize=35)
plt.xlabel('Wavelength (nm)', size=35)
plt.ylabel('Intensity (AU)', size=35)
plt.title('Visual check of wiggles', size=40)
plt.xlim(1400, 1700)
plt.ylim(-0.05, 1.05)
plt.tight_layout()

plt.figure(figsize=(19.20,10.80))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.plot(wl_scale[i], data[:,i].mean(axis=0))
    plt.grid()
    plt.title(titles_photo[i])
plt.tight_layout()