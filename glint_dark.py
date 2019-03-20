# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:48:34 2019

@author: mamartinod
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glint_classes

''' Inputs '''
datafolder = '201806_alfBoo/'
root = "C:/glint/"
data_path = root+'data/'+datafolder
dark_list = [data_path+f for f in os.listdir(data_path) if 'dark' in f]

''' Output '''
output_path = root+'reduction/'+datafolder
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
''' Computing average dark '''
superDark = np.zeros((344,96))
superNbImg = 0.

misc = []

for f in dark_list[:]:
#    print("------------\nProcess of : ", f)
    dark = glint_classes.File(f)
    
    superDark = superDark + dark.data.sum(axis=0)
    superNbImg = superNbImg + dark.nbimg
    misc.append(dark.data.mean(axis=(1,2)))
        

if superNbImg != 0.:
    superDark /= superNbImg
    np.save(output_path+'superDark', superDark)

misc = [selt for elt in misc for selt in elt]    
misc = np.array(misc)
plt.figure()
plt.plot(misc)
plt.grid()
plt.xlabel('frame')
plt.ylabel('Average dark current (ADU)')