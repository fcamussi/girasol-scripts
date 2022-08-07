#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 04:16:28 2021

@author: fernando
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


fichero = 'img1.png'

image = cv.imread(fichero, cv.IMREAD_COLOR)

[h,w,_] = image.shape

h_line = 30

R = np.copy(image[:,:,2])
G = np.copy(image[:,:,1])
B = np.copy(image[:,:,0])

x = np.arange(w)
r = R[h_line,:]
g = G[h_line,:]
b = B[h_line,:]
plt.plot(x,r,'r',linestyle='--',label='R')
plt.plot(x,g,'g',label='G')
plt.plot(x,b,'b',linestyle=':',label='B')
plt.xlabel('Píxeles')
plt.ylabel('Valores')
plt.legend(loc=0)
plt.savefig('perfil-rgb.png', dpi=300)

cv.line(image, (0,h_line), (w,h_line), (0,0,255), 1)

cv.imwrite('img1-linea.png', image)

R=R.astype(int)
G=G.astype(int)
B=B.astype(int)
ExG = 2*G-R-B
ExG[ExG>255]=255
ExG[ExG<0]=0
ExG=ExG.astype('uint8')

cv.imwrite('img1-exg.png', ExG)

e = ExG[h_line,:]
plt.figure()
plt.plot(x,e,'black',label='$ExG_{RGB}$')
plt.xlabel('Píxeles')
plt.ylabel('Valores')
plt.legend(loc=0)
plt.savefig('perfil-exg.png', dpi=300)
