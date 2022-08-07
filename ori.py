#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 04:16:28 2021

@author: fernando
"""

import cv2 as cv
import numpy as np
from scipy.signal import correlate


fichero = 'bin3.png'
mask = cv.imread(fichero, cv.IMREAD_GRAYSCALE)
h_crop=4000
w_crop=4000


# Si la máscara supera cierto tamaño se toma solo una parte central
[h,w] = mask.shape
if h_crop > h:
    h_crop = h
if w_crop > w:
    w_crop = w
h_s = int((h-h_crop)/2)
w_s = int((w-w_crop)/2)
mask = mask[h_s:h_s+h_crop,
            w_s:w_s+w_crop]

# Auto-correlación + Otsu
mask = cv.normalize(mask, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
xc = correlate(mask, mask, mode = 'full', method='fft')
xc = cv.normalize(xc, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
cv.imwrite('bin3-corr.png', xc)
_,xc_bin = cv.threshold(xc, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imwrite('bin3-corr-otsu.png', xc_bin)

# Busco el objeto central en la autocorrelación binarizada
[h,w] = xc_bin.shape
center = (int(w/2), int(h/2))
contours,_ = cv.findContours(xc_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cnt_center = None
for cnt in contours:
    if cv.pointPolygonTest(cnt, center , False) >= 0:
        cnt_center = cnt

if cnt_center is not None:
    xc_bin_cnt = cv.cvtColor(xc_bin, cv.COLOR_GRAY2RGB)
    cv.drawContours(xc_bin_cnt, [cnt_center], 0, (255,0,0), cv.FILLED)
    cv.imwrite('bin3-corr-otsu-central.png', xc_bin_cnt)

    # Calculo la orientación del objeto central calculando el ángulo del
    # autovector asociado al menor autovalor en valor absoluto de la
    # matriz de covarianza C
    z = np.matrix(cnt_center[:,0,:])
    K = z.shape[0]
    z_ = np.sum(z,axis=0)/K
    C = np.sum([np.matmul((zk-z_).T,zk-z_) for zk in z],axis=0)/(K-1)
    eigval,eigvec = np.linalg.eig(C)
    v = eigvec[np.where(eigval == np.abs(eigval).max())][0]
    orientation = np.degrees(np.arctan2(v[1],v[0]))
    print(v)
    print(orientation)
else:
    # Si no hay contorno central retorno None
    print(None)
