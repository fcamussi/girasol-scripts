#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:08:14 2021

@author: fernando
"""

import cv2 as cv
import numpy as np
import math


def __descr_eccentricity(contour):
    z = np.matrix(contour[:,0,:])
    K = z.shape[0]
    z_ = np.sum(z,axis=0)/K
    C = np.sum([np.matmul((zk-z_).T,zk-z_) for zk in z],axis=0)/(K-1)
    eigval,_ = np.linalg.eig(C)
    return math.sqrt(1-(min(eigval)/max(eigval))**2)       

def __vector(contour, ppm):
    _,_,w,h = cv.boundingRect(contour)
    bb_area = w*h
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    hull_perimeter = cv.arcLength(hull, True)

    # 1 - Área en m^2
    area_m = area / ppm**2
    # 2 - Área del bounding box en m^2
    area_bb_m = bb_area / ppm**2
    # 3 - Perímetro en m
    perimeter_m = perimeter / ppm
    # 4 - Perímetro del bounding box en m
    perimeter_bb_m = (2*w+2*h) / ppm
    # 5 - Compacidad
    compactness = perimeter**2 / area
    # 6 - Excentricidad
    eccentricity = __descr_eccentricity(contour)
    # 7 - Relación de aspecto del bounding box
    aspect_ratio_bb = w/h
    # 8 - Extent
    extent = area / bb_area
    # 9 - Convexidad
    convexity = hull_perimeter / perimeter
    # 10 - Solidez
    solidity = area / hull_area

    return dict(AREA_M = area_m,
                AREA_BB_M = area_bb_m,
                PERIMETRO_M = perimeter_m,
                PERIMETRO_BB_M = perimeter_bb_m,
                COMPACIDAD = compactness,
                EXCENTRICIDAD = eccentricity,
                RELACION_ASPECTO_BB = aspect_ratio_bb,
                EXTENT = extent,
                CONVEXIDAD = convexity,
                SOLIDEZ = solidity)


ficheros = ['bin' + str(i) + '.png' for i in range(5,8)]
ppm = 143

for i,n in enumerate([1,3,5]):
    mask = cv.imread(ficheros[i], cv.IMREAD_GRAYSCALE)
    contour,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour = contour[0]
    vector = __vector(contour,ppm)
    print('-------')
    for k,v in vector.items():
        print('%s: %.3f' % (k, v))
    print('CANTIDAD_PLANTAS: %d' % n)
