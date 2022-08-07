#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 03:56:01 2022

@author: fernando
"""

from segmentation import segmentation
from morphology import morphology
from rows_orientation import rows_orientation
from rotation import rotation
import cv2 as cv
import os.path


def fun(imageFile, ppm):
    cv_image = cv.imread(imageFile, cv.IMREAD_COLOR)
    cv_mask = segmentation(cv_image)
    cv_mask = morphology(cv_mask, ppm)
    bn = os.path.splitext(os.path.basename(imageFile))[0]
    cv.imwrite('./seg-rot/' + bn + '-mask.png', cv_mask)
    orientation = rows_orientation(cv_mask)
    if orientation is None:
        print(f'Error en la detección de la orientación en {imageFile}')
    else:
        cv_image = rotation(cv_image, -orientation)
        cv.imwrite('./seg-rot/' + bn + '-rot.png', cv_image)
        print(f'{imageFile} & {orientation:.2f} \\\\')


fun('camet1.tif', 143)
fun('camet1cr.jpg', 143)
fun('camet1sr.jpg', 143)
fun('quemu1cr.jpg', 182)
fun('quemu1sr.jpg', 182)
fun('camet2cr.jpg', 131)
fun('camet2sr.jpg', 143)
fun('quemu2cr.jpg', 182)
fun('quemu2sr.jpg', 182)
