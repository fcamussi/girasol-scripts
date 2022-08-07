#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 02:37:28 2022

@author: fernando
"""

import cv2 as cv
import pandas as pd
from sklearn.model_selection import train_test_split
from PyQt5.QtCore import QRect


file = '../programa/camet1x-marked.png'
coordenadas = '../programa/camet1x-parcelas-testeo.csv'

# parcelas de testeo
_,parcelas = train_test_split(range(1, 151), test_size=0.1, random_state=0)

df_coordenadas = pd.read_csv(coordenadas)
image = cv.imread(file)
for p in parcelas:
    rect = df_coordenadas.loc[df_coordenadas['parcela'] == p].values[:,1:]
    rect = QRect(*rect[0])
    [x,y,w,h] = rect.getRect()
    image2 = image[y:y+h,x:x+w]
    cv.imwrite(f'camet1x-parcela{p}.png', image2)


file = '../../programa/camet2crx-marked.png'
coordenadas = '../../programa/camet2crx-parcelas.csv'
parcelas = list(range(1, 11))

df_coordenadas = pd.read_csv(coordenadas)
image = cv.imread(file)
for p in parcelas:
    rect = df_coordenadas.loc[df_coordenadas['parcela'] == p].values[:,1:]
    rect = QRect(*rect[0])
    [x,y,w,h] = rect.getRect()
    image2 = image[y:y+h,x:x+w]
    cv.imwrite(f'camet2crx-parcela{p}.png', image2)
