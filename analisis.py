#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 21:46:27 2021

@author: fernando
"""

from counting_file import CountingFile
from descriptors import Descriptors
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


conteo_entrenamiento = '../programa/camet1x-entrenamiento.counting'


def extractDescriptors(counting_file):
    cv_mask = counting_file.getMask()
    contours,_ = cv.findContours(cv_mask, cv.RETR_EXTERNAL,
                                 cv.CHAIN_APPROX_NONE)
    descr_df = Descriptors.compute(contours, counting_file.getPPM(),
                                   counting_file.getPoints())
    return descr_df


counting_file = CountingFile(conteo_entrenamiento)
df = extractDescriptors(counting_file)
print(f'Cantidad muestras entrenamiento: {df.shape[0]}')

X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

# Relacion de cada variable predictora con la de respuesta
plt.figure(figsize=(9, 12))
for i in range(df.shape[1] - 1):
    plt.subplot(4, 3, i+1)
    plt.scatter(X[:,i], y, marker='.')
    xlabel = df.columns[i].replace('_','\\_')
    plt.xlabel(f'${xlabel}$')
    if (i+1) % 3 == 1: plt.ylabel('$CANTIDAD\\_PLANTAS$')
plt.tight_layout()
plt.savefig('relaciones.png', dpi=300)

# Estandarización
data = df.values
stsc = StandardScaler()
data = stsc.fit_transform(data)

# Heatmap
plt.figure()
columns = df.columns.values
cm = np.corrcoef(data.T)
mask = np.zeros_like(cm)
mask[np.diag_indices_from(mask)] = True
cm[np.triu_indices_from(mask)] = np.abs(cm[np.triu_indices_from(mask)])
sns.set(font_scale=0.5)
hm = sns.heatmap(cm,
                 mask = mask,
                 cbar=True,
                 annot=True,
                 square=False,
                 fmt='.2f',
                 annot_kws={'size': 5},
                 yticklabels=columns,
                 xticklabels=columns)
plt.tight_layout()
plt.savefig('heatmap.png', dpi=300)
