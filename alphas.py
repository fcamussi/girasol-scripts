#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 01:42:29 2022

@author: fernando
"""

from counting_file import CountingFile
from descriptors import Descriptors
import cv2 as cv
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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

# Estandarización
data = df.values
stsc = StandardScaler()
data = stsc.fit_transform(data)
X = data[:,:-1]
y = data[:,-1]

# Alphas para Ridge
alphas = 10**np.linspace(-2,10,100) # de 10^-2 a 10^10
coefs = []
for a in alphas:
    ridge = Ridge(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
ax = plt.gca()
plt.plot(alphas, coefs)
plt.legend(df.columns.values, loc=1, prop={'size': 6})
ax.set_xscale('log')
plt.xlabel('$\\lambda$')
plt.ylabel('coeficientes $\\beta$')
plt.savefig('lambdas-ridge.png', dpi=300)

# Alphas para Lasso
alphas = 10**np.linspace(-5,2,100) # de 10^-5 a 10^2
coefs = []
for a in alphas:
    lasso = Lasso(alpha = a, max_iter = 100000)
    lasso.fit(X, y)
    coefs.append(lasso.coef_)
plt.figure()
ax = plt.gca()
plt.plot(alphas, coefs)
plt.legend(df.columns.values, loc=1, prop={'size': 6})
ax.set_xscale('log')
plt.xlabel('$\\lambda$')
plt.ylabel('coeficientes $\\beta$')
plt.savefig('lambdas-lasso.png', dpi=300)
