#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 05:45:46 2021

@author: fernando
"""

from counting_file import CountingFile
from descriptors import Descriptors
import cv2 as cv
import numpy as np
from sklearn.linear_model import (LinearRegression,
                                  Ridge, RidgeCV,
                                  Lasso, LassoCV)
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import math
import pickle


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
print('--- . ---')

data = df.values
X = data[:,:-1]
y = data[:,-1]

# Estandarización
stsc_X = StandardScaler()
stsc_y = StandardScaler()
X = stsc_X.fit_transform(X)
y = stsc_y.fit_transform(y.reshape(-1,1)).flatten()


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


alphas_ridge = 10**np.linspace(-2,10,100) # de 10^-2 a 10^10
alphas_lasso = 10**np.linspace(-5,2,100) # de 10^-5 a 10^2
Cs_svr = [0.01,0.1,1,10,100]


def reg_ls():
    # Mínimos cuadrados
    ls = LinearRegression()
    ls.fit(X_train, y_train)
    model_ls.append(ls)
    pred = ls.predict(X_test)
    r2 = r2_score(y_test, pred)
    r2_ls.append(r2)

def reg_ridge():
    # Ridge regression
    ridgecv = RidgeCV(alphas=alphas_ridge, cv=10)
    ridgecv.fit(X_train, y_train)
    ridge = Ridge(alpha = ridgecv.alpha_)
    ridge.fit(X_train, y_train)
    model_ridge.append(ridge)
    pred = ridge.predict(X_test)
    r2 = r2_score(y_test, pred)
    r2_ridge.append(r2)

def reg_lasso():
    # Lasso regression
    lassocv = LassoCV(alphas=alphas_lasso, cv=10, max_iter=10000, n_jobs=8)
    lassocv.fit(X_train, y_train)
    lasso = Lasso(alpha=lassocv.alpha_, max_iter=10000)
    lasso.fit(X_train, y_train)
    model_lasso.append(lasso)
    pred = lasso.predict(X_test)
    r2 = r2_score(y_test, pred)
    r2_lasso.append(r2)

def reg_sv():
    # Support vector regression
    svr = SVR(kernel='linear', max_iter=-1)
    gscv = GridSearchCV(svr, {'C': Cs_svr}, n_jobs=8)
    gscv.fit(X_train, y_train)
    svr.set_params(**gscv.best_params_)
    svr.fit(X_train, y_train)
    model_svr.append(svr)
    pred = svr.predict(X_test)
    r2 = r2_score(y_test, pred)
    r2_svr.append(r2)


model_ls = []
model_ridge = []
model_lasso = []
model_svr = []
r2_ls = []
r2_ridge = []
r2_lasso = []
r2_svr = []
for i in range(10):
    kf = KFold(n_splits=10, shuffle=True, random_state=i)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        reg_ls()
        reg_ridge()
        reg_lasso()
        reg_sv()

plt.figure()
plt.boxplot([r2_ls,r2_ridge,r2_lasso,r2_svr],
            labels=['Mínimos cuadrados','Ridge','Lasso', 'SVR'])
plt.savefig('boxplots-r2.png', dpi=300)


N = 100
print(f'N={N}')

# Error R2
r2_promedio_ls = np.mean(r2_ls)
r2_promedio_ridge = np.mean(r2_ridge)
r2_promedio_lasso = np.mean(r2_lasso)
r2_promedio_svr = np.mean(r2_svr)
r2_std_ls = np.std(r2_ls)
r2_std_ridge = np.std(r2_ridge)
r2_std_lasso = np.std(r2_lasso)
r2_std_svr = np.std(r2_svr)
r2_stderr_ls = r2_std_ls/math.sqrt(N)
r2_stderr_ridge = r2_std_ridge/math.sqrt(N)
r2_stderr_lasso = r2_std_lasso/math.sqrt(N)
r2_stderr_svr = r2_std_svr/math.sqrt(N)
print('R2')
# print(f'LS: {r2_promedio_ls:.2f} +- {r2_stderr_ls:.2f}')
# print(f'Ridge: {r2_promedio_ridge:.2f} +- {r2_stderr_ridge:.2f}')
# print(f'Lasso: {r2_promedio_lasso:.2f} +- {r2_stderr_lasso:.2f}')
# print(f'SVR: {r2_promedio_svr:.2f} +- {r2_stderr_svr:.2f}')
print(f'LS: {r2_promedio_ls:.2f}, {r2_std_ls:.2f}')
print(f'Ridge: {r2_promedio_ridge:.2f}, {r2_std_ridge:.2f}')
print(f'Lasso: {r2_promedio_lasso:.2f}, {r2_std_lasso:.2f}')
print(f'SVR: {r2_promedio_svr:.2f}, {r2_std_svr:.2f}')
print('--- . ---')

pickle.dump(stsc_X, open('data/stsc_X', 'wb'))
pickle.dump(stsc_y, open('data/stsc_y', 'wb'))
print('Modelos con mayor R2:')
print('LS:')
i = r2_ls.index(max(r2_ls))
print(f'intercept: {model_ls[i].intercept_:.3f}')
for k,c in enumerate(model_ls[i].coef_):
    print(f'{df.columns[k]}: {c:.3f}')
pickle.dump(model_ls[i], open('data/model_ls', 'wb'))

print('Ridge:')
i = r2_ridge.index(max(r2_ridge))
print(f'intercept: {model_ridge[i].intercept_:.3f}')
for k,c in enumerate(model_ridge[i].coef_):
    print(f'{df.columns[k]}: {c:.3f}')
pickle.dump(model_ridge[i], open('data/model_ridge', 'wb'))

print('Lasso:')
i = r2_lasso.index(max(r2_lasso))
print(f'intercept: {model_lasso[i].intercept_:.3f}')
for k,c in enumerate(model_lasso[i].coef_):
    print(f'{df.columns[k]}: {c:.3f}')
pickle.dump(model_lasso[i], open('data/model_lasso', 'wb'))

print('SVR:')
i = r2_svr.index(max(r2_svr))
print(f'intercept: {model_svr[i].intercept_[0]:.3f}')
for k,c in enumerate(model_svr[i].coef_[0]):
    print(f'{df.columns[k]}: {c:.3f}')
pickle.dump(model_svr[i], open('data/model_svr', 'wb'))
