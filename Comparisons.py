#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:37:38 2018

@author: travisbarton
"""

# Comparisons


### SVM
from sklearn import svm
clf = svm.SVC(gamma = 'scale')
clf.fit()