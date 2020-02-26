# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:41:02 2019

@author: nipun
"""

"""
PROBLEM 1, PART B

"""

import numpy as np
from sklearn.decomposition import PCA


x = np.array([0, 1, 2, 2, 3, 3, 4])
y = np.array([1, 1, 1, 3, 2, 3, 5])
x_y = np.vstack((x,y)).T    ##concatenating x and y

print("Dataset is:", x_y)


pca = PCA(n_components = 2)
pca.fit(x_y)
pca_components = pca.components_

print("The PCA Components are:", pca_components)  ##The first row is First Principal Component. Second Row is Second Principal Component.

##Finding new transformed dataset using first principal component
pca1 = PCA(n_components = 1)
pca1.fit(x_y)
x_y_pca1 = pca1.transform(x_y)
print("The transformed data set is:", x_y_pca1)