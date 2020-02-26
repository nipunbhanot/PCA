# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:29:53 2019

@author: nipun
"""

"""
PROBLEM 1, PART A

"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


x = np.array([0, 1, 2, 2, 3, 3, 4])
y = np.array([1, 1, 1, 3, 2, 3, 5])
x_y = np.vstack((x,y)).T ##concatenating x and y

print("Dataset is:", x_y) 


x_y_std = StandardScaler().fit_transform(x_y) #standardizing the two columns separately

print("Standardized data is:", x_y_std)


pca = PCA(n_components = 2) 
pca.fit(x_y_std)
pca_components = pca.components_

print("The PCA Components are:", pca_components)  ##The first row is First Principal Component. Second Row is Second Principal Component. They are in sorted order


##Finding new transformed dataset using first principal component
pca1 = PCA(n_components = 1)
pca1.fit(x_y_std)
x_y_pca1 = pca1.transform(x_y_std)
print("The transformed data set is:", x_y_pca1)