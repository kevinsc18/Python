#%% 
from sklearn.datasets import load_iris
import pandas as pd 


#%%
data = load_iris().data
#%%
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']

#%%
dataset = pd.DataFrame(data,columns=names)

#%%
dataset.head()

#%%
dataset.columns[0]

#%%
# pandaa build in function hist()
fig = plt.figure(figsize = (8,8))
ax = fig.gca()
dataset.hist(ax=ax)
plt.show()

#%%
##  https://www.datacamp.com/community/tutorials/histograms-matplotlib
# use matplotlib get histogram 

import matplotlib.pyplot as plt
import numpy as np 

plt.figure(figsize=[10,10])
f,a = plt.subplots(2,2)
a = a.ravel()
for idx,ax in enumerate(a):
    ax.hist(dataset.iloc[:,idx], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    ax.set_title(dataset.columns[idx])
plt.tight_layout()

#%%
import cv2


#%%
len_rgb = cv2.imread('lena.png')
lena_gray = cv2.cvtColor(lena_rgb,cv2.COLOR_BGR2GRAY)

#%%
lena_gray.shape


#%%
