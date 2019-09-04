#%%import numpy as np
#%%
np.random.seed(100)
np_hist = np.random.normal(loc=0,scale =1,size =1000)
#%%
np_hist[:10]

#%%
np_hist.mean(),np_hist.std()

#%%
hist,bin_edges = np.histogram(np_hist)

#%%
hist


#%%
bin_edges


#%%
import matplotlib.pyplot as plt

#%%
plt.figure(figsize =[10,8])
plt.bar(bin_edges[:-1],hist,width=0.5,color='#0504aa',alpha=0.7)
plt.xlim(min(bin_edges),max(bin_edges))
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.title('Normal Distribution Histogram',fontsize=15)
plt.show()
#%%
# Let's round the bin_edges to an integer.
bin_edges = np.round(bin_edges,0)


#%%
plt.figure(figsize=[10,8])

plt.bar(bin_edges[:-1], hist, width = 0.5, color='#0504aa',alpha=0.7)..
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.title('Normal Distribution Histogram',fontsize=15)
plt.show()


#%%
plt.figure(figsize=[10,8])
x = 0.3*np.random.randn(1000)
y = 0.3*np.random.randn(1000)
n, bins, patches = plt.hist([x, y])

#%%
