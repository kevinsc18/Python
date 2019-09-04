# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import statsmodels.api as sm
from sklearn import datasets


#%%
data = datasets.load_boston()


#%%
import numpy as np
import pandas as pd


#%%
# set the features
df = pd.DataFrame(data.data,columns=data.feature_names)

# SET the target

target = pd.DataFrame(data.target, columns=["MEDV"])


#%%
# show the dataset

df.head()
# target.head()


#%%
X = df["RM"]
y = target["MEDV"]

# Fit and make the predictions by the model
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the statistics
model.summary()


#%%
X = sm.add_constant(X)

model = sm.OLS(y,X).fit()

predictions = model.predict(X)

model.summary()


#%%
# fit the regression model with more than one variable

X =df[["RM" , "LSTAT"]]

y = target["MEDV"]

model = sm.OLS(y,X).fit()
predictions = model.predict(X)

model.summary()


#%%
from sklearn import linear_model


#%%
X = df
y = target["MEDV"]
lm = linear_model.LinearRegression()
model = lm.fit(X,y)


#%%
predictions = lm.predict(X)
print(predictions[0:5])


#%%
lm.score(X,y)


#%%
lm.coef_


#%%



