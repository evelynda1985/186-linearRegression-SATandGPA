#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[7]:


data = pd.read_csv('1.01. Simple linear regression.csv')


# In[8]:


data


# In[11]:


#provides statics data from each column
data.describe()


# In[12]:


#We are going to create a linear regression which predits GPA based on the SAT score (reading, writing and math)
y = data['GPA']
x1 = data['SAT']


# In[13]:


plt.scatter(x1,y)
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()


# In[15]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit() #fit will apply a specific estimation technique (OLS in this case) to obtain the fit model
results.summary()


# In[18]:


plt.scatter(x1,y)
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1, yhat, lw=4, c='orange', label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()


# In[ ]:




