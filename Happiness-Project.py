#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go


# ### Importing Data and having a look

# In[3]:



df = pd.read_csv(r'C:\Users\lenovo\Desktop\D-S\happiness_score_dataset.csv')


# In[4]:


df.head(5)


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.columns


# #### Happiness Score and other features sorted by region

# In[8]:


df.groupby('Region')['Happiness Rank', 'Happiness Score', 'Standard Error', 'Economy (GDP per Capita)'].mean()


# ### Visualization - top 12 countries happines score

# In[13]:


plt.figure(figsize=(14,8))
topCountry=df.sort_values(by=['Happiness Rank'],ascending=True).head(12)
ax=sns.barplot(x='Country',y='Happiness Score', data=topCountry)
ax.set(xlabel='Country', ylabel='Happiness Score')


# #### Heatmap
# #### Correlation - Relation between each others

# In[16]:


plt.figure(figsize=(10,8))
corr = df.drop(['Country','Region','Happiness Rank'],axis = 1).corr()
sns.heatmap(corr, cbar = True, square = True, annot=True, linewidths = .5, fmt='.2f',annot_kws={'size': 15}) 
plt.title('Heatmap of Correlation Matrix')
plt.show()


# ### Using linear regression plotting

# In[18]:


plt.figure(figsize=(12,8))
sns.regplot(x='Generosity',y='Happiness Score' ,data=df)


# ### from upper figure Generorsity doesnt effect happiness score effectively

# In[17]:


plt.figure(figsize=(12,8))
sns.regplot(x='Economy (GDP per Capita)',y='Happiness Score' ,data=df)
#so there is a linear relation between GDP & Happiness Score


# #### From upper figure GDP and Happiness score are strongly related

# #### Below are other figures to see effect on happiness score

# In[20]:


cols = ['Standard Error', 'Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)']
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(cols):
    ax = plt.subplot(gs[i])
    #sns.distplot(df1[cn], bins=50)
    sns.regplot(x=df[cn],y='Happiness Score' ,data=df)
    ax.set_xlabel('')
    ax.set_title('Regrassion of feature: ' + str(cn))
plt.show()


# ### Region and Happiness score relation 

# In[29]:


plt.figure(figsize=(12,6))
sns.stripplot(x="Region", y="Happiness Score", data=df, jitter=True)
plt.xticks(rotation=90)
plt.show()


# ### Let's Construct model for predicting Happines score

# In[21]:


X = df.drop(['Happiness Score', 'Happiness Rank', 'Country', 'Region'], axis=1)
y = df['Happiness Score']


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[30]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print('Standardized features\n')
print(str(X_train[:5]))


# In[24]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)


# In[35]:


lm.score(X_train,y_train)


# In[36]:


lm.score(X_test,y_test)


# In[31]:


lm_result = pd.DataFrame({
    'Actual':y_test,
    'Predict':y_pred
})
lm_result['Diff'] = y_test - y_pred
lm_result.head()


# In[32]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[33]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)


# In[34]:


rf_result = pd.DataFrame({
    'Actual':y_test,
    'Predict':y_pred
})
rf_result['Diff'] = y_test - y_pred
rf_result.head()


# # Conclusion

# #### Happiness score depends on mostly all of the features. All of these terms have a great linear relationship with happiness score.

# In[ ]:




