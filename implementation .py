#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np


# In[49]:


data=pd.read_csv('E:\phython software files/data_cleaned.csv')


# In[50]:


data.shape


# In[51]:


data.head()


# In[52]:


x=data.drop(['Survived'],axis=1)


# In[53]:


y=data['Survived']


# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


train_x,test_x,train_y,test_y=train_test_split(x,y,random_state=101,stratify=y)


# In[56]:


train_y.value_counts()/len(train_y)


# In[57]:


test_y.value_counts()/len(test_y)


# In[58]:


from sklearn.tree import DecisionTreeClassifier


# In[59]:


clf=DecisionTreeClassifier()


# In[60]:


clf.fit(train_x,train_y)


# In[61]:


clf.score(train_x,train_y)


# In[62]:


clf.score(test_x,test_y)


# In[18]:


from sklearn.model_selection import GridSearchCV
parameters={'criterion' : ('gini','entropy'), 'max_depth' : (3,4,5,6,7,8,9,10), 'min_samples_leaf' : (2,3,4,5,6,7,8,9),'min_samples_split' : (2,3,4,5,6,7),'max_features':('auto','sqrt','log2')}


# In[19]:


DataGrid=GridSearchCV(DecisionTreeClassifier(),param_grid=parameters,cv=3,verbose=True,n_jobs=1)


# In[20]:


Data_grid_model=DataGrid.fit(train_x,train_y)


# In[21]:


Data_grid_model.best_estimator_


# In[22]:


Data_grid_model.score(train_x,train_y)


# In[23]:


Data_grid_model.score(test_x,test_y)


# In[40]:


tree=DecisionTreeClassifier(criterion='entropy', max_depth=8, max_features='log2',
                       random_state=None,min_samples_leaf=7, min_samples_split=5)


# In[41]:


tree.fit(train_x,train_y)


# In[42]:


tree.score(train_x,train_y)


# In[43]:


tree.score(test_x,test_y)


# In[47]:


clf.predict(train_x)


# In[46]:


pred=clf.predict(test_x)
pred

