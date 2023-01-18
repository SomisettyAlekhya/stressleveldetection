#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer


# In[5]:


df=pd.read_csv("C:/Users/cherr/Documents/btech certificates/resume/hhh.csv" ,encoding="latin-1")


# In[9]:


df.head()


# In[10]:


df.head(n=10)


# In[12]:


df.shape


# In[13]:


np.unique(df['class'])


# In[14]:


np.unique(df['message'])


# In[8]:


x=df['message'].values
y=df['class'].values
cv=CountVectorizer()
x=cv.fit_transform(x)
v=x.toarray()
print(v)


# In[6]:


first_col=df.pop('message')
df.insert(0,'message',first_col)
df


# In[9]:


trainx=x[:4180]
trainy=y[:4180]
testx=x[4180:]
testy=y[4180:]


# In[11]:


bnb=BernoulliNB(binarize=0.0)
model=bnb.fit(trainx,trainy)
ypredtrain=bnb.predict(trainx)
ypredtest=bnb.predict(testx)


# In[13]:


print(bnb.score(trainx,trainy)*100)
print(bnb.score(testx,testy)*100)


# In[16]:


from sklearn.metrics import classification_report
print(classification_report(trainy,ypredtrain))


# In[17]:


from sklearn.metrics import classification_report
print(classification_report(testy,ypredtest))


# In[18]:


import numpy as np
import pandas as pd


# In[20]:


df=pd.read_csv('C:/Users/cherr/Documents/btech certificates/resume/hhh.csv',encoding="latin-1")
df.head()


# In[21]:


df.describe()


# In[22]:


df=pd.read_csv('C:/Users/cherr/Documents/btech certificates/resume/stress.csv')
df.head()


# In[23]:


df.describe()


# In[24]:


df.isnull()


# In[25]:


df.isnull().sum()


# In[ ]:




