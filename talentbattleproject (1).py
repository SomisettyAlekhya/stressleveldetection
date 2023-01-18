#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


df=pd.read_csv("C:/Users/cherr/Documents/btech certificates/resume/hhh.csv" ,encoding="latin-1")


# In[3]:


df.head()


# In[4]:


df.head(n=10)


# In[5]:


df.shape


# In[6]:


np.unique(df['class'])


# In[7]:


np.unique(df['message'])


# In[8]:


x=df['message'].values
y=df['class'].values
cv=CountVectorizer()
x=cv.fit_transform(x)
v=x.toarray()
print(v)


# In[9]:


first_col=df.pop('message')
df.insert(0,'message',first_col)
df


# In[10]:


trainx=x[:4180]
trainy=y[:4180]
testx=x[4180:]
testy=y[4180:]


# In[11]:


bnb=BernoulliNB(binarize=0.0)
model=bnb.fit(trainx,trainy)
ypredtrain=bnb.predict(trainx)
ypredtest=bnb.predict(testx)


# In[12]:


print(bnb.score(trainx,trainy)*100)
print(bnb.score(testx,testy)*100)


# In[13]:


from sklearn.metrics import classification_report
print(classification_report(trainy,ypredtrain))


# In[14]:


from sklearn.metrics import classification_report
print(classification_report(testy,ypredtest))


# In[15]:


import numpy as np
import pandas as pd


# In[16]:


df=pd.read_csv('C:/Users/cherr/Documents/btech certificates/resume/hhh.csv',encoding="latin-1")
df.head()


# In[17]:


df.describe()


# In[18]:


df=pd.read_csv('C:/Users/cherr/Documents/btech certificates/resume/stress.csv')
df.head()


# In[19]:


df.describe()


# In[20]:


df.isnull()


# In[21]:


df.isnull().sum()


# In[22]:


import nltk
import re
from nltk. corpus import stopwords
import string
nltk. download( 'stopwords' )
stemmer = nltk. SnowballStemmer("english")
stopword=set (stopwords . words ( 'english' ))


# In[24]:



def clean(text):
    text = str(text) . lower()  #returns a string where all characters are lower case. Symbols and Numbers are ignored.
    text = re. sub('\[.*?\]',' ',text)  #substring and returns a string with replaced values.
    text = re. sub('https?://\S+/www\. \S+', ' ', text)#whitespace char with pattern
    text = re. sub('<. *?>+', ' ', text)#special char enclosed in square brackets
    text = re. sub(' [%s]' % re. escape(string. punctuation), ' ', text)#eliminate punctuation from string
    text = re. sub(' \n',' ', text)
    text = re. sub(' \w*\d\w*' ,' ', text)#word character ASCII punctuation
    text = [word for word in text. split(' ') if word not in stopword]  #removing stopwords
    text =" ". join(text)
    text = [stemmer . stem(word) for word in text. split(' ') ]#remove morphological affixes from words
    text = " ". join(text)
    return text
df [ "text"] = df["text"]. apply(clean)


# In[25]:


from sklearn. feature_extraction. text import CountVectorizer
from sklearn. model_selection import train_test_split

x = np.array (df["text"])
y = np.array (df["label"])

cv = CountVectorizer ()
X = cv. fit_transform(x)
print(X)
xtrain, xtest, ytrain, ytest = train_test_split(X, y,test_size=0.33)


# In[26]:


from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB()
model.fit(xtrain,ytrain)


# In[29]:


user=input("Enter the text")
data=cv.transform([user]).toarray()
output=model.predict(data)
print(output)


# In[ ]:




