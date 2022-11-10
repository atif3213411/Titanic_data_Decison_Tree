#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[5]:


df = pd.read_csv("train.csv")


# In[6]:


df.head()


# In[30]:


df.drop(['PassengerId' , 'Name' , 'SibSp' , 'Parch' , 'Ticket' , 'Cabin' , 'Embarked'] , axis = "columns" , inplace = True)
##inplace == true is used otherwise model had problems while training train data 
##The inplace parameter enables you to modify your dataframe directly. 
##Remember: by default, the drop() method produces a new dataframe and leaves the original dataframe unchanged.
##That's because by default, the inplace parameter is set to inplace = False 


# In[31]:


df.head()


# In[32]:


inputs = df.drop(['Survived'] , axis = "columns")


# In[33]:


target = df.Survived


# We have created the input for the data set
# 
# 

# In[34]:


##We try and map male to 1 and female to 0 
inputs.Age


# In[35]:


##THere can be seen ambigious age , so we need to remove it , by using mean value of ages 


# In[36]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())


# In[37]:


inputs.head()


# In[38]:


inputs.Sex = inputs.Sex.map({'male':1 , 'female':0})


# In[39]:


inputs.head()


# In[40]:


##Now we have cleared our data from all the ambigious inputs 


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


X_train , X_test , y_train , y_test = train_test_split(inputs , target , test_size = 0.2)


# In[43]:


len(X_train)


# In[44]:


len(X_test)


# In[45]:


from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[46]:


##We are gonna import decision tree


# In[48]:


model.fit(X_train , y_train)


# In[49]:


model.score(X_test , y_test)


# In[ ]:




