#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[15]:


df=pd.read_excel("Desktop\\Book1.xlsx")


# In[16]:


df.head()


# In[18]:


x=df.drop(['actualcomp'],axis=1)
y=df['actualcomp'].values.reshape(-1,1)


# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[20]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[23]:


from sklearn.linear_model import LinearRegression
multiple_reg = LinearRegression()
multiple_reg.fit(x_train, y_train)


# In[24]:


y_pred = multiple_reg.predict(x_test)


# In[29]:


print("Enter the input :")
actualH2O = float(input("ACTUALH2O : "))
temp = float(input("TEMP : "))
water = float(input("WATER : "))


#predicting the sales with respect to the inputs
output = multiple_reg.predict([[actualH2O,temp,water]])
print("you will get compactibility {}  by actual H2O of {} and temp of{} and water of {} ."     .format(output[0][0] if output else "not predictable",actualH2O,temp,water))


# In[ ]:




