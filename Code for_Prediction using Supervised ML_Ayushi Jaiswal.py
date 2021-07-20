#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


url = "http://bit.ly/w-data"
student_data = pd.read_csv(url)
print("Data imported successfully")
student_data


# In[3]:


student_data.shape


# In[4]:


student_data.describe()


# In[5]:


student_data.isnull().sum()


# In[6]:


student_data.plot(x='Hours',y='Scores',style='ro')
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.show


# In[7]:


x = student_data.iloc[:, :-1].values
y = student_data.iloc[:, 1].values


# In[8]:


x


# In[9]:


y


# In[10]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[11]:


x_train


# In[12]:


x_test


# In[13]:


y_train


# In[14]:


y_test


# In[15]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

print("Training complete.")


# In[16]:


line = regressor.coef_*x+regressor.intercept_
plt.scatter(x,y)
plt.plot(x,line,color='green');
plt.show()


# In[17]:


print(x_test)
y_pred = regressor.predict(x_test)


# In[18]:


df = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df


# In[19]:


hours = 9.25
own_pred = regressor.predict([[hours]])
print(f"No of hours = {hours}")
print(f"Predicted score = {own_pred[0]}")


# In[20]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))


# In[21]:


print('Max Error:', metrics.max_error(y_test,y_pred))


# In[22]:


print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))


# In[ ]:




