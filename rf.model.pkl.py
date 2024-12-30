#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np



# In[3]:


# Load the UCI Heart Disease Dataset
df = pd.read_csv("UCI Heart Disease Dataset.csv")


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[19]:


import seaborn as sns
sns.countplot(x = 'target', data =df)


# In[134]:


import matplotlib.pyplot as plt
corr_mat = df.corr()
plt.figure(figsize = (15,15))
sns.heatmap(corr_mat , annot =True)


# In[36]:


#plot histograms for each column
df.hist(figsize=(12,12))


# In[28]:


#dataset is clean and has no missing values


# In[29]:


df['target'].value_counts()


# In[30]:


df.info()


# In[31]:


#all data are numerical


# In[51]:


from sklearn.model_selection import train_test_split

y = df['target']

x = df.drop('target',axis =1)


# In[66]:


y


# In[67]:


x


# In[56]:


from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test = train_test_split(x, y , test_size=0.25 , random_state = 42)


# In[57]:


x_train.shape


# In[68]:


x_test.shape


# In[69]:


y_train


# In[62]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[63]:


x_train


# In[91]:


pip install --upgrade scikit-learn numpy threadpoolctl


# In[93]:


get_ipython().system('pip uninstall scikit-learn -y')
get_ipython().system('pip install scikit-learn')



# In[100]:


#import ml library
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
forest =RandomForestClassifier(n_estimators=20, random_state=12,max_depth=6)
model = SVC(kernel='rbf',class_weight= 'balanced',probability=True,random_state=42)
lg = LogisticRegression(class_weight='balanced',random_state=42)
dt_model = DecisionTreeClassifier(max_depth=6, random_state=42)


# In[102]:


#train the model
forest.fit(x_train,y_train)
model.fit(x_train,y_train)
lg.fit(x_train,y_train)
dt_model.fit(x_train, y_train)


# In[110]:


# Test Random Forest
forest.fit(x_train, y_train)
y_pred_forest = forest.predict(x_train)
print("Random Forest predictions successful.")

# Test SVM
model.fit(x_train, y_train)
y_pred_svm = model.predict(x_train)
print("SVM predictions successful.")

# Test Logistic Regression
lg.fit(x_train, y_train)
y_pred_lg = lg.predict(x_train)
print("Logistic Regression predictions successful.")

# Decision Tree Predictions
y_pred_dt = dt_model.predict(x_train)
print("Decision Tree predictions successful")


# In[112]:


#training Accuracy
print(accuracy_score(y_train, y_pred_forest))
print(accuracy_score(y_train, y_pred_svm))
print(accuracy_score(y_train, y_pred_lg))
print(accuracy_score(y_train, y_pred_dt))


# In[114]:


y_pred_forest = forest.predict(x_test)
y_pred_svm = model.predict(x_test)
y_pred_lg = lg.predict(x_test)
y_pred_dt = dt_model.predict(x_test)


# In[115]:


#test Accuracy
print(accuracy_score(y_test, y_pred_forest))
print(accuracy_score(y_test, y_pred_svm))
print(accuracy_score(y_test, y_pred_lg))
print(accuracy_score(y_test, y_pred_dt))


# In[127]:


from sklearn.metrics import classification_report, roc_auc_score

# Random Forest
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_forest))
roc_auc_rf = roc_auc_score(y_test, forest.predict_proba(x_test)[:, 1])
print("Random Forest ROC-AUC:", roc_auc_rf)


# In[128]:


# SVM
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
roc_auc_svm = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
print("SVM ROC-AUC:", roc_auc_svm)


# In[129]:


# Logistic Regression
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lg))
roc_auc_lr = roc_auc_score(y_test, lg.predict_proba(x_test)[:, 1])
print("Logistic Regression ROC-AUC:", roc_auc_lr)


# In[130]:


# Decision Tree
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))
roc_auc_dt = roc_auc_score(y_test, dt_model.predict_proba(x_test)[:, 1])
print("Decision Tree ROC-AUC:", roc_auc_dt)


# In[120]:


#BEST MODEL IS RANDOM FOREST


# In[135]:


import pickle
pickle.dump(forest, open('Random_forest_model.pkl' ,'wb'))


# In[ ]:




