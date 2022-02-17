#!/usr/bin/env python
# coding: utf-8

# Name - Shraddha Khadepatil

# Student id - 100820094

# # Exploratory Data Analysis

# In[25]:


import pandas as pd
import numpy as np
from sklearn import datasets


# In[27]:


# Importing dataset
df = pd.read_csv('C:/Users/shrad/Documents/Durham College/Semester 1/AI Algorithm/Exercise1&2_100820094_ShraddhaK/dataset.csv')
df


# In[28]:


df.shape


# In[29]:


display(df)


# 

# In[31]:


# Checking the pairwise correlation with dataset
import seaborn as sns
corr = df.corr()
corr
corr.head()


# In[32]:


ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# # Identifying the Outliers

# In[33]:


df.describe()


# In[34]:


df = df.drop(columns=['id'])


# In[35]:


# % distribution of values outcome
(df["diagnosis"].value_counts()*100)/len(df)


# In[36]:


# Identifying the missing values
df.isnull().sum()


# In[37]:


def correlation(df,threshold):
    col_corr= [] # List of correlated columns
    corr_matrix=df.corr() #finding correlation between columns
    for i in range (len(corr_matrix.columns)): #Number of columns
        for j in range (i):
            if abs(corr_matrix.iloc[i,j])>threshold: #checking correlation between columns
                colName=(corr_matrix.columns[i], corr_matrix.columns[j]) #getting correlated columns
                col_corr.append(colName) #adding correlated column name
    return col_corr #returning set of column names
col=correlation(df,0.8)
print('Correlated columns @ 0.8:', col)


# In[38]:


# Converting the value of diagnosis from string to binary
event={'B':0,'M':1}
df.diagnosis=[event[item] for item in df.diagnosis]
display(df)


# In[39]:


#Plotting Outliers
import seaborn as sns
sns.boxplot(data=df)


# In[40]:


X = df.drop(columns=['diagnosis']).values
y = df['diagnosis'].values


# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[42]:


print('Shape of X_train=>',X_train.shape)
print('Shape of X_test=>',X_test.shape)
print('Shape of Y_train=>',y_train.shape)
print('Shape of Y_test=>',y_test.shape)


# # Performing Cross Validation

# First using Decision Trees to check the compatibilty

# In[43]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dt.fit(X_train, y_train)


# In[44]:


# Evaluating the training set
from sklearn.metrics import f1_score
dt_pred_train = dt.predict(X_train)

print('Training Set Evaluation F1-Score=>',f1_score(y_train,dt_pred_train))


# In[45]:


dt_pred_test = dt.predict(X_test)
print('Testing Set Evaluation F1-Score=>',f1_score(y_test,dt_pred_test))


# Now using Random Forest Classifier to check the compatibility

# In[46]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, random_state = 42)
rfc.fit(X_train, y_train)

# Evaluating the Training set
rfc_pred_train = rfc.predict(X_train)
print('Training Set Evaluation F1-Score=>',f1_score(y_train,rfc_pred_train))


# In[50]:


# Evaluating on Test set
rfc_pred_test = rfc.predict(X_test)
print('Testing Set Evaluation F1-Score=>',f1_score(y_test,rfc_pred_test))


# As per the above cross-validation, we can conclude that the random forest classifier is better than Decision trees.
# Therefore, Random Forest Classifier is the better algorithm to be used for this dataset.

# In[52]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

models = []

models.append(('KNN', KNeighborsClassifier()))
models.append(('DT',  DecisionTreeClassifier()))
models.append(('RF',  RandomForestClassifier(n_estimators=100)))

# Train/Test split
X_train_cross, X_test_cross, y_train_cross, y_test_cross = train_test_split(X, y, stratify = df.diagnosis, random_state=0)

names = []
scores = []

for name, model in models:
    model.fit(X_train_cross, y_train_cross)
    y_pred_cross = model.predict(X_test_cross)
    scores.append(accuracy_score(y_test_cross, y_pred_cross))
    names.append(name)

tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)


# # Confusion Matrix and Metrics

# In[53]:


# Train/Test split
X_train_matrix, X_test_matrix, y_train_matrix, y_test_matrix = train_test_split(X, y, stratify = df.diagnosis, random_state=0)


# In[54]:


KNN_MATRIX = KNeighborsClassifier(n_neighbors=9)
KNN_MATRIX.fit(X_train_matrix,y_train_matrix)
y_pred_knn = KNN_MATRIX.predict(X_test_matrix) 


# In[55]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_matrix, y_pred_knn)


# In[56]:


# Accuracy - OK
from sklearn.metrics import accuracy_score
accuracy_score(y_test_matrix, y_pred_knn)


# In[57]:


# F1 Score - OK
from sklearn.metrics import f1_score
f1_score(y_test_matrix, y_pred_knn, average=None)


# In[ ]:




