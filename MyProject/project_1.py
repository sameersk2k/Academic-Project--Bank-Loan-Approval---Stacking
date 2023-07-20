#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib


# In[2]:


df=pd.read_csv('Bank_Data.csv')


# In[3]:


df.head()


# In[4]:


df


# In[5]:


df['Loan_Status'].value_counts()


# In[6]:


df.describe()


# In[7]:


df['CoapplicantIncome'].value_counts()


# In[8]:


df.info()


# In[9]:


# find the null values
df.isnull().sum()


# In[10]:


df['LoanAmount'].mean()


# In[11]:


# fill the missing values for numerical terms - mean/mode
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mode()[0])


# In[12]:


df['Gender'].mode()[0]


# In[13]:


# fill the missing values for categorical terms - mode
df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']=df['Married'].fillna(df['Married'].mode()[0])
df['Dependents']=df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])


# In[14]:


df.isnull().sum()


# ## Label Encoding

# In[15]:


from sklearn.preprocessing import LabelEncoder
cols=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
for x in cols:
    print(df[x].value_counts())


# In[16]:


df['Dependents'].value_counts()


# In[17]:


# replacing the value of 3+ to 4
df=df.replace(to_replace='3+',value='4')


# In[18]:


df['Dependents'].value_counts()


# In[19]:


#le=LabelEncoder()
#for col in cols:
    #df[col]=le.fit_transform(df[col])


# In[20]:


df['Dependents'].value_counts()


# In[21]:


df['Credit_History'].value_counts()


# In[22]:


df


# ## visualization

# In[23]:


#education and loan status
sns.countplot(x='Education',hue='Loan_Status',data=df)


# In[24]:


# marital status and loan_status
sns.countplot(x='Married',hue='Loan_Status',data=df)


# In[25]:


sns.countplot(x='Gender',hue='Loan_Status',data=df)


# In[26]:


sns.countplot(x='Dependents',hue='Loan_Status',data=df)


# In[27]:


sns.countplot(x='Self_Employed',hue='Loan_Status',data=df)


# In[28]:


sns.countplot(x='Credit_History',hue='Loan_Status',data=df)


# In[29]:


sns.displot(df['ApplicantIncome'])


# In[30]:


sns.displot(df['CoapplicantIncome'])


# In[ ]:





# In[ ]:





# In[31]:


le=LabelEncoder()
for col in cols:
    df[col]=le.fit_transform(df[col])


# In[32]:


df.head()


# In[33]:


X=df.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y=df[['Loan_Status']]


# In[34]:


X


# In[35]:


y


# ## Train Test Split

# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)


# In[37]:


print(X.shape,X_train.shape,X_test.shape)


# ## Training

# In[38]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,Y_train)


# ## Accuracy score on Training Data

# In[39]:


from sklearn.metrics import accuracy_score
yt_pred=model.predict(X_train)
accuracy_score(yt_pred,Y_train)#clearly overfitted


# In[40]:


from sklearn.tree import DecisionTreeClassifier
model2=DecisionTreeClassifier()
model2.fit(X_train,Y_train)


# In[41]:


yt_pred=model2.predict(X_train)
accuracy_score(yt_pred,Y_train)#clearly overfitted


# In[42]:


from sklearn.naive_bayes import GaussianNB
model3=GaussianNB()
model3.fit(X_train,Y_train)


# In[43]:


yt_pred=model3.predict(X_train)
accuracy_score(yt_pred,Y_train)# not 0verfitted


# In[44]:


from sklearn.ensemble import GradientBoostingClassifier
model4=GradientBoostingClassifier()
model4.fit(X_train,Y_train)
yt_pred=model4.predict(X_train)
accuracy_score(yt_pred,Y_train)# not 0verfitted


# ## test accuracy

# ## random forest

# In[45]:


y_pred=model.predict(X_test)
accuracy_score(y_pred,Y_test)


# ## decision tree

# In[46]:


y_pred2=model2.predict(X_test)
accuracy_score(y_pred2,Y_test)


# ## GaussianNB

# In[47]:


y_pred3=model3.predict(X_test)
accuracy_score(y_pred3,Y_test)


# In[48]:


y_pred4=model4.predict(X_test)
accuracy_score(y_pred4,Y_test)


# In[ ]:





# In[49]:


level0 = list()
level0.append(('rf',RandomForestClassifier()))
level0.append(('det', DecisionTreeClassifier()))
level0.append(('gnb', GaussianNB()))
level0.append(('gbc',GradientBoostingClassifier()))


# In[50]:


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
stack_model = StackingClassifier(estimators=level0, final_estimator = LogisticRegression(), cv=10)


# In[51]:


stack_model.fit(X_train,Y_train)


# ## testing accuracy

# In[52]:


ys_pred=stack_model.predict(X_test)
accuracy_score(ys_pred,Y_test)


# ## training accuracy

# In[53]:


yst_pred=stack_model.predict(X_train)
accuracy_score(yst_pred,Y_train)


# ## Implementing stacking Manually

# In[54]:


model=RandomForestClassifier(n_estimators=5)
model2=DecisionTreeClassifier()
model3=GaussianNB()
model4=GradientBoostingClassifier()


# In[55]:


X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)


# In[56]:


Y_train.value_counts()


# In[57]:


from sklearn.model_selection import StratifiedKFold
def Stacking(model,train,y,n_fold):
    folds=StratifiedKFold(n_splits=n_fold)
    train_pred=np.empty((0,1),int)
    for train_indices,val_indices in folds.split(train,y.values):
        x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
        print(x_train,x_val)
        y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]
        print(y_train,y_val)
        model.fit(x_train,y_train)
        train_pred=np.append(train_pred,model.predict(x_val))
        print(train_pred)

    return train_pred


# In[58]:


train_pred_1=Stacking(model=model,n_fold=5, train=X_train,y=Y_train)
train_pred_2=Stacking(model=model2,n_fold=5, train=X_train,y=Y_train)
train_pred_3=Stacking(model=model3,n_fold=5,  train=X_train,y=Y_train)
train_pred_4=Stacking(model=model4,n_fold=5,  train=X_train,y=Y_train)


# In[59]:


# convert into dataframe for later use
train_pred_1=pd.DataFrame(train_pred_1)
train_pred_2=pd.DataFrame(train_pred_2)
train_pred_3=pd.DataFrame(train_pred_3)
train_pred_4=pd.DataFrame(train_pred_4)


# In[60]:


train_data_meta = pd.concat([train_pred_1, train_pred_2,train_pred_3,train_pred_4], axis=1)
#train_data_meta = pd.concat([train_pred_1,train_pred_3,train_pred_4], axis=1)
train_data_meta


# In[61]:



meta_model = LogisticRegression()

meta_model.fit(train_data_meta,Y_train)


# In[62]:


model=RandomForestClassifier()
model2=DecisionTreeClassifier()
model3=GaussianNB()
model4=GradientBoostingClassifier()


# In[63]:


model.fit(X_train,Y_train)
model2.fit(X_train,Y_train)
model3.fit(X_train,Y_train)
model4.fit(X_train,Y_train)


# In[64]:


y1_pred=model.predict(X_test)
y2_pred=model2.predict(X_test)
y3_pred=model3.predict(X_test)
y4_pred=model4.predict(X_test)


# In[65]:


y1_pred=pd.DataFrame(y1_pred)
y2_pred=pd.DataFrame(y2_pred)
y3_pred=pd.DataFrame(y3_pred)
y4_pred=pd.DataFrame(y4_pred)


# In[66]:


test_res = pd.concat([y1_pred, y2_pred, y3_pred,y4_pred], axis=1)


# In[67]:


test_res


# In[68]:


meta_model.score(test_res,Y_test)


# In[71]:


import pickle


# In[72]:


pickle.dump(meta_model,open('StackModel.pkl','wb'))


# In[ ]:




