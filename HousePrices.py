# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:47:14 2020

@author: Taha
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize']=(10,6)
plt.show()

train.SalePrice.describe()

print(train.SalePrice.skew())
plt.hist(train.SalePrice,color='blue')
plt.show()

target=np.log(train.SalePrice)
print(target.skew())
plt.hist(target,color='blue')
plt.show()

#Woiking with numeric features
numeric_features=train.select_dtypes(include=[np.number])
numeric_features.dtypes

#Finding the correlation of features with target
corr=numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])

train.OverallQual.unique()

quality_pivot=train.pivot_table(index='OverallQual',values='SalePrice',aggfunc=np.median)


quality_pivot.plot(kind='bar',color='blue')




plt.scatter(x=train['GarageArea'],y=target)
plt.show()

 
train=train[train['GarageArea']<1200]

nulls=pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns=['number']
nulls.index.name='Features'

#Dealing with Non-numeric features
categoricals=train.select_dtypes(exclude=[np.number])
categoricals.describe()

train['enc_street']=pd.get_dummies(train.Street,drop_first=True)
test['enc_street']=pd.get_dummies(test.Street,drop_first=True)
train['enc_street'].value_counts()
test['enc_street'].value_counts()

train['enc_uti']=pd.get_dummies(train.Street,drop_first=True)
test['enc_uti']=pd.get_dummies(test.Street,drop_first=True)
train['enc_uti'].value_counts()
test['enc_uti'].value_counts()



condition_pivot=train.pivot_table(index='SaleCondition',values='SalePrice')
condition_pivot.plot(kind='bar',color='blue')

condition_pivot=train.pivot_table(index='SaleType',values='SalePrice')
condition_pivot.plot(kind='bar',color='blue')


variables={True:1,False:0}
train['enc_condition']=(train['SaleCondition']=='Partial')
test['enc_condition']=(test['SaleCondition']=='Partial')
train['enc_condition']=(train['enc_condition'].map(variables))
test['enc_condition']=(test['enc_condition'].map(variables))

variables={True:1,False:0}
train['enc_type']=(train['SaleType']=='New')
test['enc_type']=(test['SaleType']=='New')
train['enc_type']=(train['enc_type'].map(variables))
test['enc_type']=(test['enc_type'].map(variables))
 
condition_pivot=train.pivot_table(index='enc_condition',values='SalePrice')
condition_pivot.plot(kind='bar',color='blue')

 
#Dealing with missing values
data=train.select_dtypes(include=[np.number]).interpolate().dropna() 

sum(data.isnull().sum()!=0)

y=np.log(train.SalePrice)
X=data.drop(['SalePrice','Id'],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                          X, y, random_state=42, test_size=.3)


#Begin Modelling
from sklearn import linear_model
reg=linear_model.LinearRegression()
model=reg.fit(X_train,y_train)

model.score(X_test,y_test)


predictions=model.predict(X_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,predictions)

actual_values=y_test
plt.scatter(predictions,actual_values,alpha=0.7,color='b')

for i in range(-2,3):
    alphaa=10**i
    rm=linear_model.Ridge(alpha=alphaa)
    ridge_model=rm.fit(X_train,y_train)
    pred=ridge_model.predict(X_test)
    plt.scatter(pred,y_test,alpha=.75,color='b')
    plt.title(alphaa)
    overlay='{}\n{}'.format(ridge_model.score(X_test,y_test),mean_squared_error(y_test,pred))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()

 