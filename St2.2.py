#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np




df=pd.read_csv('result_data.csv')





df







df.replace([np.inf, -np.inf], np.nan, inplace=True)




df=df[df['Rt']<5].reset_index(drop=True)







df1=df[df['Rt']<=0.7]
df1['Danger']=0





df2=df[(df['Rt']>0.7) & (df['Rt']<=0.95)]
df2['Danger']=1





df3=df[df['Rt']>0.95]
df3['Danger']=2





df=pd.concat([df1, df2, df3]).reset_index(drop=True)




X=df[['new_cases', 'new_deaths', 'Rt']]
y=df['Danger']



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)








from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB





neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
preds=neigh.predict(X_test)
print(classification_report(preds, y_test))






rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_preds=rfc.predict(X_test)
print(classification_report(rfc_preds, y_test))





gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_preds=gnb.predict(X_test)
print(classification_report(gnb_preds, y_test))









df.to_csv('result_data.csv', encoding='utf-8-sig', index=False)

