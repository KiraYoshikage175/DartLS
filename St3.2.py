#!/usr/bin/env python
# coding: utf-8



import pandas as pd
pd.set_option('display.max_columns', None)




df=pd.read_csv('result_data.csv')





df.head()













X=df[['features.properties.dead_count', 'features.properties.injured_count', 'features.properties.participants_count']]
y=df['features.properties.severity']
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





result = pd.merge(df, df.groupby(['features.properties.vehicles.brand']).size().sort_values().to_frame(), on='features.properties.vehicles.brand')
result.rename(columns={0: 'brand_count'}, inplace=True)
df = result





from sklearn.preprocessing import StandardScaler






scaler = StandardScaler()
X=df[['features.properties.dead_count', 'features.properties.injured_count', 'features.properties.participants_count', 'brand_count']]
y=df['features.properties.severity']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_preds=rfc.predict(X_test)
print(classification_report(rfc_preds, y_test))



df.to_csv('result_data.csv', encoding='utf-8-sig', index=False)

