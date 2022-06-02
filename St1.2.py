#!/usr/bin/env python
# coding: utf-8






import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')









df = pd.read_csv("c1_result.csv")


df.head()



df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df.info()









df["year"] = df["pickup_datetime"].apply(lambda x : x.year)
df["month"] = df["pickup_datetime"].apply(lambda x: x.month)
df["dayofweek"] = df["pickup_datetime"].apply(lambda x: x.dayofweek)
df["hour"] = df["pickup_datetime"].apply(lambda x : x.hour)




df = df.drop("pickup_datetime", axis = 1)



df.head()








plt.figure(figsize = (20, 10))
sns.heatmap(df.corr(), annot = True)




import numpy as np



plt.figure(figsize = (10, 10))
sns.pointplot(y=np.sort(df["trip_duration"]), x = np.sort(df["passenger_count"]))




sns.lineplot(y = np.sort(df["trip_duration"]), x = np.sort(df["maximum temperature"]))




for pr in ["month", "year", "dayofweek", "hour"]:
    sns.scatterplot(y = np.sort(df["trip_duration"]), x = np.sort(df[pr]))
    plt.title(pr)
    plt.show()




X = df.drop("trip_duration", axis = 1) 
y = df["trip_duration"].array



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True)



from sklearn.metrics import r2_score, mean_absolute_error
def score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"R^2 : {r2}")
    print("-" * 20)
    print(f"MAE: {mae}")
    print("-" * 20)
    print()






from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression










rfc = RandomForestRegressor(verbose=3, n_jobs=-1)



# Lir
rfc.fit(X_train, y_train)
score(y_test, rfc.predict(X_test))




rfc.score(X_test, y_test)




grb = GradientBoostingRegressor(verbose=3)
grb.fit(X_train, y_train)
score(y_test, grb.predict(X_test))




grb.score(X_test, y_test)





lr = LinearRegression()
lr.fit(X_train, y_train)
score(y_test, lr.predict(X_test))




lr.score(X_test, y_test)



