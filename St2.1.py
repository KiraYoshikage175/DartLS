#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt






df=pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')












df.head()




df['date']=pd.to_datetime(df['date'])





df.info()





pd.set_option('display.max_rows',None)
df.isnull().sum()




pd.set_option('display.max_rows',10)



df.shape





df[['location', 'new_cases', 'new_deaths']]=df[['location', 'new_cases', 'new_deaths']].fillna(0)



grouped_cases=df[['location', 
                  'new_cases', 
                  'new_deaths']].groupby(by="location").mean().rename(columns={'new_cases':'mean_new_cases', 
                                                                               'new_deaths':'mean_new_deaths'})





grouped_cases





df=df.merge(grouped_cases, on='location')





df.head()





df=df.fillna(0)






corr=df.corr()
plt.figure(figsize=(70, 70))

heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=20)





df.info()








pd.set_option('display.max_rows',None)
df.isnull().sum()






pd.set_option('display.max_rows',10)







df.head()





plt.figure(figsize=(10, 5))
sns.kdeplot(df['iso_code'].value_counts())
plt.title('Distribution iso_code')
plt.xlabel('Значения')
plt.ylabel('Распределение')
plt.show()


plt.figure(figsize=(10, 5))
sns.kdeplot(df['continent'].value_counts())
plt.title('Distribution continent')
plt.xlabel('Значения')
plt.ylabel('Распределение')
plt.show()


plt.figure(figsize=(10, 5))
sns.kdeplot(df['location'].value_counts())
plt.title('Distribution location')
plt.xlabel('Значение')
plt.ylabel('Распределение')
plt.show()




def plot(column):
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df[column])
    plt.title('Distribution '+column)
    plt.xlabel('Значения')
    plt.ylabel('Распределение')
    plt.show()









for column in df[:100].select_dtypes(exclude=['object']).columns:
    plot(column)







df['Rt']=None
data=pd.DataFrame()
for country in df['location'].value_counts().keys():
    r=df[df['location']==country].copy()
    da=pd.DataFrame()
    for i in range(0, len(r), 8):
        tida=pd.DataFrame()
        su=r['new_cases'].tail(8).tail(4).sum()/r['new_cases'].tail(8).head(4).sum()
        tida=r.tail(8)
        tida['Rt']=su
        r.drop(r.tail(8).index,inplace=True)
        da=da.append(tida)
    data=data.append(da)





data=data.fillna(0)





data.reset_index(drop=True, inplace=True)
df=data




df.head()

















d=pd.DataFrame({'Russia': [list(df[df['location']=='Russia']['Rt'])[0]], 
                'Mexico':[list(df[df['location']=='Mexico']['Rt'])[0]], 
                'France': [list(df[df['location']=='France']['Rt'])[0]], 
                'Taiwan':[list(df[df['location']=='Taiwan']['Rt'])[0]], 
                'United States':[list(df[df['location']=='United States']['Rt'])[0]], 
                'Japan':[list(df[df['location']=='Japan']['Rt'])[0]], 
                'Canada':[list(df[df['location']=='Canada']['Rt'])[0]], 
                'Singapore':[list(df[df['location']=='Singapore']['Rt'])[0]],}).T






plt.rcParams.update({'font.size': 15,})
plt.figure(figsize=(15, 8))
plots = sns.barplot(x=d.index, y=d[0], data=df)

for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')

plt.title('Анализ эпидемиологической обстановки')
plt.ylabel('Rt - значение')
plt.xlabel('Страна')
plt.show()








df.to_csv('result_data.csv', encoding='utf-8-sig', index=False)

