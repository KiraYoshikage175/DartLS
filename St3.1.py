#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
from pprint import pprint
import glob
import codecs
import json
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)




read_files = glob.glob("data/*.geojson")
output_list = []

for f in read_files:
    with open(f, "rb") as infile:
        output_list.append(json.load(infile))

with open("merged_file.json", "w") as outfile:
    json.dump(output_list, outfile)











with codecs.open('merged_file.json', 'r', 'utf-8') as json_file:  
    data = json.load(json_file)
    
df = pd.json_normalize(data, errors='ignore')
df






df = pd.json_normalize(data, record_path=['features', 'properties', 'vehicles', 'participants'], meta = [
    ['features', 'properties','id'],
    ['features', 'properties', 'tags'],
    ['features', 'properties', 'light'],
    ['features', 'properties', 'point'], #2 cols
    ['features', 'properties', 'nearby'],
    ['features', 'properties', 'region'],
    ['features', 'properties', 'address'],
    ['features', 'properties', 'weather'],
    ['features', 'properties', 'category'],
    ['features', 'properties', 'datetime'],
    ['features', 'properties', 'severity'],
    ['features', 'properties', 'vehicles', 'year'],
    ['features', 'properties', 'vehicles', 'brand'],
    ['features', 'properties', 'vehicles', 'color'],
    ['features', 'properties', 'vehicles', 'model'],
    ['features', 'properties', 'vehicles', 'category'],
    ['features', 'properties','dead_count'],
    ['features', 'properties','participants'],
    ['features', 'properties','injured_count'],
    ['features', 'properties','parent_region'],
    ['features', 'properties','road_conditions'],
    ['features', 'properties','participants_count'],
    ['features', 'properties','participant_categories'],
], errors='ignore')

df = pd.concat([df.drop('features.properties.point', axis=1), pd.DataFrame(df['features.properties.point'].tolist())], axis=1)
df













df1 =  (df.set_index('features.properties.id')['features.properties.participants']
       .apply(pd.Series).stack()
         .apply(pd.Series).reset_index().drop('level_1',1))

df = df.merge(df1, how='left', on='features.properties.id')





df=df.drop('features.properties.participants', axis=1)






df.isna().sum()







df=df.fillna(0)





df.shape





df.info()








df.isna().sum()







df = df.explode('violations_x')
df = df.explode('features.properties.tags')
df = df.explode('features.properties.nearby')
df = df.explode('features.properties.weather')
df = df.explode('features.properties.road_conditions')
df = df.explode('features.properties.participant_categories')



df




df=df.drop_duplicates(subset=['features.properties.id'])
df=df.fillna(0)




df=df[df['features.properties.address']!=0]
df.reset_index(drop=True, inplace=True)




result = pd.merge(df, df.groupby(['features.properties.address']).size().sort_values(ascending=False).to_frame(), on="features.properties.address")
result.rename(columns={0: 'count'},inplace=True)
#result.groupby(['properties.address']).size().sort_values(ascending=False).to_frame()
df = result







corr=df.drop(['features.properties.id'], axis=1).corr()
plt.figure(figsize=(16, 16))

heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=20)







df.isna().sum()





df['Hazard_level'] = None
count_places_max = df['count'].max()
injured_max = df['features.properties.injured_count'].max()
dead_max = df['features.properties.dead_count'].max()




for i in range(len(df)):
    if df['features.properties.dead_count'][i] > 0:
        df['Hazard_level'][i] = (df['features.properties.injured_count'][i]+df['count'][i])/((injured_max+count_places_max)/2)/4
    else:
        df['Hazard_level'][i] = (df['features.properties.dead_count'][i]*100/dead_max)/100/2+0.5




df.head()





df.to_csv('result_data.csv', encoding='utf-8-sig', index=False)


