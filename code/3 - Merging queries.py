import pandas as pd
import sqlite3
import numpy as np

activities = pd.read_csv('activities.csv')
touch_screen = pd.read_csv('touch_screen_recoded.csv')

#I now begin merging data; should I want to include individual parameters for the statistical model, a merge from sociodemographic data and the final dataset on the desired traits will be sufficient.
merged_data = pd.merge(touch_screen, activities, left_on=['day', 'hour', 'minute', 'userid'], right_on=['day', 'hour', 'minute', 'userid'], how='inner')
merged_data = merged_data.drop_duplicates()
merged_data.to_csv('merged_data.csv',index= False)

conn = sqlite3.connect('project_sdl.db')
cur = conn.cursor()

merged_data.to_sql('merged1', conn)
touch_screen.to_sql('touch_screen_recoded', conn)
activities.to_sql('activities', conn)

%load_ext sql
%sql sqlite:///project_sdl.db

#Now I move on to sociodemographic data.
socio = pd.read_stata("sociopsicodemo ITA.dta")

socio['gender']=socio['w1_A01']
socio['age'] = socio['cohort']

individuals = socio[['userid','Extraversion','Openness','Conscientiousness','Neuroticism','Agreeableness', 'gender', 'department','age']]
convert_dict = {'userid': int}
individuals = individuals.astype(convert_dict)

individuals.to_csv('individuals.csv',index=False)

merged_data= pd.merge(merged_data, individuals, left_on=['userid'], right_on=['userid'], how='inner')

#Now, for each time diary observations we have the userid (and his traits), weekday, company situations, touch screen events recoded and timestamps.
#Everything we need is all stored in the same dataset, merged_data.
merged_data.to_csv('merged_data.csv', index=False)














