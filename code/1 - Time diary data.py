import pandas as pd
import matplotlib as plt
import numpy as np
import pickle
import seaborn as sns
import json

td = pd.read_stata("td_ita.dta")
td.info()

#I want to simplify the activity classification by reducing the total to a shorter list of categories.
#I did this separately, putting the results on an excel File:

activities_category.csv

activities_category = pd.read_csv('activities_category.csv', delimiter=';')

td['what'] = td['what'].replace(list(activities_category.What), list(activities_category.Category))

Company = list(td.withw.unique())

#For easiness, I recode also "with who", dividing it in only two categories: Alone, With Others.
company_2 = ['Alone', 'Alone', 'with_others','with_others','with_others','Alone','with_others',
            'with_others','with_others','with_others','Alone','Alone']
td['withw'] = td['withw'].replace(Company, company_2)

td['DataFrame Column'] = pd.to_datetime(td['datein_answ'], format='%Y-%m-%d %H:%M:%s')
td['hour'] = td['DataFrame Column'].dt.hour
td['minute'] = td['DataFrame Column'].dt.minute
td.drop(['DataFrame Column'],axis=1)

#Instead of considering all minutes, I'll just associate them with the corresponding half or the hour, all reduced to 0 or 30.
split = [30 if i['minute'] >=30 else 0 for index,i in td.iterrows()]
td['minute'] = split

#All others id are expressed as integer numbers (numpy.int32); thus, I use a new version with the same name.
users = td['id'].astype(int)
td['userid']= users

td['day'] = td.datein_answ.apply(lambda x: x.date())
td['day'] = td.day.apply(lambda x: x.strftime("%Y%m%d"))

activities = td[['userid','day','week','hour', 'minute', 'what','withw']]

activities.to_csv('activities.csv',index=False)



















