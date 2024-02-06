import pandas as pd
import matplotlib as plt
import numpy as np
import pickle
import seaborn as sns
import sqlite3

touch_screen = pd.read_csv('touchevent.csv')
touch_screen["timestamp"] = touch_screen["timestamp"].map(lambda x: str(x)[:-3]) 
touch_screen = touch_screen.drop_duplicates()

touch_screen['DataFrame Column'] = pd.to_datetime(touch_screen['timestamp'], format='%Y%m%d%H%M%S%f')
touch_screen = touch_screen.sort_values(by=["userid"]).reset_index(drop=True)

#Before proceeding with SQL, I add the time columns, in order to allow grouping observations by id and minute.
#Just like for activities, I will simplify minutes by only reducing them to 0 and 30.
touch_screen['hour'] = touch_screen['DataFrame Column'].dt.hour
touch_screen['minute'] = touch_screen['DataFrame Column'].dt.minute

split = [30 if i['minute'] >=30 else 0 for index,i in touch_screen.iterrows()]
touch_screen['minute'] = split
touch_screen = touch_screen.drop(['DataFrame Column', 'experimentid','timestamp'],axis=1)

touch_screen.to_csv('touch_screen.csv', index=False)

# I create a connector for SQL Queries.
conn = sqlite3.connect('project_sdl.db')
cur = conn.cursor()
touch_screen.to_sql('touch', conn)

#Now I proceed with merging queries by the minute, counting all corresponding observations.
%load_ext sql
%sql sqlite:///project_sdl.db

%sql
cur.execute("SELECT userid, day, hour, minute, COUNT(minute) FROM touch t GROUP BY t.day, t.hour, t.minute, t.userid")
rows = cur.fetchall()

touch_group = pd.DataFrame(rows, columns =['userid', 'day', 'hour', 'minute','touch_screen_events'])
touch_group.to_csv('touch_screen_recoded.csv', index=False)






























