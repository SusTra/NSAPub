from partial_correlation import partial_corr 
import datetime
import seaborn as sns  
import pandas as pd
import numpy as np  
import os   
import matplotlib.pyplot as plt


def to_date_time(dt): 
	return datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S.%f')  

weeks = ["Podatki_o_prometu_v_Ljubljani_5-30_oktober2020", "Podatki_o_prometu_v_Ljubljani_9-11_september2020"]  
data_path = os.path.join(".", "PROMET_PODATKI", "Googl_Traffic", "koda_travel_times+vreme", "traffic_data_acquisition")  


df = pd.DataFrame() 
for week in weeks: 
    csv_file = os.path.join(data_path, week, week+".csv")
    df_file = pd.read_csv (csv_file)
    df = df.append(df_file, sort=False)

df["timestamp"] = df["timestamp"].apply(to_date_time)  
df['weekday'] = df['timestamp'].apply(lambda x: x.weekday())  
df['time'] = df['timestamp'].apply(lambda x: x.hour*3600 + x.minute*60 + x.second) 

df.drop('timestamp', inplace=True, axis=1) 
df.drop('Unnamed: 0', inplace=True, axis=1) 
df.drop('w_city', inplace=True, axis=1) 
df.drop('w_pressure', inplace=True, axis=1) 
df.drop('w_visibility', inplace=True, axis=1)   
df.drop('w_wind_degree', inplace=True, axis=1)     
df.drop('w_clouds', inplace=True, axis=1)      

df["w_description"] = df["w_description"].astype('category')
dw_description = dict(enumerate(df["w_description"].cat.categories))  
df["w_description"] = df["w_description"].cat.codes       

df = df.dropna()   
  
print(df)  
print(list(df.columns.values))   

data_as_array = df.values 
partial_corr_array = partial_corr(np.hstack((np.ones((data_as_array.shape[0],1)), data_as_array)))[1:,1:]
corr_df = pd.DataFrame(partial_corr_array, columns = df.columns)
print(corr_df)  

corr_df.rename(index = {k: v for k, v in enumerate(df.columns)}, inplace=True)    
sns.heatmap(corr_df)
plt.show()    