import pandas as pd
import numpy as np

df_CSC = pd.read_csv("closest_comprehensive_hospitals.csv")
df_PSC = pd.read_csv("closest_primary_hospitals.csv")

df_CSC = df_CSC.rename(columns={'travel_time_minutes': 'CSC_travel_time_minutes'})
df_PSC= df_PSC.rename(columns={'travel_time_minutes': 'PSC_travel_time_minutes'})

merged_df = pd.merge(df_CSC, df_PSC, on=['origin_lat', 'origin_lon'], how='inner')

print(merged_df.head())