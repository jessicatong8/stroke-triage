import pandas as pd
import numpy as np

df_CSC = pd.read_csv("closest_comprehensive_hospitals.csv")
df_PSC = pd.read_csv("closest_primary_hospitals.csv")
transfer_times = pd.read_csv("hospital_to_hospital_distances.csv")

df_CSC = df_CSC.drop(columns="dest_type")
df_PSC = df_PSC.drop(columns="dest_type")

print(df_CSC.shape)
print(df_PSC.shape)


df_CSC = df_CSC.rename(columns={'travel_time_minutes': 'CSC_travel_time_minutes', 'dest_lat':'CSC_dest_lat', 'dest_lon':'CSC_dest_lon','distance_km':'CSC_distance_km','method':'CSC_method'})
df_PSC = df_PSC.rename(columns={'travel_time_minutes': 'PSC_travel_time_minutes', 'dest_lat':'PSC_dest_lat', 'dest_lon':'PSC_dest_lon','distance_km':'PSC_distance_km','method':'PSC_method'})

df = pd.merge(df_CSC, df_PSC, on=['origin_lat', 'origin_lon'], how='inner')
print(df.columns)
print(df.shape)

# merge in transfer times from closest PSC to closest CSC

# Round all lat/lon columns to 4 decimal places to avoid precision mismatch
df[['PSC_dest_lat', 'PSC_dest_lon', 'CSC_dest_lat', 'CSC_dest_lon']] = df[['PSC_dest_lat', 'PSC_dest_lon', 'CSC_dest_lat', 'CSC_dest_lon']].round(4)
transfer_times[['primary_lat', 'primary_lon', 'comprehensive_lat', 'comprehensive_lon']] = transfer_times[['primary_lat', 'primary_lon', 'comprehensive_lat', 'comprehensive_lon']].round(4)


df = pd.merge(
    df,
    transfer_times,
    left_on=['PSC_dest_lat', 'PSC_dest_lon', 'CSC_dest_lat', 'CSC_dest_lon'],
    right_on=['primary_lat', 'primary_lon', 'comprehensive_lat', 'comprehensive_lon'],
    how='left'
)

df = df.rename(columns={'travel_time_minutes': 'transfer_time_minutes', 'distance_km': 'transfer_distance_km'})

# drop redundant hospital lat long
df.drop(columns=['primary_lat', 'primary_lon', 'comprehensive_lat', 'comprehensive_lon'], inplace=True)
# drop from to columns
df.drop(columns=['From', 'To'], inplace=True)

print(df.head())
print(df.columns)
print(df.shape)


# Save to CSV
output_file = 'ems-strokes-traveltimes_with-transfer.csv'
df.to_csv(output_file, index=False)
print(f"Merged travel times saved to {output_file}")