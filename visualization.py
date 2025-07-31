import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

import joblib
import xgboost as xgb

# Load and clean data
hospitals = pd.read_csv("EMS-data/hospitals.csv")
hospitals = hospitals.rename(columns={
    "Y": "latitude",
    "X": "longitude"
})

# Set up input dataframe
df = pd.read_csv("EMS-data/ems-strokes-traveltimes_with-transfer.csv")
df.rename(columns = {'origin_lat':'latitude', 'origin_lon':'longitude'}, inplace=True)
df.rename(columns = {'PSC_travel_time_minutes':'time_to_primary','CSC_travel_time_minutes':'time_to_comprehensive', 'transfer_time_minutes':'transfer_time'}, inplace=True)

# hard coding sex and age for now
df['sex'] = 1 #female
df['age'] = 70 # majority of strokes occur in people aged 65 and older

# User inputs
RACE = st.slider("Select a RACE score", min_value=0, max_value=9, value=5, step=1)
LKW = st.slider("Select time since LKW (minutes)", min_value=10, max_value=270, value=60, step=5) #270 min = 4.5 hrs

df['RACE'] = RACE
df['time_since_symptoms'] = LKW



# Modeling using XGBoost trained on the simulation

# Load model
filename = 'xgboost_model.pkl'
model = joblib.load(filename)

# DataFrame for prediction
feature_list = ['sex','age','RACE','time_since_symptoms','time_to_primary','time_to_comprehensive','transfer_time']
input_df = df[feature_list]

# Predict using XGBoost model
prediction = model.predict(input_df)


df['Percent Comprehensive'] = prediction
df['Percent Comprehensive'] = df['Percent Comprehensive'].clip(lower=0, upper=100)
df['Percent Drip and Ship'] = 100 - df['Percent Comprehensive']

#display

# st.dataframe(df[['sex','age','RACE','time_since_symptoms','time_to_primary','time_to_comprehensive','transfer_time','Percent Comprehensive','Percent Drip and Ship']])



# MAPPING

# calculate color for strokes by how close to 50% the transport decision is (decision confidence)

def get_color(row):
    comp = row['Percent Comprehensive']
    drip = row['Percent Drip and Ship']

    # Use percent comprehensive as the decision score (between 0 and 100)
    score = comp

    if score <= 50:
        # Interpolate between blue and purple
        ratio = score / 50
        color_start = np.array([0, 195, 255])     # Blue
        color_end = np.array([200, 0, 255])     # Purple
    else:
        # Interpolate between purple and red
        ratio = (score - 50) / 50
        color_start = np.array([200, 0, 255])   # Purple
        color_end = np.array([255, 0, 0])       # Red

    final_color = (1 - ratio) * color_start + ratio * color_end
    final_color = final_color.astype(int)

    return f'#{final_color[0]:02x}{final_color[1]:02x}{final_color[2]:02x}'


# Apply color to simulated strokes
df['hex_color'] = df.apply(get_color, axis=1)



#color for hospital

# Default hospital color is blue
hospitals['hex_color'] = '#0000FF'  

# Set red color for CSCs
hospitals.loc[hospitals['Type'] == 'Comprehensive', 'hex_color'] = '#FF0000'  


# --- Create Folium Map ---
# Center map on mean location of simulated_triage or fallback
if not df.empty:
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
else:
    center_lat, center_lon = 40.3679, -79.9819  # default center

m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

# Create feature groups for toggling layers
triage_layer = folium.FeatureGroup(name='Simulated Stroke Patients')
heatmap_layer = folium.FeatureGroup(name='Heatmap')
hospital_layer = folium.FeatureGroup(name='Hospitals')

# Add simulated triage points as colored circles
for _, row in df.iterrows():
    try:
        lat = row['latitude']
        lon = row['longitude']
        comp = row['Percent Comprehensive']
        drip = row['Percent Drip and Ship']
        CSC_time = row['time_to_comprehensive']
        PSC_time = row['time_to_primary']
        transfer_time = row['transfer_time']
        hex_color = row['hex_color']
        popup = f"Comprehensive: {comp:.1f}%<br>Drip and ship: {drip:.1f}%<br>CSC travel time: {CSC_time:.0f} min<br>PSC travel time: {PSC_time:.0f} min"

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.7,
            popup = popup
            
        ).add_to(triage_layer)
        
    except KeyError as e:
        print(f"Skipping row due to missing column: {e}")
    except Exception as e:
        print(f"Skipping row due to unexpected error: {e}")

for _, row in hospitals.iterrows():
    color = ""
    if row['hex_color'] == '#0000FF':
        color = "blue"
    else:
        color = "red"
    folium.Marker(
    location=[row['latitude'], row['longitude']],
    icon=folium.Icon(icon="plus-square", prefix="fa", color=color),
    popup=row.get('Facility', 'Hospital'),
).add_to(hospital_layer)
    

# heat_data = [
#     [row['latitude'], row['longitude'], (row['Percent Comprehensive'] - 50) / 50]
#     for _, row in simulated_triage.iterrows()
#     if not np.isnan(row['Percent Comprehensive'])
# ]

# Add heatmap to triage_layer
# HeatMap(heat_data, min_opacity=0.3, radius=12, blur=15, max_zoom=13, max_val=1).add_to(heatmap_layer)


# Add layers to map
hospital_layer.add_to(m)
triage_layer.add_to(m)
# heatmap_layer.add_to(m)

# Add layer control (checkboxes for layers)
folium.LayerControl().add_to(m)


# Show map in Streamlit
st.set_page_config(layout="wide")

st_folium(m, width=1000, height=800)