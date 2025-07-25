import streamlit as st
import pandas as pd
import numpy as np
import os
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
from scipy.interpolate import griddata

# --- Load CSV files ---
hospitals = pd.read_csv("EMS-data/hospitals.csv")

lower_RACE = st.checkbox("Lower RACE")
diff_RACE = st.checkbox("Difference RACE (RACE 1 - RACE 9)")
# lower_LKW = st.checkbox("lower_LKW")


# Load appropriate CSV based on checkbox state
if lower_RACE:
    df =  pd.read_csv("output/RACE1.csv")
else:
    df =  pd.read_csv("output/RACE9.csv")

# # Load appropriate CSV based on checkbox state
# if lower_LKW:
#     df =  pd.read_csv("output/LKW30.csv")
# else:
#     df =  pd.read_csv("output/LKW210.csv")


def get_difference(df1, df2, comparison_descr):
    df = df1.copy(deep=True) 
    df.drop(columns=['sex','age','RACE','time_since_symptoms'])
    df['Comparison Descr'] = comparison_descr
    df['Difference'] = df1['Percent Comprehensive'] - df2['Percent Comprehensive']
    # df['Percent Drip and Ship'] = 100 - df['Percent Comprehensive']
    df['Outcome Changed'] = 0
    df.loc[(df1['Percent Comprehensive'] > 50) & (df2['Percent Comprehensive'] < 50), 'Outcome Changed'] = 100
    df.loc[(df1['Percent Comprehensive'] < 50) & (df2['Percent Comprehensive'] > 50), 'Outcome Changed'] = -100
    return df

if diff_RACE:
    df1 = pd.read_csv("output/RACE1.csv")
    df2 = pd.read_csv("output/RACE9.csv")
    df = get_difference(df1, df2, "RACE 1 - RACE 9")

simulated_triage = df

# confirmed that map does update
# if lower_RACE:
#     simulated_triage['Percent Comprehensive'] = 0

# --- Data cleaning ---

hospitals = hospitals.rename(columns={
    "Y": "latitude",
    "X": "longitude"
})

simulated_triage = simulated_triage.rename(columns = {'origin_lat':'latitude', 'origin_lon':'longitude'})
# print(simulated_triage.head(5))
# print(simulated_triage.columns)


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


def get_color_diff(row):
    value = row['Outcome Changed']
    max_abs = 20  # adjust this if your data range differs

    # Normalize the value to range [-1, 1]
    norm = max(min(value / max_abs, 1), -1)

    if norm < 0:
        # Interpolate from white to red
        ratio = norm
        color_start = np.array([255, 255, 255])  # White
        color_end = np.array([255, 50, 50])      # Red (warm tone)
    else:
         # Interpolate from white to blue
        ratio = abs(norm)
        color_start = np.array([255, 255, 255])  # White
        color_end = np.array([0, 120, 255])      # Blue (cool tone)

    final_color = (1 - ratio) * color_start + ratio * color_end
    final_color = final_color.astype(int)

    return f'#{final_color[0]:02x}{final_color[1]:02x}{final_color[2]:02x}'


# Apply color to simulated strokes
if not diff_RACE:
    simulated_triage['hex_color'] = simulated_triage.apply(get_color, axis=1)
else:
    simulated_triage['hex_color'] = simulated_triage.apply(get_color_diff, axis=1)


#color for hospital

# Default hospital color is blue
hospitals['hex_color'] = '#0000FF'  

# Set red color for CSCs
hospitals.loc[hospitals['Type'] == 'Comprehensive', 'hex_color'] = '#FF0000'  

if lower_RACE:
    "Scatterplot RACE: 1"
else:
    "Scatterplot RACE: 9"



# --- Create Folium Map ---
# Center map on mean location of simulated_triage or fallback
if not simulated_triage.empty:
    center_lat = simulated_triage['latitude'].mean()
    center_lon = simulated_triage['longitude'].mean()
else:
    center_lat, center_lon = 40.3679, -79.9819  # default center

m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

# Create feature groups for toggling layers
triage_layer = folium.FeatureGroup(name='Simulated Stroke Patients')
heatmap_layer = folium.FeatureGroup(name='Heatmap')
hospital_layer = folium.FeatureGroup(name='Hospitals')

# Add simulated triage points as colored circles
for _, row in simulated_triage.iterrows():
    try:
        lat = row['latitude']
        lon = row['longitude']
        comp = row['Percent Comprehensive']
        drip = row['Percent Drip and Ship']
        CSC_time = row['CSC_travel_time_minutes']
        PSC_time = row['PSC_travel_time_minutes']
        transfer_time = row['transfer_time']
        hex_color = row['hex_color']
        if not diff_RACE:
            popup = f"Comprehensive: {comp:.1f}%<br>Drip and ship: {drip:.1f}%<br>CSC travel time: {CSC_time:.0f} min<br>PSC travel time: {PSC_time:.0f} min"
        else:
            popup = row['Difference']

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.7,
            popup = popup
            
            # popup = f"Comprehensive: {comp:.1f}%<br>Drip and ship: {drip:.1f}%<br>CSC travel time: {CSC_time:.1f}<br>PSC travel time: {PSC_time:.1f}<br>transfer time: {transfer_time:.1f} <br>lat: {lat} <br>long: {lon}"
        ).add_to(triage_layer)
        # print("success")
        
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
    

heat_data = [
    [row['latitude'], row['longitude'], (row['Percent Comprehensive'] - 50) / 50]
    for _, row in simulated_triage.iterrows()
    if not np.isnan(row['Percent Comprehensive'])
]

# Add heatmap to triage_layer
HeatMap(heat_data, min_opacity=0.3, radius=12, blur=15, max_zoom=13, max_val=1).add_to(heatmap_layer)


# Add layers to map
hospital_layer.add_to(m)
triage_layer.add_to(m)
heatmap_layer.add_to(m)

# Add layer control (checkboxes for layers)
folium.LayerControl().add_to(m)


# Show map in Streamlit
st.set_page_config(layout="wide")

st_folium(m, width=1000, height=800)

# # Plotly interpolated heatmap

# # Only use rows with location and valid percent
# triage = simulated_triage.dropna(subset=['latitude', 'longitude', 'Percent Comprehensive'])

# # Extract coordinates and values
# lats = triage['latitude'].values
# lons = triage['longitude'].values
# vals = triage['Percent Comprehensive'].values

# # Generate a regular lat/lon grid
# grid_lat, grid_lon = np.mgrid[
#     lats.min():lats.max():100j,
#     lons.min():lons.max():100j
# ]

# # Interpolate values using linear method
# grid_vals = griddata(
#     points=(lats, lons),
#     values=vals,
#     xi=(grid_lat, grid_lon),
#     method='linear'
# )

# # Plot using Plotly Heatmap
# fig = go.Figure(data=go.Heatmap(
#     z=grid_vals,
#     x=np.linspace(lons.min(), lons.max(), 100),
#     y=np.linspace(lats.min(), lats.max(), 100),
#     colorscale='RdBu_r',
#     zmin=0,
#     zmax=100,
#     colorbar=dict(title="Percent Comprehensive"),
#     hovertemplate='Lat: %{y:.4f}<br>Lon: %{x:.4f}<br>% Comp: %{z:.1f}%<extra></extra>'
# ))

# if lower_RACE:
#     title = "RACE: 1"
# else:
#     title = "RACE: 9"

# fig.update_layout(
#     title=f"Stroke Triage Zones (Interpolated Heatmap) {title}",
#     xaxis_title="Longitude",
#     yaxis_title="Latitude",
#     height=800,
#     yaxis_scaleanchor="x",  # keep correct aspect ratio
# )

# # Show in Streamlit
# st.plotly_chart(fig, use_container_width=True)
