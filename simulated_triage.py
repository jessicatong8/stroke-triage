import streamlit as st
import pandas as pd
import numpy as np
import os
import folium
from streamlit_folium import st_folium

# --- Load CSV files ---
hospitals = pd.read_csv("EMS-data/hospitals.csv")
simulated_triage = pd.read_csv("output/output.csv")

# --- Data cleaning ---

hospitals = hospitals.rename(columns={
    "Y": "latitude",
    "X": "longitude"
})

simulated_triage = simulated_triage.rename(columns = {'origin_lat':'latitude', 'origin_lon':'longitude'})
print(simulated_triage.head(5))
print(simulated_triage.columns)


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
simulated_triage['hex_color'] = simulated_triage.apply(get_color, axis=1)


#color for hospital

# Default hospital color is blue
hospitals['hex_color'] = '#0000FF'  

# Set red color for CSCs
hospitals.loc[hospitals['Type'] == 'Comprehensive', 'hex_color'] = '#FF0000'  



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

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.7,
            popup = f"Comprehensive: {comp:.1f}%<br>Drip and ship: {drip:.1f}%<br>CSC travel time: {CSC_time:.0f} min<br>PSC travel time: {PSC_time:.0f} min"
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


# Add layers to map
triage_layer.add_to(m)
hospital_layer.add_to(m)

# Add layer control (checkboxes for layers)
folium.LayerControl().add_to(m)


# Show map in Streamlit
st.set_page_config(layout="wide")

st_folium(m, width=1000, height=800)


"""Travel times"""
"""
travel_times_file = 'EMS-data/travel_times.csv'
closest_primary_file = 'EMS-data/closest_primary_hospitals.csv'
closest_comprehensive_file = 'EMS-data/closest_comprehensive_hospitals.csv'

if os.path.exists(travel_times_file) and os.path.exists(closest_primary_file) and os.path.exists(closest_comprehensive_file):
    travel_df = pd.read_csv(travel_times_file)
    closest_primary = pd.read_csv(closest_primary_file)
    closest_comprehensive = pd.read_csv(closest_comprehensive_file)
    
    st.write("Travel times from stroke locations to hospitals (preprocessed):")
    st.dataframe(travel_df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Closest Primary hospital for each stroke location:")
        st.dataframe(closest_primary)
    with col2:
        st.write("Closest Comprehensive hospital for each stroke location:")
        st.dataframe(closest_comprehensive)
    
    st.write("Travel Time Summary:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Travel Time (All)", f"{travel_df['travel_time_minutes'].mean():.1f} min")
    with col2:
        st.metric("Average Distance (All)", f"{travel_df['distance_km'].mean():.1f} km")
    with col3:
        st.metric("Avg Time to Primary", f"{closest_primary['travel_time_minutes'].mean():.1f} min")
    with col4:
        st.metric("Avg Time to Comprehensive", f"{closest_comprehensive['travel_time_minutes'].mean():.1f} min")
        
    st.write("Detailed Statistics:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Calculations", f"{len(travel_df)}")
    with col2:
        st.metric("Google Maps API Calls", f"{len(travel_df[travel_df['method'] == 'google_maps'])}")
    with col3:
        st.metric("Haversine Calculations", f"{len(travel_df[travel_df['method'] == 'haversine'])}")
else:
    st.warning("⚠️ Travel time data not found. Please run 'preprocess_travel_times.py' first to calculate travel times.")
    st.write("Expected files:")
    st.write("- travel_times.csv")
    st.write("- closest_primary_hospitals.csv")
    st.write("- closest_comprehensive_hospitals.csv")
    st.write("To generate travel time data:")
    st.code("python preprocess_travel_times.py", language="bash")
"""
