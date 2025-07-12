import streamlit as st
import pandas as pd
import numpy as np
import os


# load csv file
df = pd.read_csv("allems-strokes.csv")
hosp_ef = pd.read_csv("hospitals.csv")

#filter csv by column "description_short" = "STROKE"
df = df[df["description_short"] == "STROKE"]

#rename census_block_group_center__y to "latitude" and census_block_group_center__x to "longitude"
df = df.rename(columns={
    "census_block_group_center__y": "latitude",
    "census_block_group_center__x": "longitude"
})

#rename columns Y, X in hosp_ef to "latitude" and "longitude"
hosp_ef = hosp_ef.rename(columns={
    "Y": "latitude",
    "X": "longitude"
})

combined_df = pd.concat([df, hosp_ef], ignore_index=True)

# Add a color column based on whether the row has a "facility" column with a value
if 'Facility' in combined_df.columns:
    # Color points blue if the row has a non-null "facility" value, otherwise red
    combined_df['color'] = combined_df['Facility'].notna().map({True: '#0000FF', False: '#FF0000'})
else:
    # If no facility column exists, color all points red
    combined_df['color'] = '#FF0000'

# Group by latitude and longitude to count points at the same location
# and preserve the color information
aggregated_df = combined_df.groupby(['latitude', 'longitude', 'color']).size().reset_index(name='count')

# Exaggerate the point sizes by multiplying by 3
aggregated_df['size'] = aggregated_df['count'] * 3

# use the column size for the size, color for the color, and latitude and longitude for the coordinates in the st.map()             
st.map(aggregated_df, color='color', size='size')

# Load preprocessed travel times
travel_times_file = 'travel_times.csv'
closest_primary_file = 'closest_primary_hospitals.csv'
closest_comprehensive_file = 'closest_comprehensive_hospitals.csv'

if os.path.exists(travel_times_file) and os.path.exists(closest_primary_file) and os.path.exists(closest_comprehensive_file):
    # Load preprocessed data
    travel_df = pd.read_csv(travel_times_file)
    closest_primary = pd.read_csv(closest_primary_file)
    closest_comprehensive = pd.read_csv(closest_comprehensive_file)
    
    # Display travel times
    st.write("Travel times from stroke locations to hospitals (preprocessed):")
    st.dataframe(travel_df)
    
    # Display closest hospitals by type
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Closest Primary hospital for each stroke location:")
        st.dataframe(closest_primary)
        
    with col2:
        st.write("Closest Comprehensive hospital for each stroke location:")
        st.dataframe(closest_comprehensive)
    
    # Show summary statistics
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
        
    # Additional statistics
    st.write("Detailed Statistics:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Calculations", f"{len(travel_df)}")
    with col2:
        primary_hospitals_df = travel_df[travel_df['dest_type'] == 'Primary']
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
