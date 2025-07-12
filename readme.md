# EMS Visualization and Travel Time Analysis

This project visualizes emergency medical service (EMS) stroke incidents and visualizes a mathematical model's recommendations for transport strategy.

The underlying model is from

**Optimization of Prehospital Triage of Patients with Suspected Ischemic Stroke: Results of a Mathematical Model**

_Ali A, Zachrison KS, Eschenfeldt PC, Schwamm LH, Hur C_

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App

```bash
streamlit run simulated_triage.py
```

This will:

- Load the preprocessed travel time data
- Load the preprocessed model outputs for stroke incident transport strategy recommendations
- Display an interactive map with stroke incidents (circles) and hospitals (squares) color coded by the model's recommended transport strategy

## Files

- `simulated_triage.py` - Main Streamlit application
- `input.csv` - model input with incident data, travel times, and some arbitrary assumptions on patient age, LKW, and RACE score
- `output.csv` - model output with the recommended transport strategy
- `ems-strokes-traveltimes.csv` - EMS stroke incidents with travel time to closest comprehensive and primary hospitals
- `travel_times.csv` - Generated travel time matrix
- `closest_hospitals.csv` - Generated closest hospital assignments
- `allems-strokes.csv` - EMS incident data filtered to only Strokes (from [WPRDC](https://data.wprdc.org/dataset/allegheny-county-911-dispatches-ems-and-fire))
- `hospitals.csv` - Hospital location data (adapted from [WPRDC](https://data.wprdc.org/dataset/hospitals) but added a new Type column for Primary vs Comprehensive)
