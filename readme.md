# EMS Visualization and Travel Time Analysis

This project visualizes model predictions for optimal stroke transport stratgies for suspected stroke patients in Allegheny County to support emergency medical services.

The underlying model is from

**Optimization of Prehospital Triage of Patients with Suspected Ischemic Stroke: Results of a Mathematical Model**

_Ali A, Zachrison KS, Eschenfeldt PC, Schwamm LH, Hur C_

To enable real-time interaction for the web visualization, an XGboost model was used to predict the mathematical model's outputs.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App

```bash
streamlit run visualization.py
```

This will:

- Load XGboost model predicting optimal stroke transport strategies
- Display an interactive map with stroke incidents (circles) and hospitals (squares) color coded by the model's recommended transport strategy

## Files

- `main.py` - Runs the mathematical model adapted from Ali et al.
- `XGboost.ipynb` - XBboost modeling predictions from the mathematical model
- `visualization.py` - Main Streamlit application
- `ems-strokes-traveltimes_with-transfer.csv` - EMS stroke incidents with travel time to closest comprehensive and primary hospitals and transfer time in between
- `travel_times.csv` - Generated travel time matrix
- `closest_hospitals.csv` - Generated closest hospital assignments
- `allems-strokes.csv` - EMS incident data filtered to only Strokes (from [WPRDC](https://data.wprdc.org/dataset/allegheny-county-911-dispatches-ems-and-fire))
- `hospitals.csv` - Hospital location data (adapted from [WPRDC](https://data.wprdc.org/dataset/hospitals) but added a new Type column for Primary vs Comprehensive)
