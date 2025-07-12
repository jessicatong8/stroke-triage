import pandas as pd

df = pd.read_csv("EMS-data/ems-strokes-traveltimes.csv")

#choosing these values arbitrarily for now
df['sex'] = "female"
df['age'] = 70 # majority of strokes occur in people aged 65 and older
df['RACE'] = 7 # cutoff > 5 for LVO, max score of 9, this may be different from mRACE
df['time_since_symptoms'] = 120 # 2hrs
df['transfer_time'] = abs(df['CSC_travel_time_minutes'] - df['PSC_travel_time_minutes']) #for now taking the diff, definitely not correct

# Save to CSV
output_file = 'input/scenarios.csv'
df.to_csv(output_file, index=False)
print("wrote to csv file")
