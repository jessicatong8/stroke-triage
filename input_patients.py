import pandas as pd
import numpy as np

def set_input_patients(sex, age, RACE_score, time_since_LKW, output_file):
    df = pd.read_csv("EMS-data/ems-strokes-traveltimes_with-transfer.csv")

    #choosing these values arbitrarily for now
    df['sex'] = sex
    df['age'] = age # majority of strokes occur in people aged 65 and older
    df['RACE'] = RACE_score # cutoff > 5 for LVO, max score of 9, this may be different from mRACE
    df['time_since_symptoms'] = time_since_LKW # 2hrs
    df['transfer_time'] = df['transfer_time_minutes']

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"wrote to {output_file}")
    return df

def random_input_patients(num_patients):
    df = pd.DataFrame({
        'sex': np.random.choice(['female','male'], size=num_patients),
        'age': np.random.randint(30,91, size=num_patients), #30-90 years old
        'RACE': np.random.randint(0,10, size=num_patients), # RACE score 0-9
        'time symptom': np.random.randint(10, 270, size=num_patients), #10 minutes - 4.5 hrs
        'time primary': np.random.uniform(10, 60, size=num_patients),
        'time comprehensive': np.random.uniform(10, 120, size=num_patients),
        })
    # Transfer time based on time_to_primary and time_to_comprehensive
    df['transfer time'] = np.random.uniform(
        abs(df['time comprehensive'] - df['time primary']),
        df['time comprehensive'] + df['time primary'], 
    )

    return df

df = random_input_patients(10000)
print(df.head())

# Save to CSV
output_file = 'input/random_10000.csv'
df.to_csv(output_file, index=False)
print(f"saved to {output_file}")