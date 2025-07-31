import pandas as pd
import numpy as np


def set_input_patients(sex, age, RACE_score, time_since_LKW, output_file):
    """
    creates input csv for the model using real EMS and hospital data from Allegheny county
    """
    df = pd.read_csv("EMS-data/ems-strokes-traveltimes_with-transfer.csv")

    df['sex'] = sex
    df['age'] = age # majority of strokes occur in people aged 65 and older
    df['RACE'] = RACE_score # cutoff > 5 for LVO, max score of 9, this is a little different from the mRACE which has a max score of 11
    df['time_since_symptoms'] = time_since_LKW # max 4.5 hours or 270 minutes
    df['transfer_time'] = df['transfer_time_minutes']

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"wrote input patients to {output_file}")
    return df

def random_input_patients(num_patients, output_file):
    """
    creates input csv for the model using randomly generated input parameters, and eventually used to train the ML model
    """

    df = pd.DataFrame({
        'sex': np.random.choice(['female','male'], size=num_patients),
        'age': np.random.randint(30,91, size=num_patients), #30-90 years old
        'RACE': np.random.randint(0,10, size=num_patients), # RACE score 0-9
        'NIHSS': 'NA',
        'time symptom': np.random.randint(10, 270, size=num_patients), #10 minutes - 4.5 hrs
        'time primary': np.random.uniform(10, 60, size=num_patients), # chose these arbitrarily, could likely be refined by referencing studies measuring median/avg travel time to nearest stroke centers
        'time comprehensive': np.random.uniform(10, 120, size=num_patients), # chose these arbitrarily, could likely be refined by referencing studies measuring median/avg travel time to nearest stroke centers
    })

    # Transfer time based on time_to_primary and time_to_comprehensive
    df['transfer time'] = np.random.uniform(
        abs(df['time comprehensive'] - df['time primary']),
        df['time comprehensive'] + df['time primary'], 
    )

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"wrote input patients to {output_file}")
    return df

random_input_patients(15000, 'input/random_15000.csv')