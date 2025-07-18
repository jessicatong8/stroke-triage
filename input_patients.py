import pandas as pd

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
