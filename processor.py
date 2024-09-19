import pandas as pd
df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')
def preprocess(df , region_df):


    df =df.merge(region_df, on='NOC', how='left')

    df.drop_duplicates(inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Medal'])], axis=1)
    return df

