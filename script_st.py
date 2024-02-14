import subprocess

# Install required packages from requirements.txt
subprocess.call("pip install -r requirements.txt", shell=True)

import numpy as np
import pandas as pd
import streamlit as st
from st_files_connection import FilesConnection
import gcsfs
import os
import lightgbm as lgb
import joblib


secrets = st.secrets["connections_gcs"]
secret_value = os.environ.get('connections_gcs')

# Create a GCS connection
conn = st.experimental_connection('gcs', type=FilesConnection)

# Read a file from the cloud storage
df = conn.read("gs://bundesliga_0410/matches.csv", input_format="csv")
Teamlist = conn.read("gs://bundesliga_0410/Teamlist.csv", input_format="csv")

st.title('Bundesliga match score prediction')

# Open the df file
st.dataframe(df)

# Create a dropdown menu
#teamname = st.selectbox('Football Team List: ', list(Teamlist['opponent'].itertuples(index=False, name=None)))
teamname = st.selectbox('Football Team List: ', Teamlist['opponent'].tolist())

df=df.drop('Unnamed: 0', axis = 1)
df["date"] = pd.to_datetime(df["date"])

# Convert categorical variables into numerical variables
df["venue_code"] = df["venue"].astype("category").cat.codes
df["team_code"] = df["team"].astype("category").cat.codes
df["opp_code"] = df["opponent"].astype("category").cat.codes
#matches_rolling_A["hour"] = matches_rolling_A["time"].str.replace(":.+", "", regex=True).astype("int")
df["day_code"] = df["date"].dt.dayofweek
df['round']=df['round'].apply(lambda x: x.replace('Matchweek', '')).astype('int')

user_inputs_A = st.text_input('Type in the name of team A', 'Dortmund')  # Example input
user_inputs_B = st.text_input('Type in the name of team B', 'Mainz 05')
user_inputs_date = st.date_input('Select a date')
venue = ['Home','Away']
user_inputs_venue = st.selectbox('Select a venue',venue)
user_inputs_round = st.number_input("Enter the matchweek", min_value =1, max_value=34, step=1, format="%d")
user_inputs_season = st.number_input('Enter the season', min_value=2014, max_value=2050, step = 1 )

# rename and match the teams name of home team and opponent team columns
Team_name = {
    'Arminia':'Arminia',
    'Augsburg':'Augsburg',
    'Bayer Leverkusen': 'Bayer Leverkusen',
    'Bayern Munich':'Bayern Munich',
    'Bochum':'Bochum',
    'Darmstadt 98':'Darmstadt 98',
    'Dortmund':'Dortmund',
    'Eintracht Frankfurt': 'Eintracht Frankfurt',
    'Freiburg': 'Freiburg',
    'Greuther Furth': 'Greuther Fürth',
    'Heidenheim': 'Heidenheim',
    'Hertha BSC': 'Hertha BSC',
    'Hoffenheim': 'Hoffenheim',
    'Koln': 'Köln',
    'Mainz 05':'Mainz 05',
    'Monchengladbach': 'Monchengladbach',
    'RB Leipzig':'RB Leipzig',
    'Schalke 04': 'Schalke 04',
    'Stuttgart': 'Stuttgart',
    'Union Berlin': 'Union Berlin',
    'Werder Bremen': 'Werder Bremen',
    'Wolfsburg': 'Wolfsburg',
    'Dusseldorf': 'Düsseldorf',
    'Hamburger SV': 'Hamburger SV',
    'Paderborn 07': 'Paderborn 07',
    'Hannover 96': 'Hannover 96',
    'Nurnberg': 'Nürnberg',
    'Ingolstadt 04':'Ingolstadt 04',
    'Eintracht Braunschweig': 'Eintracht Braunschweig'
}
Opponent_name = {
    'Arminia':'Arminia',
    'Augsburg':'Augsburg',
    'Leverkusen': 'Bayer Leverkusen',
    'Bayern Munich':'Bayern Munich',
    'Bochum':'Bochum',
    'Darmstadt 98':'Darmstadt 98',
    'Dortmund':'Dortmund',
    'Eint Frankfurt': 'Eintracht Frankfurt',
    'Freiburg': 'Freiburg',
    'Greuther Fürth': 'Greuther Fürth',
    'Heidenheim': 'Heidenheim',
    'Hertha BSC': 'Hertha BSC',
    'Hoffenheim': 'Hoffenheim',
    'Köln': 'Köln',
    'Mainz 05':'Mainz 05',
    "M'Gladbach": 'Monchengladbach',
    'RB Leipzig':'RB Leipzig',
    'Schalke 04': 'Schalke 04',
    'Stuttgart': 'Stuttgart',
    'Union Berlin': 'Union Berlin',
    'Werder Bremen': 'Werder Bremen',
    'Wolfsburg': 'Wolfsburg',
    'Düsseldorf': 'Düsseldorf',
    'Hamburger SV': 'Hamburger SV',
    'Paderborn 07': 'Paderborn 07',
    'Hannover 96': 'Hannover 96',
    'Nürnberg': 'Nürnberg',
    'Ingolstadt 04':'Ingolstadt 04',
    'Braunschweig': 'Eintracht Braunschweig'
}

df['team'] = df['team'].map(Team_name)
df['opponent'] = df['opponent'].map(Opponent_name)

# average goal per season and team
average_goal_st= df.groupby(['team', 'season'])[['gf','ga']].mean().reset_index()
df = pd.merge(df, average_goal_st, on = ["team","season"])
df[['gf','ga','average_ga_st','average_gf_st']] = df[['gf_x','ga_x','ga_y', 'gf_y']].rename(columns=
                                {'gf_x': 'gf','ga_x':'ga', 'ga_y':'average_ga_st', 'gf_y': 'average_gf_st'})
df = df.drop(['gf_x','ga_x','gf_y','ga_y'], axis = 1)

# average goal per season
average_goal_s = df.groupby('season')['gf'].mean().reset_index()
df = pd.merge(df, average_goal_s, on = "season")
df[['gf','average_gf_s']] = df[['gf_x','gf_y']].rename(columns={'gf_x': 'gf', 'gf_y': 'average_gf_s'})
df = pd.DataFrame(df)
df = df.drop(['gf_x','gf_y'], axis = 1)

# average goal per team
average_goal_t = df.groupby('team')[['gf','ga']].mean().reset_index()
df = pd.merge(df, average_goal_t, on = "team")
df[['gf','ga','average_gf_t','average_ga_t']] = df[['gf_x','ga_x','ga_y','gf_y']].rename(columns={'gf_x': 'gf','ga_x':'ga','ga_y':'average_ga_t', 'gf_y': 'average_gf_t'})
df = df.drop(['gf_x','ga_x','gf_y','ga_y'], axis = 1, inplace=True)

st.write(df)

# average goal per season per round
average_goal_sr = df.groupby(['season', 'round'])['gf'].mean().reset_index()
df = pd.merge(df, average_goal_sr, on=["season", "round"])
df[['gf','average_gf_sr']] = df[['gf_x','gf_y']].rename(columns={'gf_x': 'gf', 'gf_y': 'average_gf_sr'})

df['total_goal'] = df['gf'] + df['ga']

# Add new match to the dataframe:
match_AB = {
            'date':user_inputs_date,
            'time':'',
            'comp': 'Bundesliga',
            'round':user_inputs_round,
            'day':'',
            'venue':user_inputs_venue,
            'gf':'',
            'ga':'',
            'opponent':user_inputs_B,
            'poss':'',
            'sh':'',
            'save%':'',
            'season':user_inputs_season,
            'team':user_inputs_A,
            }
df = df.append(match_AB, ignore_index=True)

df_A = df[df["team"]==user_inputs_A]

# average of the last 3 games
def rolling_averages(group, cols_1, cols_2, new_cols_1, new_cols_2):
    group = group.sort_values("date")
    rolling_stats = group[cols_1].rolling(3, closed='left').mean()
    previous_stats = group[cols_2].rolling(1, closed='left').mean()
    group[new_cols_1] = rolling_stats
    group[new_cols_2] = previous_stats
    #group = group.dropna(subset=new_cols)
    return group

cols_1 = ["gf",'ga','poss','sh', 'save%']
new_cols_1 = [f"{c}_rolling" for c in cols_1] 

cols_2 = []
new_cols_2 = [f"{c}_last_game" for c in cols_2]
group = df_A

matches_rolling_A = rolling_averages(group, cols_1, cols_2, new_cols_1, new_cols_2)

def get_historical_data(df, group, cols, opp_cols):
    for i in range(len(group)):
        home_team = group['team'].iloc[i]
        away_team = group['opponent'].iloc[i]
        date = group['date'].iloc[i]
        
        # Filter the DataFrame for historical matches between the specified home and away teams
        historical_matches_1 = df[(df['team'] == away_team) & (df['opponent'] == home_team)] # historical matches stats of opponent team
        historical_matches_2 = df[(df['team'] == home_team) & (df['opponent'] == away_team)] # historical matches stats of home team
        
        # Exclude the current match by filtering based on the date
        historical_matches_1 = historical_matches_1[historical_matches_1['date'] < date]
        historical_matches_2 = historical_matches_2[historical_matches_2['date'] < date]

        # Select opponent's last match
        matches = df[(df['team'] == away_team) & (df['opponent'] == home_team)]
        matches = matches[matches['date'] == date]
        
        # Select opponent's defensive stats last match
        matches[opp_cols]
        
        # Select relevant columns for historical data
        historical_data_1 = historical_matches_1[cols]
        historical_data_2 = historical_matches_2[cols]
        
        # Optionally, you can aggregate the historical data (e.g., take the mean)
        historical_data_1 = historical_data_1.mean()
        historical_data_2 = historical_data_2.mean()
        
        new_cols_1 = [f'{c}_hist_opp' for c in cols]
        new_cols_2 = [f'{c}_hist_home' for c in cols]
        new_cols_3 = [f'{c}_opp' for c in opp_cols]

        # Append the new columns to the DataFrame for each iteration
        for col, new_col in zip(cols, new_cols_1):
            group.loc[group.index[i], new_col] = historical_data_1[col]

        for col, new_col in zip(cols, new_cols_2):
            group.loc[group.index[i], new_col] = historical_data_2[col]

        for col, new_col in zip(opp_cols, new_cols_3):
            group.at[group.index[i], new_col] = matches[col].iloc[0]
            
    return group

# Example usage
cols = ['gf', 'sh', 'save%','poss']
opp_cols = ['save%_rolling','sh_rolling', 'gf_rolling']
df = df.copy()  # Ensure that the original DataFrame is not modified
group = matches_rolling_A
matches_rolling_A = get_historical_data(df, group, cols, opp_cols)

matches_rolling_A = matches_rolling_A[['date', 'time', 'comp', 'round', 'day', 'venue', 'opponent', 'poss',
       'sh', 'save%', 'season', 'team', 'gf_rolling', 'ga_rolling',
       'poss_rolling', 'sh_rolling', 'save%_rolling', 'gf_hist_opp',
       'sh_hist_opp', 'save%_hist_opp', 'poss_hist_opp', 'gf_hist_home',
       'sh_hist_home', 'save%_hist_home', 'poss_hist_home',
       'save%_rolling_opp', 'sh_rolling_opp', 'gf_rolling_opp', 'venue_code',
       'team_code', 'opp_code', 'day_code', 'average_ga_st', 'average_gf_st',
       'average_gf_s', 'ga', 'average_gf_t', 'average_ga_t', 'total_goal',
       'gf', 'average_gf_sr']]

st.write(matches_rolling_A)
