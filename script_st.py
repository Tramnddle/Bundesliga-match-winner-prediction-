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

# Create a dropdown menu
teamname = st.selectbox('Football Team List: ', Teamlist['opponent'].tolist())

df=df.drop(['Unnamed: 0','Unnamed: 0.1'], axis = 1)

user_inputs_A = st.text_input('Type in the name of team A', 'Dortmund')  # Example input
user_inputs_B = st.text_input('Type in the name of team B', 'Mainz 05')
user_inputs_date = st.date_input('Select a date')
venue = ['Home','Away']
user_inputs_venue = st.selectbox('Select a venue',venue)
user_inputs_round = 'Matchweek ' + str(st.number_input("Enter the matchweek", min_value =1, max_value=34, step=1, format="%d"))
user_inputs_season = st.number_input('Enter the season', min_value=2014, max_value=2050, step = 1 )

# Add new match to the dataframe:
df.loc[len(df.index)] = [user_inputs_date,None,'Bundesliga',user_inputs_round,None,user_inputs_venue,None,None,user_inputs_B,None,None,None,user_inputs_season,user_inputs_A]

df["date"] = pd.to_datetime(df["date"])

# Convert categorical variables into numerical variables
df["venue_code"] = df["venue"].astype("category").cat.codes
df["team_code"] = df["team"].astype("category").cat.codes
df["opp_code"] = df["opponent"].astype("category").cat.codes
#matches_rolling_A["hour"] = matches_rolling_A["time"].str.replace(":.+", "", regex=True).astype("int")
df["day_code"] = df["date"].dt.dayofweek
df['round']=df['round'].apply(lambda x: x.replace('Matchweek', '')).astype('int')

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
df = df.drop(['gf_x','gf_y'], axis = 1)


# average goal per team
average_goal_t = df.groupby('team')[['gf','ga']].mean().reset_index()
df = pd.merge(df, average_goal_t, on = "team")
df[['gf','ga','average_gf_t','average_ga_t']] = df[['gf_x','ga_x','ga_y','gf_y']].rename(columns={'gf_x': 'gf','ga_x':'ga','ga_y':'average_ga_t', 'gf_y': 'average_gf_t'})
df = df.drop(['gf_x','ga_x','gf_y','ga_y'], axis = 1)


# average goal per season per round
average_goal_sr = df.groupby(['season', 'round'])['gf'].mean().reset_index()
df = pd.merge(df, average_goal_sr, on=["season", "round"])
df[['gf','average_gf_sr']] = df[['gf_x','gf_y']].rename(columns={'gf_x': 'gf', 'gf_y': 'average_gf_sr'})


df['total_goal'] = df['gf'] + df['ga']

df_A = df[df['team']==user_inputs_A]
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
matches_rolling = df.groupby('team').apply(lambda x: rolling_averages(x, cols_1, cols_2, new_cols_1, new_cols_2))
matches_rolling.index = matches_rolling.index.droplevel()
st.dataframe(matches_rolling_A)

# def get_historical_data(df, group, cols, opp_cols):
    
home_team = matches_rolling.loc[len(matches_rolling.index), 'team']
away_team = matches_rolling.loc[len(matches_rolling.index), 'opponent']
date = matches_rolling.loc[len(matches_rolling.index), 'date']
        
# Filter the DataFrame for historical matches between the specified home and away teams
historical_matches_1 = matches_rolling[(matches_rolling['team'] == away_team) & (matches_rolling['opponent'] == home_team)] # historical matches stats of opponent team
historical_matches_2 = matches_rolling[(matches_rolling['team'] == home_team) & (matches_rolling['opponent'] == away_team)] # historical matches stats of home team
        
        # Exclude the current match by filtering based on the date
historical_matches_1 = historical_matches_1[historical_matches_1['date'] < date]
historical_matches_2 = historical_matches_2[historical_matches_2['date'] < date]

        # Select opponent's last match
matches = df[(df['team'] == away_team) & (df['opponent'] == home_team)]
matches = matches[matches['date'] == date]

cols = ['gf', 'sh', 'save%','poss']
opp_cols = ['save%_rolling','sh_rolling', 'gf_rolling']       
        
        # Select relevant columns for historical data
historical_data_1 = historical_matches_1[cols]
historical_data_2 = historical_matches_2[cols]
        
        # Optionally, you can aggregate the historical data (e.g., take the mean)
historical_data_1 = historical_data_1.mean()
historical_data_2 = historical_data_2.mean()
        
new_cols_1 = [f'{c}_hist_opp' for c in cols]
new_cols_2 = [f'{c}_hist_home' for c in cols]
new_cols_3 = [f'{c}_opp' for c in opp_cols]

match_AB = matches_rolling.iloc[-1]
match_AB[new_cols_1] = historical_data_1
match_AB[new_cols_2] = historical_data_2
match_AB[new_cols_3] = matches[opp_cols]

st.dataframe(match_AB)




