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
teamname = st.selectbox('Football Team List: ', list(Teamlist['opponent'].itertuples(index=False, name=None)))

user_inputs_A = st.text_input('Type in the name of team A', 'Dortmund')  # Example input
user_inputs_B = st.text_input('Type in the name of team B', 'Mainz 05')