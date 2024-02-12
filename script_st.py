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


