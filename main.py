import pandas as pd
import numpy as np
import matplotlib as mlp 
import seaborn as sns #conda install seaborn
from sklearn import preprocessing

# 1. download conda: https://www.anaconda.com/docs/getting-started/miniconda/install#windows-command-prompt
# 2. after download create environment in VSCode terminal:
#       conda create -n DataScienceProject 
#   where DataScienceProject is the name of the environment
# 3. activate the environment:
#   conda activate DataScienceProject
# 4. install packages: 
#   conda install matplotlib scikit-learn pandas
# 5. Open your command pallet using settings gear in bottom left corner
# 6. Select Python Interpreter: Python 3.13.2 ('DataScienceProject)
# 7. Run file to make sure theres no install issues

#initalize dataset 
GlobalImpact = pd.read_csv('Global_AI_Content_Impact_Dataset.csv')
GlobalImpact.sort_values(by=['Country', 'Year'], ascending=True, inplace=True)
print(GlobalImpact.head())

#Data Cleaning
GlobalImpact.drop_duplicates(inplace=True)

#fill missing numeric values with the mean
mean = GlobalImpact.mean(numeric_only=True)
GlobalImpact.fillna(value=mean, inplace=True)

#Data Visualization





