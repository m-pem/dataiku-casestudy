# /Exploratory Data Analysis

# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in data
column_names = pd.read_csv('raw_data/column_names.csv', sep=':', header=None)
actual_column_names = column_names.iloc[:,0].tolist()
raw_census_data_learn = pd.read_csv('raw_data/census_income_learn.csv', header=None, names=actual_column_names)
raw_census_data_test = pd.read_csv('raw_data/census_income_test.csv', header=None, names=actual_column_names)
raw_census_data_all = pd.concat([raw_census_data_learn, raw_census_data_test], ignore_index=True)
raw_census_data_all.head().to_csv("raw_data/census_income_all_preview.csv", ",", index=False)
raw_census_data_all.to_csv("raw_data/census_income_all.csv", ",", index=False)

# data exploration

raw_census_data_all = pd.read_csv('raw_data/census_income_all.csv', sep=',')


# Exploratory Data Analysis: Numerical and/or graphical representations of the data that
# may help inform insights and/or tactics for answering the research question of interest.

# going to select a few elements that I believe will drive income or that I have a hunch about
# Reason being we want to show that we would normally partner with business partners to help guide where we should focus
# maybe an example of something that says x

# next I'm going to do some exploritory data analysis on those features we've selected

# first thing we need to do is read about the data

# sex, age, member of a labor union, country of birth self, education, major occupation code, wage per hour

# take our data selecet our columns and create some summary statistics. 

# How are we going to take into account the weighting. Are we going to explode the rows or are we going to incorporate it into our processing, I think when we groupby sum we just need to take that into account

# Count of sex
raw_census_data_all['instance weight']


# Count of member of a labor union
# 
# Count of country of birth self
# 
# Count of education
# 
# Count of major occupation code
# 
# Count of wage per hour 