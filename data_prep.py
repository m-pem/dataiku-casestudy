# data_ingestion_and_cleaning.py
# Here we will read in the data and perform some basic cleaning operations.

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

census_income_all = pd.read_csv('raw_data/census_income_all_cleaned.csv')


# update column names so that they have no spaces

census_income_all.columns = census_income_all.columns.str.replace(' ', '_')

# Update target variable for modelling
census_income_all['target'] = census_income_all['target'].replace({'-50000': 0, '50000': 1})

# Drop irrelevant columns, wage_per_hour as the data looks like junk, year as we dont need it
census_income_all = census_income_all.drop(columns=['wage_per_hour', 'year'])

# Min max scaling of numerical columns, there arent any!

# I want to create age group's
census_income_all['age_group'] = pd.cut(census_income_all['age'], bins=[0, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=['0-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])

# Drop the age column
census_income_all = census_income_all.drop(columns=['age'])

# country of birth - group them into human development index categories well on second thoughts after looking at the large number of categories I will just go locally bourn or bourn overseas
# https://en.wikipedia.org/wiki/List_of_countries_by_Human_Development_Index
# I would like to do something a bit more complex here but I will just group them into local or overseas
census_income_all['country_of_birth_self'] = np.where(census_income_all['country_of_birth_self'] == 'United-States', 'Local', 'Overseas')   

# I will now update the education column to group the values into categories
education_levels_dict = pd.read_csv('raw_data/education_levels.csv')

# lets join it into the main data
census_income_all = census_income_all.merge(education_levels_dict, how='left', left_on='education', right_on='Education Level')

# drop the education and Education Level columns, rename category to education
census_income_all = census_income_all.drop(columns=['education', 'Education Level'])
census_income_all = census_income_all.rename(columns={'Category': 'education'})

# Lets encode the categorical variables
# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Select categorical columns to encode
categorical_columns = ['sex', 'age_group','full_or_part_time_employment_stat', 'member_of_a_labor_union', 'country_of_birth_self', 'major_occupation_code', 'education']

# Fit and transform the data
encoded_data = encoder.fit_transform(census_income_all[categorical_columns])

# Create a DataFrame with the encoded data
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the encoded columns with the original DataFrame
finalised_data_frame = pd.concat([census_income_all.drop(columns=categorical_columns), encoded_df], axis=1)

finalised_data_frame.to_csv('processed_data/census_income_all_encoded.csv', index=False, header=True)