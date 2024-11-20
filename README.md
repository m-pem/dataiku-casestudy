# dataiku-casestudy

Here we perform some data analysis

## Files
- exploratory_data_analysis.ipynb: initaial data processing and data exploration
- data_prep.py: Process cleaned data into features
- model_development.py: Define framework to train, tune, and evaluate models
- model_evaluation.py: Train and explore model performance

# Overview of project

## Problem Statement

For this technical assessment, you have been tasked with identifying characteristics that are
associated with a person making more or less than $50,000 per year; the target variable for
your research question is the final column of the datasets.

As the data scientist on this project, you are to attempt to answer this question by constructing a
data analysis/modeling pipeline. Code submissions should be in Python and making the solution
easily readable and replicable by the team will give you additional marks. In the event you would
like to use a different language or tool, please ask. Considerations for your data analysis should
include, but are not limited to, the following:

- Exploratory Data Analysis: Numerical and/or graphical representations of the data that
may help inform insights and/or tactics for answering the research question of interest.
- Data Preparation: Data cleaning, preprocessing, feature engineering, etc., that may aid
in improving data clarity & model generation.
-Data Modeling: The building of a few competing models to predict the target variable.
-Model Assessment: A selection of the best model based on performance comparisons.
-Results: A concise summary of key findings, recommendations, & future improvements.

# Reproduce results

To reproduce the results you will first need to download the datafiles from [here](https://drive.google.com/drive/folders/1PPsjCoM130k3n3V4roq-yF74jkPjkVd7)

Save them at the root of the project in a file called raw_data/

With the files downloaded you will then need to install the packages required to run this project.

I reccomend using a virtual env (see [here](https://docs.python.org/3/library/venv.html) for guidence) to manage your work space, with Python 3.11.5

With the virtual env created and activated, or not, in the root dir you can install all the required dependencies using:

`pip install -r requirenents.txt`

With the pakcages sucesfully installed you are now able to reproduce the results. I have been switching between a notebook format and classic scripting in the VS code editor. When using the notebook make sure you're selecting the python interpreter from your virtual env if using. 




Start with the exploratory data analysis to meld the datasets and do some exploratory data analysis
Then run the data data prep script to create the features for the modeling
Run model evaluation to train, evaluate, and review the outcome of the modeling.

