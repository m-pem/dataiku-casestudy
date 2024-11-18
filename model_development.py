import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight


# read in data
data_for_model = pd.read_csv('processed_data/census_income_all_encoded.csv')

# Extract sample weights
sample_weights = data_for_model['instance_weight']

# separate features and target 
X = data_for_model.drop(columns=['target', 'instance_weight'])
y = data_for_model['target']


## Right now I'm working to address the imbalanced nature of the data
# I've tried the below but I think because I'm already using sample weight creating class weights isnt working.
# The reason this is important is because I'm not getting decent results.

# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y),
#     y=y,
#     sample_weight=sample_weights
# )
# class_weight_dict = dict(zip(np.unique(y), class_weights))


# I'd like to set up a model pipeline to make it easier to evaluate different models - hmmm maybe not? Seems like a waste of time



# split into test / train
X_train, X_test, y_train, y_test, train_weights, test_weights = train_test_split(X, y, sample_weights, test_size=0.2, random_state=42)

# Create a random forest model

# define the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# fit the model
rf_model.fit(X_train, y_train, sample_weight=train_weights)

# evaluate model
predictions = rf_model.predict(X_test)
pred_proba = rf_model.predict_proba(X_test)[:,1]

# Calculate metrics
print("Classification Report (weighted by sample weights):")
print(metrics.classification_report(y_test, predictions, sample_weight=test_weights))

print("\nConfusion Matrix:")
print(metrics.confusion_matrix(y_test, predictions, sample_weight=test_weights))

# ROC-AUC score
roc_auc = metrics.roc_auc_score(y_test, pred_proba, sample_weight=test_weights)
print(f"\nROC-AUC Score: {roc_auc:.3f}")



# calculate feature importance
feature_importance = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
print(feature_importance_df)

# evaluate model

##################### Reweighted to account for the fact that the data is imbalanced #####################

# calculate feature importance

# save model