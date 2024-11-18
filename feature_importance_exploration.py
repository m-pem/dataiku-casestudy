from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate permutation importance for each class
result = permutation_importance(
    rf_model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    sample_weight=test_weights
)

# Create importance DataFrame per class
class_names = ['<=50K', '>50K']  # adjust based on your actual class labels
importance_per_class = pd.DataFrame()

for class_idx, class_name in enumerate(class_names):
    # Get feature importance for this class
    class_importances = rf_model.feature_importances_
    
    # Create DataFrame for this class
    class_df = pd.DataFrame({
        'feature': X.columns,
        f'importance_{class_name}': class_importances
    })
    
    if importance_per_class.empty:
        importance_per_class = class_df
    else:
        importance_per_class = importance_per_class.merge(class_df, on='feature')

# Sort by average importance
importance_per_class['avg_importance'] = importance_per_class.iloc[:, 1:].mean(axis=1)
importance_per_class = importance_per_class.sort_values('avg_importance', ascending=False)

# Plot top 10 features
plt.figure(figsize=(10, 6))
importance_per_class.head(10).plot(
    x='feature',
    y=[f'importance_{class_name}' for class_name in class_names],
    kind='bar'
)
plt.title('Top 10 Feature Importance by Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Top 10 Features by Class:")
print(importance_per_class.head(10))