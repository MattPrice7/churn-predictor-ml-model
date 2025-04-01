import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

##############################################################
# 1. DATA LOADING AND PREPROCESSING
##############################################################

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['contract', 'paymentmethod', 'internetservice'], drop_first=True)

# Map binary columns
binary_columns = [
    'seniorcitizen', 'partner', 'dependents', 
    'phoneservice', 'multiplelines', 'onlinesecurity', 
    'onlinebackup', 'deviceprotection', 'techsupport', 
    'streamingtv', 'streamingmovies', 'paperlessbilling'
]
for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
    
# Map gender and churn
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})

# Re-clean column names to catch any new columns created by get_dummies
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

# Drop columns that are not needed
df = df.drop(columns=['customerid', 'totalcharges'])

# Separate label and features
y = df.churn
features = [col for col in df.columns if col != 'churn']
x = df[features]

##############################################################
# 2. DATA SPLITTING
##############################################################
# Split data into training (60%), validation (20%), and test (20%) sets.
train_x, temp_x, train_y, temp_y = train_test_split(x, y, test_size=0.4, random_state=1)
val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=1)

##############################################################
# 3. BASELINE DECISION TREE MODEL
##############################################################
# Fit a baseline Decision Tree model
churn_model = DecisionTreeClassifier(random_state=1)
churn_model.fit(train_x, train_y)
val_predictions = churn_model.predict(val_x)
val_probabilities = churn_model.predict_proba(val_x)[:, 1]
auc = roc_auc_score(val_y, val_probabilities)
accuracy = accuracy_score(val_y, val_predictions)
print("Baseline Decision Tree:")
print("AUC:", auc)
print("Accuracy:", accuracy)

##############################################################
# 4. DECISION TREE TUNING (max_leaf_nodes)
##############################################################
# Test different max_leaf_nodes values for the Decision Tree
max_leaf_node_values = [10, 20, 50, 100, 200]
best_auc = 0
best_max_leaf_nodes = None

for leaf_node_value in max_leaf_node_values:
    churn_model = DecisionTreeClassifier(random_state=1, max_leaf_nodes=leaf_node_value)
    churn_model.fit(train_x, train_y)
    val_predictions = churn_model.predict(val_x)
    val_probabilities = churn_model.predict_proba(val_x)[:, 1]
    auc = roc_auc_score(val_y, val_probabilities)
    accuracy = accuracy_score(val_y, val_predictions)
    print(f"Max Leaf Nodes: {leaf_node_value} -> AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        best_max_leaf_nodes = leaf_node_value

print(f"\nBest Max Leaf Nodes (Decision Tree): {best_max_leaf_nodes} with AUC: {best_auc:.4f}")

# (Older iteration of Decision Tree tuning remains commented out below)
"""
# Old code snippet for decision tree tuning:
for leaf_node_value in [10,20,50,100,200]:
    ...
"""

##############################################################
# 5. BASELINE RANDOM FOREST MODEL
##############################################################
# Fit a baseline Random Forest model
rf_model = RandomForestClassifier(random_state=2)
rf_model.fit(train_x, train_y)
rf_val_predictions = rf_model.predict(val_x)
rf_val_probabilities = rf_model.predict_proba(val_x)[:, 1]
rf_auc = roc_auc_score(val_y, rf_val_probabilities)
rf_accuracy = accuracy_score(val_y, rf_val_predictions)
print("\nBaseline Random Forest:")
print("RF AUC:", rf_auc)
print("RF Accuracy:", rf_accuracy)

##############################################################
# 6. RANDOM FOREST HYPERPARAMETER TUNING WITH GRID SEARCH
##############################################################
# Define a parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='roc_auc', verbose=0, n_jobs=-1)
grid_search.fit(train_x, train_y)
best_params = grid_search.best_params_
print(f"Best hyperparameters (Random Forest): {best_params}")

best_rf_model = grid_search.best_estimator_
gs_val_predictions = best_rf_model.predict(val_x)
gs_val_probabilities = best_rf_model.predict_proba(val_x)[:, 1]

# Evaluate performance on the validation set
gs_auc = roc_auc_score(val_y, gs_val_probabilities)
gs_accuracy = accuracy_score(val_y, gs_val_predictions)
precision = precision_score(val_y, gs_val_predictions)
recall = recall_score(val_y, gs_val_predictions)
f1 = f1_score(val_y, gs_val_predictions)
print("Optimized Random Forest on Validation:")
print("AUC:", gs_auc)
print("Accuracy:", gs_accuracy)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

##############################################################
# 7. THRESHOLD OPTIMIZATION ON VALIDATION SET
##############################################################
# Optimize threshold (e.g., for F1 or recall) on the validation set
thresholds = np.linspace(0, 1, 101)
best_f1 = 0
best_f1_threshold = 0.5
best_recall = 0
best_recall_threshold = 0.5

for thresh in thresholds:
    temp_preds = (gs_val_probabilities >= thresh).astype(int)
    current_f1 = f1_score(val_y, temp_preds)
    current_recall = recall_score(val_y, temp_preds)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_f1_threshold = thresh
    if current_recall > best_recall:
        best_recall = current_recall
        best_recall_threshold = thresh
    print(f"Threshold: {thresh:.2f} - F1: {current_f1:.4f}, Recall: {current_recall:.4f}")

print(f"\nBest F1 Threshold on validation: {best_f1_threshold:.2f} with F1: {best_f1:.4f}")
print(f"Best Recall Threshold on validation: {best_recall_threshold:.2f} with Recall: {best_recall:.4f}")

# Create optimized predictions on validation set using the chosen threshold
custom_threshold = best_f1_threshold  # or best_recall_threshold, based on your objective
opt_val_predictions = (gs_val_probabilities >= custom_threshold).astype(int)

opt_accuracy = accuracy_score(val_y, opt_val_predictions)
opt_precision = precision_score(val_y, opt_val_predictions)
opt_recall = recall_score(val_y, opt_val_predictions)
opt_f1 = f1_score(val_y, opt_val_predictions)
print("Optimized Metrics on Validation (using custom threshold):")
print("Accuracy:", opt_accuracy)
print("Precision:", opt_precision)
print("Recall:", opt_recall)
print("F1 Score:", opt_f1)

##############################################################
# 8. VISUALIZATION (Optional; can be commented out)
##############################################################
# Feature Importance

print("\nFeature Importance")
feature_importances = best_rf_model.feature_importances_
importance_df = pd.DataFrame({'feature': features, 'importance': feature_importances}).sort_values(by='importance', ascending=False)
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# ROC Curve (using optimized threshold predictions might be less meaningful since ROC is threshold-independent)
print("\nROC Curve")
fpr, tpr, thresholds_curve = roc_curve(val_y, gs_val_probabilities)
plt.plot(fpr, tpr, label=f'AUC = {gs_auc:.2f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Precision Recall Curve
print("\nPrecision Recall Curve")
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(val_y, gs_val_probabilities)
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, marker='.', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Confusion Matrix (using optimized predictions)
print("\nConfusion Matrix")
cm = confusion_matrix(val_y, opt_val_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f"Confusion Matrix (Threshold = {custom_threshold:.2f})")
plt.show()


##############################################################
# 9. FINAL EVALUATION ON TEST SET
##############################################################
# Apply the chosen threshold to the test set

test_probabilities = best_rf_model.predict_proba(test_x)[:, 1]
test_predictions = (test_probabilities >= custom_threshold).astype(int)

test_auc = roc_auc_score(test_y, test_probabilities)
test_accuracy = accuracy_score(test_y, test_predictions)
test_precision = precision_score(test_y, test_predictions)
test_recall = recall_score(test_y, test_predictions)
test_f1 = f1_score(test_y, test_predictions)

print("\nTest Set Evaluation:")
print("Test AUC:", test_auc)
print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1 Score:", test_f1)
print(classification_report(test_y, test_predictions))


##############################################################
# 10. EXPORT RESULTS FOR TABLEAU PUBLIC (Optional)
##############################################################
"""
# Save test set predictions and metrics
results_df = test_x.copy()
results_df['Actual_Churn'] = test_y
results_df['Predicted_Churn'] = test_predictions
results_df['Predicted_Probability'] = test_probabilities
results_df.to_csv('test_results_for_tableau.csv', index=False)
print("Results saved to 'test_results_for_tableau.csv'.")

# Save model performance metrics
metrics = {
    'Test_AUC': [test_auc],
    'Test_Accuracy': [test_accuracy],
    'Test_Precision': [test_precision],
    'Test_Recall': [test_recall],
    'Test_F1': [test_f1]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('model_metrics_for_tableau.csv', index=False)
print("Metrics saved to 'model_metrics_for_tableau.csv'.")

# Save hyperparameter tuning results from GridSearchCV
tuning_results_df = pd.DataFrame(grid_search.cv_results_)
tuning_results_df.to_csv('tuning_results.csv', index=False)
print("Tuning results saved to 'tuning_results.csv'.")

# Save feature importances
importance_df.to_csv('feature_importances.csv', index=False)
print("Feature importances saved to 'feature_importances.csv'.")
"""

