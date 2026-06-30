# =====================================
# IMPORT LIBRARIES
# =====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    ConfusionMatrixDisplay, roc_curve, auc
)

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("creditcard.csv")
target = 'Class'

# =====================================
# TIME BUCKET (ONLY FOR ANALYSIS)
# =====================================
df['time_bucket'] = pd.qcut(df['Time'], q=10, duplicates='drop')

# =====================================
# SELECT NUMERIC FEATURES
# =====================================
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove(target)
num_cols.remove('Time')

# =====================================
# BINNING (ONCE)
# =====================================
df_woe = df.copy()
for col in num_cols:
    df_woe[col] = pd.qcut(df[col], 10, duplicates='drop')

# =====================================
# WOE + IV FUNCTION
# =====================================
def calculate_woe_iv(data, feature, target):
    temp = pd.crosstab(data[feature], data[target])

    if temp.shape[1] < 2:
        return None, 0

    temp.columns = ['Good', 'Bad']
    temp['Dist_Good'] = temp['Good'] / temp['Good'].sum()
    temp['Dist_Bad'] = temp['Bad'] / temp['Bad'].sum()

    temp['WOE'] = np.log((temp['Dist_Good'] + 1e-6) / (temp['Dist_Bad'] + 1e-6))
    temp['IV'] = (temp['Dist_Good'] - temp['Dist_Bad']) * temp['WOE']

    return temp, temp['IV'].sum()

# =====================================
# CALCULATE WOE & IV
# =====================================
woe_dict, iv_dict = {}, {}

for col in num_cols:
    woe_table, iv = calculate_woe_iv(df_woe, col, target)
    if woe_table is not None:
        woe_dict[col] = woe_table['WOE'].to_dict()
        iv_dict[col] = iv

# Save IV
iv_df = pd.DataFrame(iv_dict.items(), columns=['Feature', 'IV'])
iv_df = iv_df.sort_values(by='IV', ascending=False)
iv_df.to_csv("iv_values.csv", index=False)

# =====================================
# WOE TRANSFORMATION
# =====================================
for col in woe_dict:
    df_woe[col] = df_woe[col].map(woe_dict[col])

df_woe = df_woe.fillna(0)

# =====================================
# FEATURE SELECTION
# =====================================
selected_features = [col for col, iv in iv_dict.items() if iv > 0.02]

X = df_woe[selected_features]
y = df_woe[target]

# =====================================
# SAVE TRANSFORMED DATASET WITH IV
# =====================================
transformed_df = df_woe[selected_features + [target]].copy()

for col in selected_features:
    transformed_df[col + "_IV"] = iv_dict[col]

transformed_df.to_csv("woe_transformed_with_iv.csv", index=False)

# =====================================
# TRAIN TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# MODELS
# =====================================
log_model = LogisticRegression(max_iter=1000)
dt_model = DecisionTreeClassifier(max_depth=4)

log_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# =====================================
# METRICS FUNCTION
# =====================================
def get_metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred)
    )

# Predictions
y_pred_log = log_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)

log_metrics = get_metrics(y_test, y_pred_log)
dt_metrics = get_metrics(y_test, y_pred_dt)

# =====================================
# CROSS VALIDATION
# =====================================
def cross_val(model):
    acc = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    f1 = cross_val_score(model, X, y, cv=5, scoring='f1').mean()
    return acc, f1

log_cv = cross_val(log_model)
dt_cv = cross_val(dt_model)

# =====================================
# SAVE FINAL METRICS
# =====================================
metrics_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [log_metrics[0], dt_metrics[0]],
    'F1 Score': [log_metrics[1], dt_metrics[1]],
    'Precision': [log_metrics[2], dt_metrics[2]],
    'Recall': [log_metrics[3], dt_metrics[3]],
    'Avg Accuracy (CV)': [log_cv[0], dt_cv[0]],
    'Avg F1 Score (CV)': [log_cv[1], dt_cv[1]]
})

metrics_df.to_csv("final_model_metrics.csv", index=False)

# =====================================
# SAVE PREDICTIONS
# =====================================
results = pd.DataFrame({
    'Actual': y_test,
    'Logistic_Prediction': y_pred_log,
    'DecisionTree_Prediction': y_pred_dt
})

results.to_csv("model_predictions.csv", index=False)

# =====================================
# CONFUSION MATRIX
# =====================================
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_log)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# =====================================
# ROC CURVE
# =====================================
y_prob = log_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =====================================
# DECISION TREE PLOT
# =====================================
plt.figure(figsize=(20, 15))
plot_tree(dt_model, filled=True, feature_names=X.columns, max_depth=4)
plt.title("Decision Tree")
plt.show()

# =====================================
# ROLL RATE ANALYSIS
# =====================================
roll = pd.crosstab(df['time_bucket'], df[target])
roll['Total'] = roll.sum(axis=1)
roll['Bad Rate'] = roll[1] / roll['Total']
roll.to_csv("roll_rate.csv")

# =====================================
# VINTAGE ANALYSIS
# =====================================
vintage = df.groupby('time_bucket')[target].agg(['count', 'sum', 'mean']).reset_index()
vintage.columns = ['Cohort', 'Total Accounts', 'Bad Accounts', 'Bad Rate']
vintage.to_csv("vintage_analysis.csv", index=False)