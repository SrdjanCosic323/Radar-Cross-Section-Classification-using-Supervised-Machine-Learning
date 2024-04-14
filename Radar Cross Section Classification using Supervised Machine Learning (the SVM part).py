# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Set font size
plt.rcParams.update({'font.size': 16})

# Load the data
dataset = sio.loadmat('data_supervised_srdjan.mat')['data'].T
target = np.argmax(sio.loadmat('t_supervised_srdjan.mat')['t'], axis=0)

# Split-out validation dataset
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(dataset, target, test_size=validation_size, random_state=seed)

# Task 4: Basic Classifier with default settings

# Support Vector Machine (SVM)
num_folds = 7
scoring = 'accuracy'
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
model = SVC()
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
accuracy = np.mean(cv_results) * 100  # Convert to percentage
msg = "SVM: %.2f%% (%f)" % (accuracy, np.std(cv_results))
print(msg)

# Task 4: Standardize the dataset
pipelines = []
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])))

results = []
names = []

for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    accuracy = np.mean(cv_results) * 100  # Convert to percentage
    msg = "%s: %.2f%% (%f)" % (name, accuracy, np.std(cv_results))
    print(msg)

# Store accuracies
no_scale_accuracies = []
scale_accuracies = []

# First, calculate accuracies for models without scaling
model = SVC()
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
accuracy = np.mean(cv_results) * 100  # Convert to percentage
no_scale_accuracies.append(accuracy)

# Then, calculate accuracies for models with scaling
model = Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
accuracy = np.mean(cv_results) * 100  # Convert to percentage
scale_accuracies.append(accuracy)

# Plotting
bar_width = 0.35
index = np.arange(len(no_scale_accuracies))

fig, ax = plt.subplots()
rects1 = ax.bar(index, no_scale_accuracies, bar_width, label='Without Scaling')
rects2 = ax.bar(index + bar_width, scale_accuracies, bar_width, label='With Scaling')

ax.set_xlabel('Models')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy Comparison: With and Without Scaling')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['SVM'])
ax.legend()

# Task 5: Hyperparameter tuning by Gridsearch

# Define parameter grid for SVM
param_grid_svm = {
    'C': [0.1, 0.5, 1, 1.5, 2, 2.5, 3],
}

# Perform Grid Search
best_params = {}

# SVM
grid_search = GridSearchCV(SVC(), param_grid_svm, cv=num_folds, scoring=scoring, n_jobs=-1)
grid_result = grid_search.fit(X_train, Y_train)
best_params['SVM'] = grid_result.best_params_

print("Best Parameters for SVM:", best_params['SVM'])

# Retrain and evaluate SVM with tuned hyperparameters
tuned_model = SVC(**best_params['SVM'])
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
cv_results = cross_val_score(tuned_model, X_train, Y_train, cv=kfold, scoring=scoring)
accuracy = np.mean(cv_results) * 100
print(f"SVM (Tuned): {accuracy:.2f}%")
from sklearn.model_selection import learning_curve, validation_curve

# Task 7: Overfitting/Underfitting Analysis for SVM and C Parameter

# Function to evaluate the SVM model with different C values
def evaluate_svm_model(c_value, X_train, Y_train, X_validation, Y_validation):
    # Initialize SVM model with the given C value
    model = SVC(C=c_value)

    # Train the model
    model.fit(X_train, Y_train)

    # Calculate training accuracy
    train_accuracy = model.score(X_train, Y_train) * 100
    # Calculate validation accuracy
    validation_accuracy = model.score(X_validation, Y_validation) * 100

    return train_accuracy, validation_accuracy

# Define a list of C values to iterate over
c_values = [0.1, 0.5, 1, 1.5, 2, 2.5, 3]

# Iterate over each C value and print the results
print("SVM Model Analysis:")
for c_value in c_values:
    train_accuracy, validation_accuracy = evaluate_svm_model(c_value, X_train, Y_train, X_validation, Y_validation)
    print(f"SVM with C={c_value}: Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {validation_accuracy:.2f}%")


# Additional Task: Bagging and Cross-Validation for Support Vector Machines

from sklearn.ensemble import BaggingClassifier

# Additional Task: Bagging with Support Vector Machines
bagging_svm_model = BaggingClassifier(base_estimator=SVC(**best_params['SVM']), n_estimators=10, random_state=seed)
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
cv_results_bagging_svm = cross_val_score(bagging_svm_model, X_train, Y_train, cv=kfold, scoring=scoring)
accuracy_bagging_svm = np.mean(cv_results_bagging_svm) * 100
print(f"SVM with Bagging: Accuracy: {accuracy_bagging_svm:.2f}%")

# Evaluate Bagging with cross-validation
bagging_cv_results = cross_val_score(bagging_svm_model, X_train, Y_train, cv=kfold, scoring=scoring)
bagging_cv_accuracy = np.mean(bagging_cv_results) * 100
print(f"SVM with Bagging and Cross-Validation: Accuracy: {bagging_cv_accuracy:.2f}%")



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.svm import SVC
import seaborn as sns

# Assuming X_train, Y_train, X_validation, Y_validation are already defined

# Binarize the output for ROC and Precision-Recall curves
Y_bin_train = label_binarize(Y_train, classes=np.unique(Y_train))
Y_bin_validation = label_binarize(Y_validation, classes=np.unique(Y_train))
n_classes = Y_bin_train.shape[1]

# Initialize and train the final SVM model with RBF kernel and C=2.5
final_svm = SVC(C=2.5, kernel='rbf', probability=True)
final_svm.fit(X_train, Y_train)

# Predict probabilities for ROC and Precision-Recall curves using cross-validation
Y_proba = cross_val_predict(final_svm, X_validation, Y_validation, method='predict_proba')

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_bin_validation[:, i], Y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_bin_validation.ravel(), Y_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve for SVM')
plt.legend(loc="lower right")
plt.show()

# Compute Precision-Recall curve and area for each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_bin_validation[:, i], Y_proba[:, i])
    average_precision[i] = auc(recall[i], precision[i])

# Plot Precision-Recall curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(recall[i], precision[i], label=f'Precision-Recall curve of class {i} (area = {average_precision[i]:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multiclass Precision-Recall Curve for SVM')
plt.legend(loc="lower left")
plt.show()

# Predict the classes on the validation set
Y_pred = final_svm.predict(X_validation)

# Compute the confusion matrix
cm = confusion_matrix(Y_validation, Y_pred)

# Plot the confusion matrix using Seaborn's heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')  # fmt='g' to avoid scientific notation
plt.title('Confusion Matrix for SVM')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()