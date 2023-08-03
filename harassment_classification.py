# -*- coding: utf-8 -*-
"""harassment classification.ipynb
"""

!pip install scikit-learn
!pip install sentence-transformers

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load your tweet data into a pandas DataFrame with 'text' and 'label' columns
# For binary classification (harassment vs not-harassment)
# For multiclass classification (indirect harassment, physical harassment, sexual harassment)

# Sample data (Replace with your actual data)
data = pd.DataFrame({
    'text': ['This tweet is harassment', 'I love this weather', 'Indirect harassment here', 'Physical harassment incident', 'Sexual harassment in the workplace'],
    'label': ['harassment', 'not-harassment', 'indirect harassment', 'physical harassment', 'sexual harassment']
})

# Feature extraction using Sentence-BERT
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
X = model.encode(data['text'])

# For binary classification
# Label mapping for binary classification
label_mapping_binary = {'harassment': 1, 'not-harassment': 0}
y_binary = data['label'].map(label_mapping_binary)

# Split data into training and test sets for binary classification
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Initialize and train binary classification models
lr_model_binary = LogisticRegression()
svm_model_binary = SVC(kernel='linear')

lr_model_binary.fit(X_train_binary, y_train_binary)
svm_model_binary.fit(X_train_binary, y_train_binary)

# Predict and evaluate binary classification models
y_pred_lr_binary = lr_model_binary.predict(X_test_binary)
y_pred_svm_binary = svm_model_binary.predict(X_test_binary)

print("Binary Classification Results:")
print("Logistic Regression Accuracy:", accuracy_score(y_test_binary, y_pred_lr_binary))
print("SVM Accuracy:", accuracy_score(y_test_binary, y_pred_svm_binary))

# For multiclass classification
# Label mapping for multiclass classification
label_mapping_multiclass = {'indirect harassment': 0, 'physical harassment': 1, 'sexual harassment': 2}
y_multiclass = data['label'].map(label_mapping_multiclass)

# Split data into training and test sets for multiclass classification
X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass = train_test_split(X, y_multiclass, test_size=0.2, random_state=42)

# Initialize and train multiclass classification models
lr_model_multiclass = LogisticRegression(multi_class='ovr')  # One-vs-Rest strategy
svm_model_multiclass = SVC(kernel='linear')

lr_model_multiclass.fit(X_train_multiclass, y_train_multiclass)
svm_model_multiclass.fit(X_train_multiclass, y_train_multiclass)

# Predict and evaluate multiclass classification models
y_pred_lr_multiclass = lr_model_multiclass.predict(X_test_multiclass)
y_pred_svm_multiclass = svm_model_multiclass.predict(X_test_multiclass)

print("\nMulticlass Classification Results:")
print("Logistic Regression Accuracy:", accuracy_score(y_test_multiclass, y_pred_lr_multiclass))
print("SVM Accuracy:", accuracy_score(y_test_multiclass, y_pred_svm_multiclass))

# Classification report for multiclass classification
print("\nClassification Report:")
print(classification_report(y_test_multiclass, y_pred_lr_multiclass))