# -*- coding: utf-8 -*-
"""harassment classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12N3tS-t74Lk4PdrnULGiDuhnrUZmXinQ
"""

!pip install scikit-learn
!pip install sentence-transformers

pip install fasttext

import numpy as np
import pandas as pd
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load your tweet data into a pandas DataFrame with 'text' and 'label' columns
# For binary classification (harassment vs not-harassment)
# For multiclass classification (indirect harassment, physical harassment, sexual harassment)

# Sample data (Replace with your actual data)
data = pd.DataFrame({
    'text': ['This tweet is harassment', 'I love this weather', 'Indirect harassment here', 'Physical harassment incident', 'Sexual harassment in the workplace'],
    'label': ['harassment', 'not-harassment', 'indirect harassment', 'physical harassment', 'sexual harassment']
})

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Write training and test data to files for FastText format
train_file = 'train.txt'
test_file = 'test.txt'

with open(train_file, 'w') as f:
    for text, label in zip(X_train, y_train):
        f.write(f'__label__{label} {text}\n')

with open(test_file, 'w') as f:
    for text, label in zip(X_test, y_test):
        f.write(f'__label__{label} {text}\n')

# Train the FastText model for binary classification
model_binary = fasttext.train_supervised(input=train_file, epoch=10)

# Test the FastText model for binary classification
y_pred_binary = model_binary.predict(X_test.tolist())[0]
y_pred_binary = [label[0].replace('__label__', '') for label in y_pred_binary]

print("Binary Classification Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))

# Train the FastText model for multiclass classification
model_multiclass = fasttext.train_supervised(input=train_file, epoch=10, label_prefix='__label__')

# Test the FastText model for multiclass classification
y_pred_multiclass = model_multiclass.predict(X_test.tolist())[0]
y_pred_multiclass = [label[0].replace('__label__', '') for label in y_pred_multiclass]

print("\nMulticlass Classification Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_multiclass))
print("Classification Report:")
print(classification_report(y_test, y_pred_multiclass))