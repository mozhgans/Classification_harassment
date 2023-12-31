# -*- coding: utf-8 -*-
"""harassment classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12N3tS-t74Lk4PdrnULGiDuhnrUZmXinQ
"""

!pip install scikit-learn
!pip install sentence-transformers

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
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

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'])

# Convert text data to sequences
sequences = tokenizer.texts_to_sequences(data['text'])

# Pad sequences to a fixed length for the bi-LSTM model
max_length = 50  # Adjust the max_length as needed
sequences_padded = pad_sequences(sequences, maxlen=max_length, padding='post')

# Encode the class labels for binary classification
# For multiclass classification, no encoding is needed as Keras handles it internally
label_encoder = LabelEncoder()
y_binary = label_encoder.fit_transform(data['label'])

# Split data into training and test sets for binary classification
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(sequences_padded, y_binary, test_size=0.2, random_state=42)

# Initialize and train the bi-LSTM model for binary classification
num_classes_binary = 1  # For binary classification, output has only one class (harassment vs not-harassment)

model_binary = Sequential()
model_binary.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_length))
model_binary.add(Bidirectional(LSTM(128)))
model_binary.add(Dense(num_classes_binary, activation='sigmoid'))

model_binary.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_binary.summary()

model_binary.fit(X_train_binary, y_train_binary, epochs=10, batch_size=32, validation_data=(X_test_binary, y_test_binary))

# Predict and evaluate the bi-LSTM model for binary classification
y_pred_binary = model_binary.predict_classes(X_test_binary).flatten()
print("Binary Classification Results:")
print("Accuracy:", accuracy_score(y_test_binary, y_pred_binary))
print("Classification Report:")
print(classification_report(y_test_binary, y_pred_binary))

# For multiclass classification, follow a similar approach as above
# Encode the class labels for multiclass classification
label_encoder_multiclass = LabelEncoder()
y_multiclass = label_encoder_multiclass.fit_transform(data['label'])

# Split data into training and test sets for multiclass classification
X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass = train_test_split(sequences_padded, y_multiclass, test_size=0.2, random_state=42)

# Initialize and train the bi-LSTM model for multiclass classification
num_classes_multiclass = len(label_encoder_multiclass.classes_)

model_multiclass = Sequential()
model_multiclass.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_length))
model_multiclass.add(Bidirectional(LSTM(128)))
model_multiclass.add(Dense(num_classes_multiclass, activation='softmax'))

model_multiclass.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_multiclass.summary()

model_multiclass.fit(X_train_multiclass, y_train_multiclass, epochs=10, batch_size=32, validation_data=(X_test_multiclass, y_test_multiclass))

# Predict and evaluate the bi-LSTM model for multiclass classification
y_pred_multiclass = model_multiclass.predict_classes(X_test_multiclass).flatten()
print("\nMulticlass Classification Results:")
print("Accuracy:", accuracy_score(y_test_multiclass, y_pred_multiclass))
print("Classification Report:")
print(classification_report(y_test_multiclass, y_pred_multiclass))