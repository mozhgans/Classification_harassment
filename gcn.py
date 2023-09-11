# -*- coding: utf-8 -*-
"""harassment classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12N3tS-t74Lk4PdrnULGiDuhnrUZmXinQ
"""

!pip install scikit-learn
!pip install sentence-transformers

pip install torch torchvision
pip install transformers
pip install sentence-transformers
pip install scikit-learn
pip install pandas
pip install numpy

import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax
from torch_geometric.nn import GCNConv
from sentence_transformers import SentenceTransformer

class GCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNClassifier, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)
        self.sbert_model = SentenceTransformer('your_sbert_model_name_or_path')  # Replace with the path to your SBERT model

    def forward(self, x, edge_index):
        # Use SBERT embeddings as node features
        x = self.sbert_model.encode(x)
        x = relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return softmax(x, dim=1)

# Example usage of the GCN model
input_dim = 768  # Dimension of SBERT embeddings (replace with the actual dimension)
hidden_dim = 64
output_dim = 3  # 3 classes: indirect, physical, sexual harassment

# Example adjacency matrix (randomly generated for illustration purposes)
# Replace this with your actual adjacency matrix
edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)

# Example tweet texts (replace this with your actual tweet data)
tweets = ["tweet text 1", "tweet text 2", "tweet text 3", "tweet text 4", "tweet text 5"]

# Create the GCN model
gcn_model = GCNClassifier(input_dim, hidden_dim, output_dim)

# Forward pass through the GCN model
predictions = gcn_model(tweets, edge_index)

# Training and evaluation code for the GCN model should be added based on your specific dataset and task.
# The edge_index represents the graph structure (relationships between tweets).
# The 'tweets' list contains the tweet texts, and you should preprocess them to get SBERT embeddings as node features.
# The 'predictions' will contain the model's output probabilities for each class for each tweet.