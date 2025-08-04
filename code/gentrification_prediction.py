"""
Downstream Gentrification Prediction Pipeline

This script demonstrates three methods for using unsupervised building embeddings (from two years) to predict gentrification status:
    1. Feature concatenation: Concatenate past and current embeddings. In this file we have used embeddings from 2013 (past) and 2020 (current) for our long-span configuration.
        **Need to do the same for short-span configuration.
    2. Cross-attention: Use a shallow, trainable cross-attention to combine embeddings.
    3. Cross-attention + supervised contrastive loss: Fine-tune the post-attention module with a supervised contrastive objective.
For each approach, both Random Forest and Gradient Boosting classifiers are evaluated.

Requirements:
    - Embedding files (e.g., `building_embeddings.csv` for each year), generated form embedding_creation.py for each year
    - Label files with tract-level gentrification status , need to generate GT as the same way mentioned in paper
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# -------------- Data Loading Utilities --------------

def load_embeddings_labels(emb_path_2013, emb_path_2020, label_path_2013, label_path_2020):
    emb_2013 = pd.read_csv(emb_path_2013)
    emb_2020 = pd.read_csv(emb_path_2020)
    label_2013 = pd.read_csv(label_path_2013)
    label_2020 = pd.read_csv(label_path_2020)

    # Merge
    merged_2013 = pd.merge(emb_2013, label_2013, on='GEOID', how='inner')
    merged_2020 = pd.merge(emb_2020, label_2020, on='GEOID', how='inner')

    # Common GEOIDs
    common_tract_ids = sorted(list(set(merged_2013['GEOID']) & set(merged_2020['GEOID'])))
    merged_2013 = merged_2013[merged_2013['GEOID'].isin(common_tract_ids)]
    merged_2020 = merged_2020[merged_2020['GEOID'].isin(common_tract_ids)]

    # Embeddings and labels
    embed_cols = merged_2013.filter(regex=r'^embed_\d+$').columns.tolist()
    E_2013 = merged_2013[embed_cols].values
    E_2020 = merged_2020[embed_cols].values
    GEOIDs = merged_2013["GEOID"].values

    le = LabelEncoder()
    y = le.fit_transform(merged_2020["Gentrifying_status"].values)  # Use 2020 label for main task

    return E_2013, E_2020, y, GEOIDs

# -------------- Cross-Attention --------------

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
    def forward(self, E_2013, E_2020):
        E_2013 = E_2013.unsqueeze(1)
        E_2020 = E_2020.unsqueeze(1)
        cross_feats, attn_weights = self.cross_attn(query=E_2020, key=E_2013, value=E_2013)
        delta = E_2020 - E_2013
        combined = torch.cat([E_2020.squeeze(1), delta.squeeze(1), cross_feats.squeeze(1)], dim=-1)
        return combined, attn_weights

# -------------- SupConLoss --------------

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        norm = torch.norm(features, p=2, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        features = features / norm
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature
        anchor_dot_contrast = torch.clamp(anchor_dot_contrast, min=-10, max=10)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask - torch.eye(batch_size, device=features.device)
        exp_logits = torch.exp(logits) * mask
        exp_logits_sum = exp_logits.sum(1, keepdim=True)
        log_prob = logits - torch.log(exp_logits_sum + 1e-5)
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss

# -------------- Classifier for final prediction --------------

def train_classifiers(X_train, X_test, y_train, y_test, method_name):
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    }
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"\n{method_name} - {clf_name} results:")
        print(classification_report(y_test, y_pred, target_names=["Not Gentrified (0)", "Gentrified (1)"]))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not Gentrified (0)", "Gentrified (1)"],
                    yticklabels=["Not Gentrified (0)", "Gentrified (1)"])
        plt.title(f"{method_name} - {clf_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

# -------------- Concatenation Approach --------------

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def concatenation(E_2013, E_2020, y):
    X_delta = E_2020 - E_2013
    X = np.concatenate([E_2020, X_delta], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y )

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train and evaluate classifiers
    train_classifiers(X_train, X_test, y_train, y_test, method_name="Concatenation")


# --------------Cross-Attention --------------

def cross_attention(E_2013, E_2020, y, method_name="Cross-Attention"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    E_2013_train, E_2013_test, E_2020_train, E_2020_test, y_train, y_test = train_test_split(
        E_2013, E_2020, y, test_size=0.2, random_state=42, stratify=y
    )
  
    # Initialize the cross-attention model (random weights, no training)
    cross_att_model = CrossAttention(embed_dim=E_2013.shape[1]).to(device)

    # Use the model to generate features
    cross_att_model.eval()
    with torch.no_grad():
        train_combined, _ = cross_att_model(
            torch.tensor(E_2013_train, dtype=torch.float32).to(device),
            torch.tensor(E_2020_train, dtype=torch.float32).to(device)
        )
        test_combined, _ = cross_att_model(
            torch.tensor(E_2013_test, dtype=torch.float32).to(device),
            torch.tensor(E_2020_test, dtype=torch.float32).to(device)
        )
        X_train = train_combined.cpu().numpy()
        X_test = test_combined.cpu().numpy()

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train, y_train = smote.fit_resample(X_train, y_train)
  
    # Train and evaluate classifiers
    train_classifiers(X_train_res, X_test, y_train_res, y_test, method_name=method_name)


# -------------- Cross-Attention + SupConLoss --------------

def cross_attention_supcon(E_2013, E_2020, y, epochs=50):
  
  E_2013_train, E_2013_test, E_2020_train, E_2020_test, y_train, y_test = train_test_split(
        E_2013, E_2020, y, test_size=0.2, random_state=42, stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    E_2013_train_tensor = torch.tensor(E_2013_train, dtype=torch.float32).to(device)
    E_2020_train_tensor = torch.tensor(E_2020_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    
    E_2013_test_tensor = torch.tensor(E_2013_test, dtype=torch.float32).to(device)
    E_2020_test_tensor = torch.tensor(E_2020_test, dtype=torch.float32).to(device)
    # y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device) # Not needed for inference
    
    # Train cross-attention model using TRAIN data
    post_processor = PostHocCrossAttention(embed_dim=E_2013.shape[1]).to(device)
    contrastive_loss = SupConLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(post_processor.parameters(), lr=1e-4)
    train_dataset = TensorDataset(E_2013_train_tensor, E_2020_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        post_processor.train()
        total_loss = 0
        for E13, E20, labels in train_loader:
            combined, _ = post_processor(E13, E20)
            loss = contrastive_loss(combined, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg SupConLoss: {avg_loss:.4f}")
    
    # Generate combined embeddings for TRAIN and TEST sets using trained model
    post_processor.eval()
    with torch.no_grad():
        combined_train, _ = post_processor(E_2013_train_tensor, E_2020_train_tensor)
        combined_test, _ = post_processor(E_2013_test_tensor, E_2020_test_tensor)
    X_train = combined_train.cpu().numpy()
    X_test = combined_test.cpu().numpy()
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train and evaluate classifiers
    train_classifiers(X_train_res, X_test, y_train_res, y_test, method_name="Cross-Att + SupConLoss")

# -------------- Main --------------

if __name__ == "__main__":
    emb_path_2013 = "/path/to/31_building_embeddings_2013.csv"
    emb_path_2020 = "/path/to/31_building_embeddings_2020.csv"
    label_path_2013 = "/path/to/Gentrification_status_2023_Chicago.csv"
    label_path_2020 = "/path/to/Gentrification_status_2023(from 2020)_Chicago.csv"

    # Load embeddings and labels
    E_2013, E_2020, y, GEOIDs = load_embeddings_labels(emb_path_2013, emb_path_2020, label_path_2013, label_path_2020)

    print("\n Feature Concatenation")
    concatenation(E_2013, E_2020, y)

    print("\n Cross-Attention")
    cross_attention(E_2013, E_2020, y)

    print("\n Cross-Attention + Supervised Contrastive Loss")
    cross_attention_supcon(E_2013, E_2020, y, epochs=50)
