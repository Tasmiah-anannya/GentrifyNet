# GentrifyNet
This repository includes essential scripts of the proposed framework GentrifyNet in our paper: GentrifyNet: Predicting Gentrification at Speed of Urban Change through Region Representation Learning from Building Footprint Dynamics

# Overview
Data Preprocessing: Rasterizes building footprints, encodes building types, and combines tract-level CSVs into a unified dataset.
Embedding Model: Dual-scale transformer encoder for sequence-to-embedding learning. Supports ablation to a single encoder for comparison.
Feature Ablation: Embeddings can be trained with or without specific feature groups (e.g., remove building type).
Downstream Tasks: Embedding-based tract-level classification, including temporal change and cross-attention for capturing dynamics.

# Pipeline
**1. Preprocessing:**
  - rasterization.py and resnet.py
  - Combine tract-level CSV files into a single CSV with required columns (combined_building_list.csv).
  - Rasterize building polygons into images and then use pre-trained Resnet-18 to encode building shape, type, height, and rotation.
  - Generates:
        -building_features.npy (per-building feature vectors)
        -building_rot_type.npz (rotation, height, one-hot type)
        -combined_building_list.csv
    
**2. Embedding Learning**
  - dual_scale_encoder.py
  - Train DualScaleEmbedder (or SingleScaleEmbedder for ablation) on tract-wise building sequences using self-supervised contrastive loss.
  - Generates:
        -Tract-level embeddings: <CITY>_<YEAR>_building_embeddings.csv

**3. Gentrification Prediction**
  - gentrification_prediction.py
  - Merge embeddings with tract-level labels (e.g., gentrification status).
  - Three approaches for using embeddings as features:
      a. Concatenation: [E_2020, E_2020-E_2013]
      b. Cross-Attention: Combines past/current embeddings
      c. Cross-Attention + Supervised Contrastive Loss (optional)
  - Train and evaluate classifiers: Random Forest (RF) and Gradient Boosting (GB).

**4. Feature Ablation Study**
  - To evaluate feature importance, retrain the embedding model with selected features removed (e.g., without type) from dual_scale_encoder.py and re-run the full pipeline.

**5. Architecture Ablation Study**
   - single_encoder.py 
   - Replace DualScaleEmbedder with SingleScaleEmbedder to measure the effect of dual vs. single transformer encoders.

# Note
    - Preprocessing and embedding scripts are modular and can be adapted for any city/year.
    - The pipeline is generic; update CITY, YEAR, and folder paths as needed.
    - For large-scale ablation studies, automate with simple batch scripts looping over feature/architecture configs.
