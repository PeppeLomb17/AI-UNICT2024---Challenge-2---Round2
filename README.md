# AI-UNICT2024---Challenge-2---Round2
The objective of the competition is to train a deep learning model to perform image classification on a dataset with eight classes. The main difficulty of the competition lies in the presence of a domain shift between training and test data, as you can see by inspecting the dataset.


# ViT + KMeans for Background-Invariant Image Classification

**Author:** Giuseppe Lombardia  
**Course:** Deep Learning – MSc in Data Science for Management, University of Catania  


---

## 1. Overview
This repository presents a complete pipeline for **unsupervised image classification under domain shift**, combining a pretrained **Vision Transformer (ViT-B/16)** as feature extractor with **KMeans clustering** and **Hungarian matching** for optimal label assignment.

The goal is to achieve **background-invariant classification**, learning object-centered representations rather than background patterns.  
The project was developed within the **AI-UNICT 2024 Challenge (Round 2)** and is fully documented in the accompanying report.

---

## 2. Motivation
The dataset exhibits a strong **domain shift**:
- Training set: images with heterogeneous and cluttered backgrounds.
- Test set: images with uniform backgrounds.

Supervised models tend to overfit to textures and colors of the background.  
This project proposes an **unsupervised, feature-based approach** leveraging Vision Transformer embeddings, clustering, and optimal class assignment.

---

## 3. Pipeline Summary
1. **Crop training images** using bounding boxes to isolate each object.  
2. **Extract ViT-B/16 embeddings** (768-D) with frozen weights.  
3. **Compute class centroids** from the cropped training features.  
4. **Cluster test embeddings** with **KMeans (k = 8)**.  
5. **Match clusters to classes** via the **Hungarian algorithm** on centroid distances.  
6. **Evaluate multiple random seeds** to ensure cluster balance and compactness.

---

## 4. Repository Structure

vit-kmeans-image-classification/
│
├─ notebooks/
│   └─ ViT_+_KMeans.ipynb
│
├─ paper/
│   └─ Image_Classification_pro.pdf
│
├─ data/
│   ├─ raw/                # train, test, train.csv, sample_submission.csv
│   └─ processed/          # cropped images and features (after execution)
│
└─ results/
├─ figures/            # plots, t-SNE, centroid distances
└─ submissions/        # generated CSV predictions

---

## 5. Installation
```bash
pip install -r requirements.txt

Dependencies
	•	torch, torchvision, timm
	•	scikit-learn, scipy
	•	numpy, pandas
	•	matplotlib, umap-learn (optional)

---
## 6. How to Run

All the code is contained within the Jupyter notebook
