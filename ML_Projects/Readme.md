
# 🛒 Product Recommendation System

> An end-to-end **Item-Item Collaborative Filtering** recommendation engine with **DBSCAN clustering analysis**, deployed as an interactive **Streamlit** web application.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Exploratory Data Analysis](#1-exploratory-data-analysis)
  - [Feature Engineering](#2-feature-engineering)
  - [Outlier Detection](#3-outlier-detection)
  - [Clustering Analysis](#4-clustering-analysis)
  - [Recommendation Engine](#5-recommendation-engine)
- [Model Evaluation](#model-evaluation)
- [Streamlit App](#streamlit-app)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

This project builds a **Product Recommendation System** from scratch using collaborative filtering techniques. It processes user-product rating data, performs comprehensive exploratory analysis, engineers meaningful features, applies multiple clustering algorithms to segment user-product interactions, and ultimately delivers personalized product recommendations through an interactive web interface.

### Problem Statement
Given a dataset of user ratings for products, build a system that can:
1. **Recommend similar products** based on item-item collaborative filtering (cosine similarity)
2. **Segment products** into meaningful clusters using unsupervised learning
3. **Predict ratings** a user might give to an unrated product
4. **Deploy** the system as an accessible web application

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Raw Data Layer                          │
│                      (rating_short.csv)                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Processing Pipeline                     │
│  ┌──────────┐  ┌──────────────────┐  ┌───────────────────────┐ │
│  │   EDA    │→ │ Feature Engineer │→ │  Outlier Detection    │ │
│  │          │  │  - Avg Ratings   │  │  (Isolation Forest)   │ │
│  │          │  │  - Deviations    │  │                       │ │
│  │          │  │  - Log Transforms│  │                       │ │
│  └──────────┘  └──────────────────┘  └───────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   K-Means    │ │ Hierarchical │ │   DBSCAN     │
│  Clustering  │ │  Clustering  │ │  Clustering  │
│  (K=4)       │ │  (Ward, K=4) │ │  (Tuned)     │
│  Sil: 0.303  │ │  Sil: 0.302  │ │  Sil: 0.468  │
└──────────────┘ └──────────────┘ └──────┬───────┘
                                         │ Best Model
              ┌──────────────────────────┘
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Recommendation Engine                          │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐ │
│  │  Item-Item Collaborative│  │   Rating Prediction          │ │
│  │  Filtering              │  │   (Weighted Cosine Sim)      │ │
│  │  (Cosine Similarity)    │  │                              │ │
│  └─────────────────────────┘  └──────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Deployment Layer                            │
│              Streamlit Web Application (app.py)                 │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐│
│  │ recommendation_  │  │  train_item_user_matrix.npz          ││
│  │ artifacts.pkl    │  │  (Sparse Matrix Representation)      ││
│  └──────────────────┘  └──────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Comprehensive EDA** | Distribution analysis, correlation heatmaps, PPS matrix, scatter/violin/pie charts |
| 🛠️ **Feature Engineering** | Rating deviations, log transformations, product & user aggregated statistics |
| 🚫 **Outlier Detection** | Isolation Forest with 1% contamination to clean noisy ratings |
| 📊 **Multi-Algorithm Clustering** | K-Means, Agglomerative Hierarchical, and DBSCAN — compared via Silhouette Scores |
| 🤝 **Item-Item Collaborative Filtering** | Cosine similarity on sparse item-user matrices for scalable recommendations |
| ⭐ **Rating Prediction** | Predicts user ratings for unrated products using weighted similarity scores |
| 🌐 **Interactive Streamlit App** | Real-time product recommendations with adjustable similarity thresholds |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | scikit-learn (K-Means, DBSCAN, Agglomerative, Isolation Forest) |
| **Similarity Computation** | SciPy (sparse matrices), scikit-learn (cosine similarity) |
| **Visualization** | Matplotlib, Seaborn |
| **Feature Analysis** | PPS (Predictive Power Score) |
| **Deployment** | Streamlit |
| **Serialization** | Pickle, SciPy `save_npz` / `load_npz` |

---

## Project Structure

```
Product-Recommendation-System/
│
├── Product_Recommendation_System.ipynb   # Full Jupyter Notebook (EDA → Modeling → Deployment)
├── product_recommendation_system.py      # Python script version of the notebook
├── app.py                                # Streamlit web application
├── rating_short.csv                      # Dataset (user-product ratings)
├── recommendation_artifacts.pkl          # Serialized model artifacts (mappings, scaler, clusters)
├── train_item_user_matrix.npz            # Sparse item-user matrix for similarity computation
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/uavinash15/Product-Recommendation-System.git
cd Product-Recommendation-System
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage

### Using the Streamlit App

1. **Enter a Product ID** — Type or paste a product ID in the search box (e.g., `B00004HYS6`)
2. **Adjust Parameters:**
   - **Number of Recommendations** — Slider (1–20) to control how many similar products to return
   - **Similarity Threshold** — Select from `0.0, 0.1, 0.3, 0.5, 0.7` to filter by minimum cosine similarity
3. **Click "Get Recommendations"** — View similar products ranked by similarity score along with DBSCAN cluster information

### Sample Product IDs

| Product ID | Notes |
|-----------|-------|
| `B00004HYS6` | High similarity matches (≥ 0.7) |
| `B00004R8VM` | High similarity matches (≥ 0.7) |
| `B0074BW614` | General recommendations |
| `B000001OM5` | General recommendations |

---

## Methodology

### 1. Exploratory Data Analysis

- **Dataset**: User-product rating data with columns `userid`, `productid`, `rating` (1–5 scale)
- **Key Findings:**
  - No missing values or data type inconsistencies
  - No outliers detected in the rating distribution (confirmed via box plot)
  - Very weak correlation (~0.03) between number of ratings and average rating for both products and users
  - Rating distribution is right-skewed with the most frequent rating being **5.0**

### 2. Feature Engineering

| Feature | Formula | Purpose |
|---------|---------|---------|
| `product_avg_rating` | Mean rating per product | Product-level quality signal |
| `user_avg_rating` | Mean rating per user | User-level bias indicator |
| `rating_deviation_from_product_avg` | `rating - product_avg_rating` | Captures unique user-product preferences |
| `rating_deviation_from_user_avg` | `rating - user_avg_rating` | Normalizes individual rating behavior |
| `product_num_ratings_log` | `log(1 + product_num_ratings)` | Reduces skewness of count features |
| `user_num_ratings_log` | `log(1 + user_num_ratings)` | Reduces skewness of count features |

### 3. Outlier Detection

- **Algorithm**: Isolation Forest (`contamination=0.01`, `n_estimators=100`)
- **Result**: ~1% of data points removed as anomalous ratings
- **Rationale**: Removes noise to improve model robustness and similarity quality

### 4. Clustering Analysis

Three clustering algorithms were applied and compared:

| Algorithm | Clusters | Silhouette Score | Notes |
|-----------|----------|------------------|-------|
| **K-Means** | 4 | 0.303 | Optimal K=4 via Elbow Method |
| **Hierarchical (Ward)** | 4 | 0.302 | Dendrogram confirms 4 clusters |
| **DBSCAN (Tuned)** | — | **0.468** ✅ | Best score; `eps=1.0`, `min_samples=5` |

> **Winner**: Tuned DBSCAN achieved the highest Silhouette Score, indicating the best-defined clusters. The tuned DBSCAN labels are used in the deployed Streamlit app.

### 5. Recommendation Engine

#### Item-Item Collaborative Filtering
1. Build a **sparse user-product matrix** (CSR format) to handle the large, sparse rating space efficiently
2. Transpose to get the **item-user matrix**
3. Compute **cosine similarity** between all item pairs
4. For a target product, rank all other products by similarity score and return the top-K

#### Rating Prediction
- Uses **weighted average** of ratings from similar items the user has rated
- Falls back to the user's average rating for cold-start scenarios

---

## Model Evaluation

The recommendation system was evaluated using an 80/20 train-test split:

### Regression Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **MAE** | ~1.09 | Average prediction error of ~1 rating point |
| **RMSE** | ~1.41 | Penalizes larger errors; reasonable for a 1-5 scale |

### Classification Metrics (Threshold: Rating ≥ 5.0 = "Liked")

| Metric | Score |
|--------|-------|
| **Precision** | ~0.62 |
| **Recall** | ~0.72 |
| **F1-Score** | ~0.67 |

### Known Limitations & Areas for Improvement

| Challenge | Description |
|-----------|-------------|
| ❄️ **Cold Start** | Users or products with very few ratings receive `None` predictions |
| 📉 **Sparsity** | The user-product matrix is highly sparse (~99.9%), reducing similarity signal quality |
| 🔄 **Scalability** | Full item-similarity matrix (~40K × 40K) does not fit in memory; the app uses the train subset |

---

## Streamlit App

The deployed Streamlit application provides:

- **🔎 Product Search** — Enter any product ID from the dataset
- **📊 DBSCAN Cluster Info** — View which cluster the product belongs to along with key metrics
- **📋 Ranked Recommendations** — Table of similar products sorted by cosine similarity score
- **⚙️ Adjustable Threshold** — Control the minimum similarity score for recommendations
- **ℹ️ Sidebar** — Sample product IDs for quick testing

---

## Future Improvements

- [ ] **Matrix Factorization (SVD/ALS)** — Better latent factor representations for improved predictions
- [ ] **Hybrid Recommender** — Combine collaborative filtering with content-based features
- [ ] **Deep Learning** — Neural Collaborative Filtering (NCF) for complex interaction patterns
- [ ] **Real-Time Updates** — Incremental model updating as new ratings arrive
- [ ] **A/B Testing Framework** — Measure recommendation quality with live user feedback
- [ ] **Cloud Deployment** — Deploy on Streamlit Cloud / AWS / GCP for public access

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <b>⭐ If you found this project helpful, please consider giving it a star!</b>
</p>
