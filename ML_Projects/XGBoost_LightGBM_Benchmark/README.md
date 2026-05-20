# 🚢 Titanic Survival — XGBoost vs LightGBM Benchmark

A Streamlit web application that benchmarks **XGBoost** and **LightGBM** on the classic Titanic survival prediction dataset.

## 🎯 Features

- **Interactive hyper-parameter tuning** — adjust trees, depth, learning rate, and test split via sidebar
- **Automated preprocessing** — handles missing values, encodes categorical features
- **6 evaluation metrics** — Accuracy, Precision, Recall, F1 Score, ROC AUC, Training Time
- **Rich visualizations** — ROC curves, confusion matrices, feature importance charts
- **Side-by-side comparison** — see how XGBoost and LightGBM stack up

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/uavinash15/DataScience-.git
cd DataScience-/ML_Projects/XGBoost_LightGBM_Benchmark

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📂 Project Structure

```
XGBoost_LightGBM_Benchmark/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── Titanic_train.csv       # Training dataset (891 samples)
├── Titanic_test.csv        # Test dataset (418 samples)
├── .streamlit/
│   └── config.toml         # Dark theme configuration
├── .gitignore
└── README.md
```

## 📊 Dataset

The classic [Titanic dataset](https://www.kaggle.com/c/titanic) with features:
- **Pclass** — Passenger class (1, 2, 3)
- **Sex** — Gender
- **Age** — Age in years
- **SibSp** — # of siblings/spouses aboard
- **Parch** — # of parents/children aboard
- **Fare** — Ticket fare
- **Embarked** — Port of embarkation (C, Q, S)

## 🛠️ Tech Stack

- **Streamlit** — Web UI framework
- **XGBoost** — Gradient boosting (XGB)
- **LightGBM** — Light gradient boosting (LGB)
- **scikit-learn** — Preprocessing & metrics
- **Matplotlib** — Visualizations

## 📜 License

This project is for educational purposes.
