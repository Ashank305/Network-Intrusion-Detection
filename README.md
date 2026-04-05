# 🛡️ Network Intrusion Detection System

A Machine Learning-powered **Network Intrusion Detection System (NIDS)** that analyzes network traffic and classifies connections as **Normal** or **Attack** using a **Random Forest Classifier** trained on the **KDD Cup 1999 dataset**.

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## 📋 Table of Contents

- [About](#about)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)

---

## 📖 About

Network Intrusion Detection uses **machine learning** to analyze network traffic and identify malicious activities by distinguishing normal and abnormal patterns. Instead of manually defining rules, the ML model learns patterns from data to automatically classify network connections.

### Types of Attacks Detected

| Attack Type | Description |
|-------------|-------------|
| **DoS** | Denial of Service — flooding the server |
| **Probe** | Scanning for vulnerabilities |
| **R2L** | Remote to Local — unauthorized remote access |
| **U2R** | User to Root — privilege escalation |

---

## 🏗️ Architecture

```
Data Collection → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction
     (KDD)          (Encoding)        (Scaling)        (Random Forest)   (Metrics)    (Normal/Attack)
```

**Pipeline Steps:**
1. **Data Collection** — KDD Cup 1999 dataset (41 features)
2. **Preprocessing** — Label encoding, standard scaling
3. **Train-Test Split** — 80/20 stratified split
4. **Model Training** — Random Forest Classifier (100 trees)
5. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC
6. **Prediction** — Classify new traffic as Normal or Attack

---

## 📊 Dataset

- **Name:** KDD Cup 1999
- **Source:** UCI Machine Learning Repository / sklearn
- **Records:** ~494,021 (10% subset)
- **Features:** 41 network traffic attributes
- **Classes:** Normal, DoS, Probe, R2L, U2R

---

## 🛠️ Technologies

| Technology | Purpose |
|-----------|---------|
| Python 3.x | Programming Language |
| Scikit-learn | ML Model (Random Forest) |
| Pandas & NumPy | Data Processing |
| Plotly | Interactive Visualizations |
| Streamlit | Web Frontend |

---

## ⚡ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/network-intrusion-detection.git
cd network-intrusion-detection

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎮 Usage

### Run the Streamlit App
```bash
streamlit run app.py
```

### Run Training Standalone
```bash
python -m model.train
```

---

## 📈 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | ~99.5% |
| **Precision** | ~99.5% |
| **Recall** | ~99.5% |
| **F1-Score** | ~99.5% |
| **AUC-ROC** | ~0.999 |

---

## 📂 Project Structure

```
├── app.py                  # Streamlit web application
├── model/
│   ├── __init__.py
│   ├── preprocess.py       # Data loading & preprocessing
│   ├── train.py            # Model training pipeline
│   └── evaluate.py         # Evaluation metrics & plots
├── saved_model/            # Trained model artifacts
│   ├── random_forest.pkl
│   ├── encoders.pkl
│   ├── scaler.pkl
│   └── test_data.npz
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## 🧠 Algorithm: Random Forest

**Why Random Forest?**
- Handles large, high-dimensional datasets efficiently
- Reduces overfitting through ensemble learning (multiple decision trees)
- Provides better accuracy compared to a single decision tree
- Works well with both numerical and categorical features
- The dataset is imbalanced, so Random Forest helps improve generalization across attack classes

---

## 👨‍💻 Author

**Ashan**

---

## 📝 License

This project is for educational purposes — Machine Learning Project Submission.
