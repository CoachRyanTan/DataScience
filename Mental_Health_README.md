# 🧠 Student Mental Health Risk Screening Tool

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-lightblue?logo=pandas)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Viz-Matplotlib-yellow?logo=plotly)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Viz-Seaborn-9cf?logo=seaborn)](https://seaborn.pydata.org/)
[![Dashboard](https://img.shields.io/badge/Deployment-Custom%20Screening%20Dashboard-success?logo=streamlit)]()
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-87.3%25-brightgreen)]()
[![AUC](https://img.shields.io/badge/AUC--ROC-92.4%25-green)]()
[![Sensitivity](https://img.shields.io/badge/Sensitivity-82.6%25-yellowgreen)]()
[![License](https://img.shields.io/badge/License-Educational-lightgrey)]()

<!-- 🧩 GitHub Actions Badges -->
[![CI](https://github.com/your-username/mental-health-screener/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/mental-health-screener/actions/workflows/ci.yml)
[![Notebook Validation](https://github.com/your-username/mental-health-screener/actions/workflows/notebook-validation.yml/badge.svg)](https://github.com/your-username/mental-health-screener/actions/workflows/notebook-validation.yml)
[![Model Retraining](https://github.com/your-username/mental-health-screener/actions/workflows/model-retrain.yml/badge.svg)](https://github.com/your-username/mental-health-screener/actions/workflows/model-retrain.yml)

---

## 🎯 Project Goal

Build a **predictive machine learning system** to identify at-risk students for proactive mental health support — addressing the treatment gap where only **21.4% of high-risk students seek help voluntarily**.

---

## 📊 Dataset Used

**Student Mental Health Dataset** from Kaggle  
🔗 [View Dataset](https://www.kaggle.com/datasets/shariful07/student-mental-health)

- 101 students with comprehensive mental health and academic data  
- 11 features including demographics, academic performance, and mental health indicators  
- Mental health conditions: Depression, Anxiety, Panic Attacks  
- Treatment-seeking behavior tracking  

---

## 🛠 Tech Stack

| Category | Tools Used |
|-----------|-------------|
| **Language** | Python 3.8+ |
| **Data Analysis** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Random Forest, Logistic Regression, Gradient Boosting) |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Custom dashboard with batch processing |

---

## 📈 Methodology

### 1. Data Preprocessing & Feature Engineering
- Standardized 11 raw features from student survey data  
- Created **Mental Health Score** (0–3 scale) from depression, anxiety, and panic indicators  
- Defined **Risk Categories** (Low / Medium / High)  
- One-hot encoded categorical variables for ML models  

### 2. Exploratory Data Analysis
- Identified trends in mental health prevalence  
- Examined academic pressure and help-seeking gaps  
- Highlighted correlations between performance and mental health  

### 3. Predictive Modeling
- Compared Random Forest, Logistic Regression, and Gradient Boosting  
- Built both **binary (high-risk vs not)** and **multi-class (low/med/high)** models  
- Prioritized **sensitivity** to minimize missed high-risk students  

### 4. Deployment
- Built an **interactive screening dashboard** for counseling teams  
- Enabled **batch screening** for student cohorts  
- Integrated **export features** for case management systems  

---

## 📊 Results & Key Findings

### 🎯 Critical Treatment Gap
- **27.7%** students identified as high-risk  
- Only **21.4%** of high-risk students sought treatment  
- **78.6% treatment gap** — confirming the need for proactive screening  

### 🏆 Best Model: Random Forest Classifier
| Metric | Score |
|---------|--------|
| **Accuracy** | 87.3% |
| **F1-Score** | 85.1% |
| **Sensitivity** | 82.6% |
| **AUC-ROC** | 92.4% |

### 🔍 Top 5 Predictive Features
1. Academic Program (Engineering/Medicine = highest risk)  
2. Year of Study (Upper years show higher risk)  
3. CGPA (High achievers face anxiety risk)  
4. Gender (Female > Male depression rates)  
5. Age Group (21–23 years = most vulnerable)  

---

## 📈 Demographic Insights
| Category | Key Finding |
|-----------|-------------|
| **Depression** | 34.7% prevalence |
| **Gender Gap** | Female: 38.7% vs Male: 23.1% |
| **Academic Pressure** | 36.3% anxiety among high CGPA students |
| **Program Risk** | Engineering & Medicine highest |

---

---

## ⚙️ How to Run This Project

You can explore and reproduce this project through **Google Colab (recommended)** or by running it locally on your own computer.

---

### 🟢 Option 1 — Run on Google Colab (Recommended for Reviewers)

The project is fully runnable in a cloud environment with no setup needed.

1. Click to open the notebook directly in Google Colab:  
   👉 [**Open Student Mental Health Screening Tool in Colab**](https://colab.research.google.com/drive/your-colab-id-here)
2. Make sure you are logged into your Google account.  
3. Go to **Runtime → Run all** to execute all code cells in sequence.  
4. The notebook will automatically:
   - Import required libraries  
   - Load and preprocess the dataset  
   - Train multiple ML models (Random Forest, Logistic Regression, Gradient Boosting)  
   - Generate evaluation metrics and visualizations  
   - Display feature importance and model comparison charts  

**💡 Ideal for:** Recruiters or hiring managers who want to validate technical execution, ML workflow, and reporting structure in real time.

---

### ⚙️ Option 2 — Run Locally on Your Machine

#### Step 1 — Clone the Repository
```bash
git clone https://github.com/yourusername/student-mental-health-screener.git
cd student-mental-health-screener
```
Step 2 — (Optional) Create a Virtual Environment
```bash
Copy code
python -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows
```
Step 3 — Install Required Dependencies
```bash
Copy code
pip install -r requirements.txt
```
Step 4 — Launch the Notebook
```bash
Copy code
jupyter notebook Student_Mental_Health_Screener.ipynb
```
Step 5 — View Model Outputs
Once all cells are executed, you will see:
- Model comparison table (Accuracy, AUC, F1-Score)
- ROC curve visualization (roc.png)
- Feature importance chart
- Dashboard screenshots or app deployment logs (if enabled)

📁 Project Folder Structure
```bash
Copy code
student-mental-health-screener/
│
├── Student_Mental_Health_Screener.ipynb     # Main Jupyter notebook
├── data/                                    # Raw and processed datasets
│   ├── student_mental_health.csv
│
├── roc.png                                  # ROC Curve visualization
├── feature_importance.png                   # Top predictive features
├── dashboard_demo.png                       # Screenshot of Streamlit dashboard
│
├── requirements.txt                         # Dependencies list
└── README.md                                # Project documentation
```
💼 Skills Demonstrated

✅ Data Cleaning & Feature Engineering — Categorical encoding, risk scoring
✅ Exploratory Data Analysis (EDA) — Correlation heatmaps, feature trends
✅ Model Development & Evaluation — Random Forest, Logistic Regression, Gradient Boosting
✅ Model Comparison & ROC Analysis — AUC-ROC visualization and metric summary
✅ Dashboard Deployment — Streamlit dashboard for mental health screening simulation
✅ Communication for Stakeholders — Translating ML outputs into actionable counseling insights
