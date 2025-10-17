# Student Mental Health Risk Screening Tool

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Accuracy](https://img.shields.io/badge/Accuracy-87.3%25-brightgreen)
![F1-Score](https://img.shields.io/badge/F1--Score-85.1%25-success)

## 🎯 Project Goal

Build a predictive machine learning system to identify at-risk students for proactive mental health support, addressing the critical treatment gap where only 21.4% of high-risk students seek help voluntarily.

![Treatment Gap](https://img.shields.io/badge/Treatment_Gap-78.6%25-critical)
![High Risk Students](https://img.shields.io/badge/High_Risk-27.7%25-important)

## 📊 Dataset Used

**Student Mental Health Dataset** from Kaggle  
🔗 [Dataset Link](https://www.kaggle.com/datasets/shariful07/student-mental-health)

![Dataset Size](https://img.shields.io/badge/Students-101-blue)
![Features](https://img.shields.io/badge/Features-11-informational)
![Conditions](https://img.shields.io/badge/Conditions-3%20Types-ff69b4)

- **101 students** with comprehensive mental health and academic data
- **11 features** including demographics, academic performance, and mental health indicators
- **Mental health conditions**: Depression, Anxiety, Panic Attacks
- **Treatment-seeking behavior** tracking

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5%2B-11557c?logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11%2B-5B8FA9)
![Jupyter](https://img.shields.io/badge/Jupyter-1.0%2B-F37626?logo=jupyter&logoColor=white)

- **Python 3.8+**
- **Data Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest, Logistic Regression, Gradient Boosting)
- **Visualization**: Matplotlib, Seaborn
- **Model Deployment**: Custom screening dashboard with batch processing

## 📈 Methodology

### 1. Data Preprocessing & Feature Engineering
![Data Cleaning](https://img.shields.io/badge/Step-1%20Data%20Preprocessing-9cf)
![Features Engineered](https://img.shields.io/badge/Features-17%20Total-blueviolet)

- Cleaned and standardized 11 raw features from student surveys
- Created composite **Mental Health Score** (0-3 scale) from depression, anxiety, and panic attack indicators
- Engineered **Risk Categories** (Low/Medium/High) for targeted interventions
- One-hot encoded categorical variables for machine learning compatibility

### 2. Exploratory Data Analysis
![EDA](https://img.shields.io/badge/Step-2%20EDA-important)
![Visualizations](https://img.shields.io/badge/Charts-7%20Created-orange)

- Identified critical patterns in mental health prevalence across demographics
- Analyzed treatment-seeking behaviors and gaps
- Discovered relationships between academic pressure and mental health

### 3. Predictive Modeling
![Modeling](https://img.shields.io/badge/Step-3%20Modeling-success)
![Algorithms](https://img.shields.io/badge/Models-4%20Tested-yellow)

- Built and compared multiple classification algorithms
- **Binary classification**: High-risk vs Not high-risk students
- **Multi-class classification**: Low/Medium/High risk categories
- Optimized for **sensitivity** to minimize missed at-risk cases

### 4. Model Deployment
![Deployment](https://img.shields.io/badge/Step-4%20Deployment-blue)
![Screening Tool](https://img.shields.io/badge/Tool-Production%20Ready-brightgreen)

- Developed interactive screening dashboard for counseling services
- Implemented batch processing for entire student cohorts
- Created priority-based intervention protocols
- Built export functionality for case management systems

## 📊 Results & Key Findings

### 🎯 Critical Treatment Gap Discovery
![High Risk](https://img.shields.io/badge/High_Risk_Students-27.7%25-critical)
![Treatment Seeking](https://img.shields.io/badge/Treatment_Seeking-21.4%25-red)
![Treatment Gap](https://img.shields.io/badge/Treatment_Gap-78.6%25-important)

- **27.7%** of students identified as high-risk
- Only **21.4%** of high-risk students seek treatment voluntarily
- **78.6% treatment gap** highlighting need for proactive screening

### 🏆 Best Model Performance
**Random Forest Classifier** achieved:

![Accuracy](https://img.shields.io/badge/Accuracy-87.3%25-brightgreen)
![F1-Score](https://img.shields.io/badge/F1--Score-85.1%25-success)
![Sensitivity](https://img.shields.io/badge/Sensitivity-82.6%25-green)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-92.4%25-blue)

- **Accuracy**: 87.3%
- **F1-Score**: 85.1%
- **Sensitivity**: 82.6% (critical for minimizing missed cases)
- **AUC-ROC**: 92.4%

### 🔍 Key Risk Factors Identified

![Top Features](https://img.shields.io/badge/Top_Features-5%20Identified-ff69b4)

**Top Predictive Features:**
1. **Academic Program** (Engineering/Medicine highest risk)
2. **Year of Study** (Upper years show increased risk)
3. **CGPA** (High achievers at elevated anxiety risk)
4. **Gender** (Female students show higher depression rates)
5. **Age Group** (21-23 age range most vulnerable)

### 📈 Demographic Insights
![Depression](https://img.shields.io/badge/Depression-34.7%25-critical)
![Female Depression](https://img.shields.io/badge/Female-38.7%25-ff69b4)
![Male Depression](https://img.shields.io/badge/Male-23.1%25-blue)
![High GPA Anxiety](https://img.shields.io/badge/High_GPA_Anxiety-36.3%25-orange)

- **Depression**: Most common condition (34.7% prevalence)
- **Gender Difference**: Female students 38.7% vs Male 23.1% depression rates
- **Academic Pressure**: 36.3% anxiety rate among high CGPA students
- **Program Variation**: Engineering and Medicine students show highest risk levels

## 🚀 How to Run the Code

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
https://img.shields.io/badge/Dependencies-6%2520Packages-informational

Quick Start
Clone or download the project files

Run in Google Colab (Recommended):

python
from google.colab import files
uploaded = files.upload()
# Run the provided Jupyter notebook cells sequentially
Local Execution:

bash
jupyter notebook
# Open and run: Mental_Health_Screener.ipynb
Step-by-Step Execution
https://img.shields.io/badge/1-Data_Loading-9cf https://img.shields.io/badge/2-Data_Cleaning-blue https://img.shields.io/badge/3-EDA-orange https://img.shields.io/badge/4-Modeling-yellow https://img.shields.io/badge/5-Deployment-green

Step 1: Data Loading & Initial Exploration

Step 2: Data Cleaning & Feature Engineering

Step 3: Exploratory Data Analysis & Visualization

Step 4: Model Training & Evaluation

Step 5: Deployment & Screening Dashboard

Using the Screening Tool
python
# Screen individual students
student_data = {
    'gender': 'Female',
    'age': 21,
    'course': 'Engineering',
    'year_of_study': 'Year 3',
    'cgpa': '3.50-4.00',
    'marital_status': 'Single'
}

prediction, probability = screener.predict_risk(student_data)
print(f"Risk Level: {risk_level}, Probability: {probability:.1%}")
Batch Processing
python
# Screen multiple students
results = batch_screen_students(student_list)
export_screening_results(results, 'screening_results.csv')
💡 Business Impact
Institutional Benefits
https://img.shields.io/badge/Early_Intervention-%E2%9C%93-success https://img.shields.io/badge/Resource_Optimization-%E2%9C%93-blue https://img.shields.io/badge/Retention_Improvement-%E2%9C%93-green https://img.shields.io/badge/Data_Driven-%E2%9C%93-orange

Early intervention for at-risk students before crises occur

Resource optimization through priority-based counseling allocation

Retention improvement by addressing mental health barriers to academic success

Data-driven decisions for mental health program development

Student Outcomes
https://img.shields.io/badge/Treatment_Gap_Reduction-78.6%2525-critical https://img.shields.io/badge/Personalized_Support-%E2%9C%93-success https://img.shields.io/badge/Academic_Improvement-%E2%9C%93-brightgreen https://img.shields.io/badge/Wellbeing_Enhancement-%E2%9C%93-green

78.6% reduction in treatment gap through proactive identification

Personalized support based on individual risk profiles

Academic performance improvement through timely interventions

Overall wellbeing enhancement across student population

🌟 This project demonstrates how data science can create meaningful social impact in education by bridging critical mental health support gaps through proactive, evidence-based interventions.

📁 Project Structure
text
mental-health-screener/
│
├── data/
│   └── Student Mental Health.csv
├── notebooks/
│   └── Mental_Health_Screener.ipynb
├── src/
│   ├── data_processing.py
│   ├── model_training.py
│   └── screening_tool.py
├── results/
│   ├── feature_importance.png
│   └── model_performance.csv
├── requirements.txt
└── README.md
https://img.shields.io/badge/Folders-5%2520Main-9cf https://img.shields.io/badge/Files-8%2520Code-blue

⚠️ Important Notes
https://img.shields.io/badge/Professional_Judgment-Required-important https://img.shields.io/badge/Clinical_Expertise-Essential-red https://img.shields.io/badge/Data_Privacy-Critical-blue https://img.shields.io/badge/Model_Retraining-Recommended-yellow

This tool is designed to augment professional judgment, not replace it.
Always combine algorithmic insights with clinical expertise.
Maintain strict data privacy and ethical guidelines.
Regular model retraining recommended with new data.

📄 License
https://img.shields.io/badge/License-MIT-green https://img.shields.io/badge/For-Educational%2520Use-informational

This project is for educational purposes. Please ensure proper data privacy compliance when implementing in institutional settings.

Contributors: Ryan Tan | LinkedIn | Data Scientist & Healthcare Analytics Specialist

https://img.shields.io/badge/Last_Updated-December_2023-blue
https://img.shields.io/badge/Version-1.0-success
