# === MENTAL HEALTH ANALYSIS PROJECT ===
# STEP 1: Direct Upload to Google Colab (FIXED)

print("=== LOADING STUDENT MENTAL HEALTH DATASET ===")

# 1.1 Import libraries
from google.colab import files
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up visualization
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("✅ Libraries imported successfully!")

# 1.2 Upload and load the dataset
print("\n--- UPLOAD DATASET ---")
print("Please upload 'Student Mental health.csv' when prompted...")

# Upload the file
uploaded = files.upload()

# Get the actual filename from the uploaded files
filename = list(uploaded.keys())[0]
print(f"✅ Uploaded file: {filename}")

# Load the dataset using the actual filename
mental_health_df = pd.read_csv(io.BytesIO(uploaded[filename]))

print("✅ Dataset loaded successfully!")

# 1.3 Explore the dataset
print("\n" + "="*60)
print("DATASET EXPLORATION")
print("="*60)

print(f"📊 Dataset Shape: {mental_health_df.shape}")
print(f"👥 Number of Students: {mental_health_df.shape[0]}")
print(f"📋 Number of Features: {mental_health_df.shape[1]}")

print("\n🔤 COLUMN NAMES:")
for i, col in enumerate(mental_health_df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n👀 FIRST 5 ROWS:")
print(mental_health_df.head())

print("\n📋 DATA TYPES:")
print(mental_health_df.dtypes)

print("\n❌ MISSING VALUES:")
missing_data = mental_health_df.isnull().sum()
if missing_data.sum() == 0:
    print("✅ No missing values found!")
else:
    for col, count in missing_data.items():
        if count > 0:
            print(f"  - {col}: {count} missing values ({count/len(mental_health_df)*100:.1f}%)")

# 1.4 Analyze specific values in key columns
print("\n" + "="*60)
print("COLUMN VALUE ANALYSIS")
print("="*60)

# Check all columns in the dataset
for col in mental_health_df.columns:
    print(f"\n📊 {col}:")
    value_counts = mental_health_df[col].value_counts()
    print(value_counts)
    
    # Show percentages
    percentages = mental_health_df[col].value_counts(normalize=True) * 100
    for value, count in value_counts.items():
        percent = percentages[value]
        print(f"   {value}: {count} students ({percent:.1f}%)")

# 1.5 Basic statistics
print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)
print(mental_health_df.describe(include='all'))

# 1.6 Check for mental health indicators
print("\n" + "="*60)
print("MENTAL HEALTH INDICATORS SUMMARY")
print("="*60)

# Look for columns that might contain mental health information
mental_health_keywords = ['depression', 'anxiety', 'panic', 'stress', 'treatment', 'mental']
mental_health_cols = []

for col in mental_health_df.columns:
    col_lower = str(col).lower()
    if any(keyword in col_lower for keyword in mental_health_keywords):
        mental_health_cols.append(col)

if mental_health_cols:
    print("🎯 Mental Health Related Columns Found:")
    for col in mental_health_cols:
        print(f"  - {col}")
else:
    print("ℹ️  No obvious mental health columns found")

# 1.7 Initial insights
print("\n" + "="*60)
print("INITIAL INSIGHTS")
print("="*60)

print("🔍 Quick Analysis:")
print(f"- Total students: {len(mental_health_df)}")
print(f"- Columns available: {list(mental_health_df.columns)}")

# Check if we have demographic info
demographic_cols = ['gender', 'age', 'course', 'year', 'cgpa']
available_demo = [col for col in demographic_cols if any(col in str(c).lower() for c in mental_health_df.columns)]
print(f"- Demographic info: {available_demo}")

print("\n✅ STEP 1 COMPLETED SUCCESSFULLY!")
print("📝 Ready for Step 2: Data Cleaning and Preprocessing!")

# Save the filename for reference
dataset_filename = filename
print(f"💾 Dataset saved as variable: mental_health_df")
print(f"📄 Original filename: {dataset_filename}")
