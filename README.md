# 💊 Drug Side Effects Analysis

## 📋 Description
This project analyzes drug side effects reported by patients. It covers data preprocessing, feature engineering, and interactive visualizations using Python libraries such as Pandas, Scikit-learn, Plotly, and Streamlit. An interactive Streamlit dashboard allows users to explore the data, filter by features, visualize trends, and download customized datasets.

## 📑 Table of Contents
1. [Installation](#installation)
2. [Data](#data)
3. [Pipeline Overview](#pipeline-overview)
4. [Streamlit Dashboard](#streamlit-dashboard)
5. [How to Run](#how-to-run)

## 🔧 Installation
To set up this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Pusula_Name_Surname.git

2. Install the required Python libraries:
  pip install -r requirements.txt

## 🗂️ Data
# The dataset used in this project includes:

Patient Information: Age, gender, weight, height, allergies, chronic diseases.
Drug Information: Drug name, dosage, start/end dates.
Reported Side Effects: The side effects reported by patients and the delay between drug use and onset of side effects.
You can load the dataset from the Excel file: side_effect_data.xlsx.

## 🚀 Pipeline Overview
The project pipeline consists of the following steps:

# Data Loading:
Loads the dataset from the Excel file using the load_data(filepath) function.

# Exploratory Data Analysis (EDA):
The check_df(dataframe) function provides an overview of the data, including shape, data types, missing values, and quantile distributions.

# Feature Engineering:
The feature_engineering(df) function creates new features such as:
New_Yas: Age of the patient calculated from the report date and birth date.
New_Ilac_Suresi: Duration of drug use.
New_BMI: Body Mass Index (BMI) calculated from weight and height.
New_Alerji_Varmi: Binary feature indicating the presence of allergies.
New_Yan_Etki_Gecikme_Suresi: Time between starting the drug and the report of side effects.

# Handling Missing Data:
The handle_missing_values(df) and impute_data(df) functions handle missing values using methods such as KNN Imputation for numerical columns and filling missing categorical values with appropriate labels.

# Categorical Encoding:
The encode_variables(df) function applies Label Encoding for binary columns and One-Hot Encoding for nominal categorical columns.

# Scaling:
The scale_data(df) function standardizes numerical columns (e.g., weight, height) using StandardScaler.

# Visualization:
The num_summary(dataframe, numerical_col, plot=False) and cat_summary(dataframe, col_name, plot=False) functions are used to visualize numerical and categorical data, respectively.

## 📊 Streamlit Dashboard
The Streamlit dashboard provides an interactive way to explore and visualize the data. It allows users to:

Filter the dataset by age range, drug name, and other features.
Visualize categorical data using pie charts.
Visualize numerical data using histograms and scatter plots.
Download filtered datasets as CSV files.
Key Features:
Filtering: Filter data by age, drug, or side effects.
Pie Charts: Visualize categorical variables such as drug names or side effects.
Histograms & Scatter Plots: Analyze numerical variables like weight, height, and BMI.
Download Data: Export filtered data for further analysis.

## 🖥️ How to Run
Running the Pipeline: You can execute the full data processing pipeline by running the following command:
  df_transformed = run_pipeline("side_effect_data.xlsx")

   
Running the Streamlit App: To launch the Streamlit dashboard and explore the data interactively:
  streamlit run app.py
  
This will open a web-based interface where you can filter, visualize, and download the data.
