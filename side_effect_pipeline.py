import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)


# Load data
def load_data(filepath):
    df = pd.read_excel(filepath)
    return df


# General overview of the data
def check_df(dataframe, head=5):
    print("########### Shape ###########")
    print(dataframe.shape)
    print("########### Types ###########")
    print(dataframe.dtypes)
    print("########### Head ###########")
    print(dataframe.head(head))
    print("########### Tail ###########")
    print(dataframe.tail(head))
    print("########### NA ###########")
    print(dataframe.isnull().sum())
    print("########### Quantiles ###########")
    print(dataframe.describe([0.05, 0.50, 0.95, 0.99]).T)


# Feature engineering
def feature_engineering(df):
    df['New_Yas'] = df['Yan_Etki_Bildirim_Tarihi'].dt.year - df['Dogum_Tarihi'].dt.year
    df['New_Ilac_Suresi'] = (df['Ilac_Bitis_Tarihi'] - df['Ilac_Baslangic_Tarihi']).dt.days
    df['New_BMI'] = df['Kilo'] / (df['Boy'] / 100) ** 2
    df['New_BMI_Kategori'] = df['New_BMI'].apply(bmi_category)
    df['New_Alerji_Varmi'] = df['Alerjilerim'].notnull().astype(int)
    df['New_Yan_Etki_Gecikme_Suresi'] = (df['Yan_Etki_Bildirim_Tarihi'] - df['Ilac_Baslangic_Tarihi']).dt.days
    return df


def bmi_category(bmi):
    if bmi < 18.5:
        return 'Zayif'
    elif 18.5 <= bmi < 24.9:
        return 'Normal'
    elif 25 <= bmi < 29.9:
        return 'Fazla Kilolu'
    else:
        return 'Obez'


# Handle missing values
def handle_missing_values(df):
    df['Cinsiyet'] = df['Cinsiyet'].fillna('Bilinmiyor')
    df['Il'] = df['Il'].fillna(df['Il'].mode()[0])
    df['Alerjilerim'] = df['Alerjilerim'].fillna('Alerjim yok')

    kronik_cols = df.filter(like='Kronik').columns
    df[kronik_cols] = df[kronik_cols].fillna('Kronik hastalık yok')

    df['Kan Grubu'] = df['Kan Grubu'].fillna('Bilinmiyor')
    return df


# Encode categorical variables
def encode_variables(df):
    binary_cols = ['New_Alerji_Varmi']
    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    nominal_cols = ['Cinsiyet', 'Il', 'Ilac_Adi', 'Yan_Etki', 'Alerjilerim',
                    'Kronik Hastaliklarim', 'Baba Kronik Hastaliklari',
                    'Anne Kronik Hastaliklari', 'Kiz Kardes Kronik Hastaliklari',
                    'Erkek Kardes Kronik Hastaliklari', 'Kan Grubu', 'New_BMI_Kategori']

    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    return df


# Impute missing numerical data using KNN
def impute_data(df):
    imputer = KNNImputer(n_neighbors=5)
    df[['Kilo', 'Boy']] = imputer.fit_transform(df[['Kilo', 'Boy']])
    df['New_BMI'] = df['Kilo'] / (df['Boy'] / 100) ** 2
    return df


# Scale numerical features
def scale_data(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df


# Visualizations (numerical columns)
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        fig = px.histogram(dataframe, x=numerical_col, nbins=30, title=f"{numerical_col} Dağılımı",
                           color_discrete_sequence=px.colors.sequential.Peach, opacity=0.8)
        fig.update_layout(plot_bgcolor='rgba(240, 240, 240, 0.9)',
                          xaxis_title=numerical_col,
                          yaxis_title='Count',
                          title_x=0.5,
                          font=dict(family="Arial, sans-serif", size=12, color="Black"))
        fig.show()


# Visualizations (categorical columns)
def cat_summary(dataframe, col_name, plot=False):
    df = pd.DataFrame({'Count': dataframe[col_name].value_counts(),
                       'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}).reset_index().rename(
        columns={'index': col_name})
    print(df)

    if plot:
        fig = px.pie(df, values='Count', names=col_name, title=f'Distribution of {col_name}',
                     hole=0.3, color_discrete_sequence=px.colors.sequential.Sunsetdark)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.show()


# Run the entire pipeline
def run_pipeline(filepath):
    # Load data
    df = load_data(filepath)

    # Initial data check
    check_df(df)

    # Drop unnecessary columns
    df.drop(["Kullanici_id", "Uyruk"], axis=1, inplace=True)

    # Feature engineering
    df = feature_engineering(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Encode categorical variables
    df = encode_variables(df)

    # Impute missing data
    df = impute_data(df)

    # Scale numerical features
    df = scale_data(df)

    # Visualization of categorical variables
    cat_cols, num_cols, _, _ = grab_col_names(df)
    for col in cat_cols:
        cat_summary(df, col, plot=True)

    # Visualization of numerical variables
    for col in num_cols:
        num_summary(df, col, plot=True)

    return df


# Identify column types
def grab_col_names(dataframe, cat_th=10, car_th=100):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "object"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "object"]
    date_cols = [col for col in dataframe.columns if dataframe[col].dtype == 'datetime64[ns]']
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if
                dataframe[col].dtypes != "object" and dataframe[col].dtype != 'datetime64[ns]']
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car, date_cols


# File path
filepath = "side_effect_data.xlsx"

# Run the pipeline
df_transformed = run_pipeline(filepath)

# Display first few rows of transformed data
print(df_transformed.head())
