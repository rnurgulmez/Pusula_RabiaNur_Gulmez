import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer

# Sayfa başlığı ve tema ayarları
st.set_page_config(page_title="Drug Side Effects Analysis", layout="wide")

# Başlık
st.title("💊 Drug Side Effects Analysis")

# Veri yükleme
@st.cache_data
def load_data():
    df = pd.read_excel("side_effect_data.xlsx")
    return df

df = load_data()

# Genel veri çerçevesini gösterme
st.subheader("🔍 Genel Veri İncelemesi")
with st.expander("Veriyi Göster", expanded=False):
    st.write(df.head())

# Fonksiyonları tanımlayalım
def cat_summary(dataframe, col_name):
    df_cat = pd.DataFrame({
        'Count': dataframe[col_name].value_counts(),
        'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)
    })
    return df_cat

# Numerik değişkenlerin analizi
def num_summary(dataframe, numerical_col):
    st.write(dataframe[numerical_col].describe())
    # Modern ve estetik histogram
    fig = px.histogram(dataframe, x=numerical_col, nbins=30, title=f"{numerical_col} Dağılımı",
                       color_discrete_sequence=px.colors.sequential.Teal_r)  # Teal rengi modern bir görünüm sağlar
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # Şeffaf arka plan
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="Black"
        )
    )
    st.plotly_chart(fig)

# Veri hazırlama ve Feature Engineering
def feature_engineering(df):
    df['New_Yas'] = df['Yan_Etki_Bildirim_Tarihi'].dt.year - df['Dogum_Tarihi'].dt.year
    df['New_Ilac_Suresi'] = (df['Ilac_Bitis_Tarihi'] - df['Ilac_Baslangic_Tarihi']).dt.days
    df['New_BMI'] = df['Kilo'] / (df['Boy'] / 100) ** 2
    return df

df = feature_engineering(df)

# Veri Filtreleme (Yaş ve İlaç adına göre)
st.sidebar.subheader("⚙️ Veri Filtreleme")
min_age, max_age = st.sidebar.slider("Yaş Aralığı Seçin", int(df['New_Yas'].min()), int(df['New_Yas'].max()), (20, 50))
selected_ilac = st.sidebar.selectbox("İlaç Seçin", df['Ilac_Adi'].unique())

filtered_data = df[(df['New_Yas'] >= min_age) & (df['New_Yas'] <= max_age) & (df['Ilac_Adi'] == selected_ilac)]
st.write(f"Filtrelenmiş veri (Yaş: {min_age}-{max_age}, İlaç: {selected_ilac})")
st.write(filtered_data.head())

# Kategorik değişkenlerin analizi
st.subheader("📊 Kategorik Değişkenlerin Analizi")
selected_cat_col = st.selectbox("Bir kategorik değişken seçin", df.select_dtypes(include='object').columns)
cat_data = cat_summary(df, selected_cat_col)
st.write(cat_data)

# Kategorik Değişkenler için Pasta Grafiği
fig_cat = px.pie(cat_data, values='Count', names=cat_data.index, title=f"{selected_cat_col} Dağılımı",
                 color_discrete_sequence=px.colors.sequential.Sunset)
fig_cat.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color="Black"
    )
)
st.plotly_chart(fig_cat)

# Sayısal değişkenlerin analizi
st.subheader("📈 Sayısal Değişkenlerin Analizi")
selected_num_col = st.selectbox("Bir sayısal değişken seçin", df.select_dtypes(include=['float64', 'int64']).columns)
num_summary(df, selected_num_col)

# Eksik veri doldurma
st.subheader("🛠️ Eksik Değerlerin KNN ile Doldurulması")
with st.expander("Eksik Veriyi Göster", expanded=False):
    missing_data_ratio = df.isnull().sum() / len(df) * 100
    st.write(missing_data_ratio)

imputer = KNNImputer(n_neighbors=5)
df[['Kilo', 'Boy']] = imputer.fit_transform(df[['Kilo', 'Boy']])

# Korelasyon ve scatter plot
st.subheader("📊 Boy ve Kilo Arasındaki Korelasyon")
correlation = df['Kilo'].corr(df['Boy'])
st.write(f"Kilo ve Boy arasındaki korelasyon: {correlation:.2f}")

fig_corr = px.scatter(df, x='Boy', y='Kilo', trendline='ols', title="Boy ve Kilo İlişkisi",
                      color_discrete_sequence=["#6D5B97"])  # Modern bir mor ton
fig_corr.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color="Black"
    )
)
st.plotly_chart(fig_corr)

# Standartlaştırma (Scaling)
scaler = StandardScaler()
df[['Kilo', 'Boy']] = scaler.fit_transform(df[['Kilo', 'Boy']])

st.subheader("🔍 Özelliklerin Standartlaştırılması")
with st.expander("Standartlaştırılmış Veriyi Göster", expanded=False):
    st.write(df[['Kilo', 'Boy']].head())

# Arayüz düzenlemeleri
st.sidebar.title("📑 Analiz Adımları")
st.sidebar.write("1. Genel veri incelemesi")
st.sidebar.write("2. Kategorik ve Sayısal değişken analizleri")
st.sidebar.write("3. KNN ile eksik veri doldurma")
st.sidebar.write("4. Boy ve Kilo arasındaki korelasyon analizi")

# Ekstra görsel: Boxplot ile sayısal dağılım
st.subheader("📊 Boy ve Kilo Dağılımı (Boxplot)")
fig_box = go.Figure()
fig_box.add_trace(go.Box(y=df['Boy'], name='Boy', marker_color='rgba(102, 197, 204, 0.7)'))  # Modern mavi-yeşil renk
fig_box.add_trace(go.Box(y=df['Kilo'], name='Kilo', marker_color='rgba(248, 156, 116, 0.7)'))  # Modern turuncu renk
fig_box.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color="Black"
    )
)
st.plotly_chart(fig_box)

# Veriyi CSV olarak indirme
st.subheader("💾 Veriyi İndir")
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(df)

st.download_button(
    label="CSV olarak indir",
    data=csv,
    file_name='filtered_data.csv',
    mime='text/csv',
)
