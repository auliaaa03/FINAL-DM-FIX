import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="Clustering & Regresi COVID-19 Indonesia",
    layout="wide"
)

st.title("📊 Analisis Clustering & Regresi COVID-19 Indonesia")
st.write("Aplikasi ini menampilkan clustering provinsi dan regresi linear COVID-19.")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("Covid-19_Indonesia_Dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# ==============================
# PILIH TANGGAL
# ==============================
tanggal = st.date_input(
    "📅 Pilih Tanggal",
    min_value=df["Date"].min().date(),
    max_value=df["Date"].max().date()
)

df_tanggal = df[df["Date"].dt.date == tanggal]

if df_tanggal.empty:
    st.error("❌ Data tidak tersedia untuk tanggal tersebut")
    st.stop()

# ==============================
# CLUSTERING
# ==============================
st.header("🔹 Clustering Provinsi")

k = st.slider("Jumlah Cluster (k)", 2, 5, 3)

fitur_cluster = df_tanggal[[
    "Total_Cases",
    "Total_Deaths",
    "Total_Recovered",
    "Population",
    "Population_Density"
]]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(fitur_cluster)

kmeans = KMeans(n_clusters=k, random_state=42)
df_tanggal["Cluster"] = kmeans.fit_predict(X_scaled)

# ==============================
# VISUALISASI CLUSTER
# ==============================
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    df_tanggal["Population_Density"],
    df_tanggal["Total_Deaths"],
    c=df_tanggal["Cluster"],
    cmap="viridis",
    s=100
)

ax.set_xlabel("Kepadatan Penduduk")
ax.set_ylabel("Total Kematian")
ax.set_title("Sebaran Cluster Provinsi")

for i, prov in enumerate(df_tanggal["Province"]):
    ax.annotate(
        prov,
        (df_tanggal["Population_Density"].iloc[i],
         df_tanggal["Total_Deaths"].iloc[i]),
        fontsize=8
    )

st.pyplot(fig)

# ==============================
# REGRESI LINEAR
# ==============================
st.header("📈 Regresi Linear")

X = df_tanggal[["Population_Density"]]
y = df_tanggal["Total_Deaths"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

st.subheader("Hasil Evaluasi Model")
st.write(f"**R² Score:** {r2:.3f}")
st.write(f"**RMSE:** {rmse:.3f}")

# ==============================
# GRAFIK PREDIKSI VS AKTUAL
# ==============================
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(y, y_pred)
ax2.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    "r--"
)
ax2.set_xlabel("Aktual")
ax2.set_ylabel("Prediksi")
ax2.set_title("Prediksi vs Aktual")

st.pyplot(fig2)

