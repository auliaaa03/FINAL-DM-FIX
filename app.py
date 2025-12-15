import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Clustering & Regresi COVID-19 Indonesia",
    layout="wide"
)

st.title("📊 Analisis Clustering & Regresi COVID-19 Indonesia")
st.write("Aplikasi ini menampilkan clustering provinsi dan regresi linear COVID-19.")

# =========================
# LOAD DATA (ANTI KEYERROR)
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Covid-19_Indonesia_Dataset.csv")

    # CARI KOLOM TANGGAL
    possible_date_cols = ["Date", "Tanggal", "Date_New", "Date Reported"]

    date_col = None
    for col in possible_date_cols:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        st.error("❌ Kolom tanggal tidak ditemukan di dataset")
        st.stop()

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "Date"})

    return df

df = load_data()

# =========================
# PILIH TANGGAL
# =========================
tanggal = st.date_input(
    "📅 Pilih Tanggal",
    min_value=df["Date"].min().date(),
    max_value=df["Date"].max().date()
)

df_tanggal = df[df["Date"].dt.date == tanggal]

if df_tanggal.empty:
    st.warning("⚠️ Tidak ada data pada tanggal tersebut")
    st.stop()

# =========================
# CLUSTERING
# =========================
st.header("🔹 Clustering Provinsi")

k = st.slider("Jumlah Cluster (k)", 2, 5, 3)

fitur = [
    "Total_Cases",
    "Total_Deaths",
    "Total_Recovered",
    "Population",
    "Population_Density"
]

# Pastikan kolom ada
for col in fitur:
    if col not in df_tanggal.columns:
        st.error(f"❌ Kolom `{col}` tidak ada di dataset")
        st.stop()

X = df_tanggal[fitur]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, random_state=42)
df_tanggal["Cluster"] = kmeans.fit_predict(X_scaled)

# =========================
# VISUALISASI CLUSTER
# =========================
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(
    df_tanggal["Population_Density"],
    df_tanggal["Total_Deaths"],
    c=df_tanggal["Cluster"],
    cmap="viridis",
    s=100
)

ax.set_xlabel("Kepadatan Penduduk")
ax.set_ylabel("Total Kematian")
ax.set_title("Sebaran Cluster Provinsi")

st.pyplot(fig)

# =========================
# REGRESI LINEAR
# =========================
st.header("📈 Regresi Linear")

X_reg = df_tanggal[["Population_Density"]]
y_reg = df_tanggal["Total_Deaths"]

model = LinearRegression()
model.fit(X_reg, y_reg)

y_pred = model.predict(X_reg)

r2 = r2_score(y_reg, y_pred)
rmse = np.sqrt(mean_squared_error(y_reg, y_pred))

st.subheader("Evaluasi Model")
st.write(f"**R² Score:** {r2:.3f}")
st.write(f"**RMSE:** {rmse:.3f}")

# =========================
# PLOT PREDIKSI VS AKTUAL
# =========================
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(y_reg, y_pred)
ax2.plot(
    [y_reg.min(), y_reg.max()],
    [y_reg.min(), y_reg.max()],
    "r--"
)

ax2.set_xlabel("Aktual")
ax2.set_ylabel("Prediksi")
ax2.set_title("Prediksi vs Aktual")

st.pyplot(fig2)
