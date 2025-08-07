import streamlit as st
st.set_page_config(page_title="Carbon Footprint Calculator", layout="centered")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- Load Dataset ---
df = pd.read_csv("owid-co2-data.csv")

# Filter out aggregate entries (e.g. "World", "Asia") by keeping 3-letter country codes
df = df[df['iso_code'].str.len() == 3]

# Select and clean relevant columns
df = df[['gdp', 'energy_per_capita', 'population', 'co2']].dropna()

# Feature matrix and target variable
X = df[['gdp', 'energy_per_capita', 'population']]
y = df['co2']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data for training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
mse = mean_squared_error(y_test, model.predict(X_test))

# --- Streamlit App UI ---
st.title("🌍 AI-Based Carbon Footprint Calculator")
st.write("Estimate national-level CO₂ emissions using GDP, energy use, and population.")

st.sidebar.header("🔧 Input National/Regional Stats")
gdp = st.sidebar.number_input("GDP (USD per capita)", min_value=100.0, value=1000.0)
energy = st.sidebar.number_input("Energy Use (kWh per capita)", min_value=10.0, value=500.0)
population = st.sidebar.number_input("Population", min_value=10000.0, value=100000.0)

# Scale input
input_scaled = scaler.transform([[gdp, energy, population]])

# Predict
if st.button("📊 Estimate CO2 Emission"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"🌱 Estimated CO₂ Emission: **{prediction:.2f} tons per year**")

    # Suggestions
    st.subheader("💡 Personalized Suggestions")
    if gdp > 10000:
        st.info("💸 Consider investing in renewable energy and green innovation.")
    if energy > 5000:
        st.warning("⚡ High energy use detected. Promote efficiency and cleaner grids.")
    if population > 5000000:
        st.info("👥 Promote sustainable urban development and transport systems.")
    if prediction > 5:
        st.warning("⚠️ Emissions are high. Climate policies and lifestyle changes are essential.")
    else:
        st.success("✅ Emissions are at a reasonable level!")

    # Normalize values for fair comparison
    values = [gdp, energy, population]
    total = sum(values)
    norm_values = [v / total for v in values]

    labels = ['GDP', 'Energy Use', 'Population']
    colors = ['skyblue', 'lightgreen', 'orange']

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        norm_values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.8,
        labeldistance=1.15
    )

    for text in texts + autotexts:
        text.set_fontsize(10)
        text.set_fontweight('bold')

    ax.axis('equal')
    st.pyplot(fig)

# # Footer
# st.markdown("---")
# st.markdown("Created for 🏆 GTU Poster Competition 2025 | Powered by OWID Dataset")
