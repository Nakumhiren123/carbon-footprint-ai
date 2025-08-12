import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Page Configuration ---
st.set_page_config(page_title="Carbon Footprint Calculator", layout="centered", initial_sidebar_state="expanded")

# --- Model Training (Cached to run only once) ---
@st.cache_resource
def train_model():
    """Loads data and trains the RandomForestRegressor model."""
    # Load dataset from the provided CSV file
    df = pd.read_csv("owid-co2-data.csv")

    # Filter out aggregate entries (e.g., "World", "Asia")
    df = df[df['iso_code'].str.len() == 3]

    # Select and clean relevant columns, removing rows with missing values
    df = df[['gdp', 'energy_per_capita', 'population', 'co2']].dropna()

    # Define feature matrix (X) and target variable (y)
    X = df[['gdp', 'energy_per_capita', 'population']]
    y = df['co2']

    # Initialize and fit the scaler on the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data for training and testing
    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler

# Train the model and get the scaler
model, scaler = train_model()


# --- Streamlit App UI ---
st.title("🌍 AI-Based Carbon Footprint Calculator")
st.write("Estimate national-level CO₂ emissions using GDP, energy use, and population.")
st.markdown("---")

# --- User Input Sidebar ---
st.sidebar.header("🔧 Input Your Data")
gdp = st.sidebar.number_input("GDP (in USD per capita)", min_value=100.0, value=10000.0, step=500.0)
energy = st.sidebar.number_input("Energy Use (in kWh per capita)", min_value=10.0, value=3000.0, step=100.0)
population = st.sidebar.number_input("Population", min_value=10000.0, value=5000000.0, step=10000.0)


# --- Prediction and Chart Display ---
if st.sidebar.button("📊 Estimate CO₂ Emission"):

    # Scale the user input for the model
    input_data = [[gdp, energy, population]]
    input_scaled = scaler.transform(input_data)

    # Predict CO₂ emission
    prediction = model.predict(input_scaled)[0]

    st.subheader("📈 Estimated Emission")
    st.success(f"**{prediction:.2f} tons** of CO₂ per year")

    # --- Display Pie Chart based on User Input ---
    st.subheader("📊 Contribution of Inputs")

    # Use the raw, unscaled user inputs for an intuitive pie chart
    raw_values = [gdp, energy, population]
    labels = ['GDP', 'Energy Use', 'Population']
    colors = ['#66b3ff', '#99ff99', '#ffcc99'] # Skyblue, Light Green, Orange

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        raw_values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.85,
        explode=(0.02, 0.02, 0.02) # Slightly separate the wedges
    )

    # Style the text on the chart
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
        autotext.set_fontweight('bold')


    ax.axis('equal')  # Ensures the pie chart is a circle
    st.pyplot(fig)

    # --- Personalized Suggestions ---
    st.subheader("💡 Personalized Suggestions")
    if gdp > 15000:
        st.info("💸 Your economy is strong. This is a great opportunity to invest in renewable energy and green technology.")
    if energy > 5000:
        st.warning("⚡️ High energy consumption detected. Promote energy efficiency and advocate for cleaner power grids.")
    if population > 10000000:
        st.info("👥 For large populations, promoting sustainable urban development and public transport systems is crucial.")
    if prediction > 10:
        st.warning("⚠️ The estimated emission level is high. Implementing strong climate policies and encouraging lifestyle changes are essential.")
    else:
        st.success("✅ Your estimated emissions are at a reasonable level. Keep up the good work!")


# --- Footer ---
st.markdown("---")
st.markdown("Created for 🏆 GTU Poster Competition 2025 | Powered by OWID Dataset")

# import streamlit as st
# st.set_page_config(page_title="Carbon Footprint Calculator", layout="centered")

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error

# # --- Load Dataset ---
# df = pd.read_csv("owid-co2-data.csv")

# # Filter out aggregate entries (e.g. "World", "Asia") by keeping 3-letter country codes
# df = df[df['iso_code'].str.len() == 3]

# # Select and clean relevant columns
# df = df[['gdp', 'energy_per_capita', 'population', 'co2']].dropna()

# # Feature matrix and target variable
# X = df[['gdp', 'energy_per_capita', 'population']]
# y = df['co2']

# # Normalize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split data for training
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Train model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate model
# mse = mean_squared_error(y_test, model.predict(X_test))

# # --- Streamlit App UI ---
# st.title("🌍 AI-Based Carbon Footprint Calculator")
# st.write("Estimate national-level CO₂ emissions using GDP, energy use, and population.")

# st.sidebar.header("🔧 Input National/Regional Stats")
# gdp = st.sidebar.number_input("GDP (USD per capita)", min_value=100.0, value=1000.0)
# energy = st.sidebar.number_input("Energy Use (kWh per capita)", min_value=10.0, value=500.0)
# population = st.sidebar.number_input("Population", min_value=10000.0, value=100000.0)

# # Scale input
# input_scaled = scaler.transform([[gdp, energy, population]])

# # Predict
# if st.button("📊 Estimate CO2 Emission"):
#     prediction = model.predict(input_scaled)[0]
#     st.success(f"🌱 Estimated CO₂ Emission: **{prediction:.2f} tons per year**")

#     # Suggestions
#     st.subheader("💡 Personalized Suggestions")
#     if gdp > 10000:
#         st.info("💸 Consider investing in renewable energy and green innovation.")
#     if energy > 5000:
#         st.warning("⚡ High energy use detected. Promote efficiency and cleaner grids.")
#     if population > 5000000:
#         st.info("👥 Promote sustainable urban development and transport systems.")
#     if prediction > 5:
#         st.warning("⚠️ Emissions are high. Climate policies and lifestyle changes are essential.")
#     else:
#         st.success("✅ Emissions are at a reasonable level!")

#     # Normalize values for fair comparison
#     values = [gdp, energy, population]
#     total = sum(values)
#     norm_values = [v / total for v in values]

#     labels = ['GDP', 'Energy Use', 'Population']
#     colors = ['skyblue', 'lightgreen', 'orange']

#     fig, ax = plt.subplots(figsize=(6, 6))
#     wedges, texts, autotexts = ax.pie(
#         norm_values,
#         labels=labels,
#         autopct='%1.1f%%',
#         startangle=90,
#         colors=colors,
#         pctdistance=0.8,
#         labeldistance=1.15
#     )

#     for text in texts + autotexts:
#         text.set_fontsize(10)
#         text.set_fontweight('bold')

#     ax.axis('equal')
#     st.pyplot(fig)

# # # Footer
# # st.markdown("---")
# # st.markdown("Created for 🏆 GTU Poster Competition 2025 | Powered by OWID Dataset")
