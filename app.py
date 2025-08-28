import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
import os

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    file_path = "spacex_launch_dash.csv"   # Make sure file is saved in project folder
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error("CSV file not found! Please save spacex_launch_dash.csv in your project folder.")
        return None

df = load_data()

# ----------------------------
# Add Launch Site Coordinates (Manually)
# ----------------------------
launch_sites_coords = pd.DataFrame({
    'Launch Site': ['CCAFS LC-40', 'CCAFS SLC-40', 'KSC LC-39A', 'VAFB SLC-4E'],
    'Lat': [28.5623, 28.5632, 28.5733, 34.6321],
    'Long': [-80.5774, -80.5773, -80.6469, -120.6106]
})

# Merge with main dataframe to ensure coords available
if df is not None:
    df = df.merge(launch_sites_coords, on="Launch Site", how="left")

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.set_page_config(page_title="SpaceX Prediction Project", layout="wide")
st.title("🚀 SpaceX Launch Analysis & Prediction Dashboard")

menu = st.sidebar.radio(
    "Select Section:",
    ["EDA", "Geospatial Analysis", "ML Models", "Dashboard"]
)

# ----------------------------
# Section 1: EDA
# ----------------------------
if menu == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    st.write("### First 5 rows of dataset")
    st.dataframe(df.head())

    st.write("### Success vs Failure Count")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='class', ax=ax)
    st.pyplot(fig)

    st.write("### Payload Mass Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Payload Mass (kg)'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

# ----------------------------
# Section 2: Geospatial Analysis
# ----------------------------
elif menu == "Geospatial Analysis":
    st.header("🌍 Geospatial Analysis of Launch Sites")

    if "Lat" in df.columns and "Long" in df.columns:
        m = folium.Map(location=[df["Lat"].mean(), df["Long"].mean()], zoom_start=4)
        for _, row in df.iterrows():
            if pd.notna(row["Lat"]) and pd.notna(row["Long"]):
                folium.Marker([row["Lat"], row["Long"]],
                              popup=row["Launch Site"]).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.warning("No Latitude/Longitude columns found in dataset!")

# ----------------------------
# Section 3: ML Models
# ----------------------------
elif menu == "ML Models":
    st.header("🤖 Machine Learning Models for Launch Prediction")

    X = df[['Payload Mass (kg)']]  # demo only
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = accuracy_score(y_test, preds)

    st.write("### Model Accuracy Comparison")
    st.write(results)

    best_model = max(results, key=results.get)
    st.success(f"Best Model: {best_model} with Accuracy = {results[best_model]:.2f}")

# ----------------------------
# Section 4: Dashboard
# ----------------------------
elif menu == "Dashboard":
    st.header("📊 Interactive Dashboard (Payload & Launch Sites)")

    site = st.selectbox("Select Launch Site", ["ALL"] + list(df["Launch Site"].unique()))
    payload = st.slider("Select Payload Range (Kg)",
                        int(df['Payload Mass (kg)'].min()),
                        int(df['Payload Mass (kg)'].max()),
                        (2000, 8000))

    # Pie Chart
    if site == "ALL":
        fig = px.pie(df, values='class', names='Launch Site',
                     title='Total Success Launches by Site')
    else:
        filtered_df = df[df['Launch Site'] == site]
        fig = px.pie(filtered_df, names='class',
                     title=f'Success vs Failure for {site}')
    st.plotly_chart(fig)

    # Scatter Plot
    mask = (df['Payload Mass (kg)'] >= payload[0]) & (df['Payload Mass (kg)'] <= payload[1])
    filtered_df = df[mask]
    if site != "ALL":
        filtered_df = filtered_df[filtered_df['Launch Site'] == site]

    fig2 = px.scatter(filtered_df, x='Payload Mass (kg)', y='class',
                      color='Booster Version Category',
                      title='Payload vs Outcome')
    st.plotly_chart(fig2)













