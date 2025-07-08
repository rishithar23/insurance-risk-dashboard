# insurance_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ------------------- Data Simulation -------------------

def load_policy_data():
    np.random.seed(42)
    num_policies = 500
    data = pd.DataFrame({
        'Policy_ID': range(1, num_policies + 1),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], num_policies),
        'Policy_Type': np.random.choice(['Health', 'Auto', 'Home', 'Life'], num_policies),
        'Coverage_Amount': np.random.randint(10000, 1000000, num_policies),
        'Claim_History': np.random.randint(0, 5, num_policies),
        'Customer_Age': np.random.randint(18, 75, num_policies),
        'Premium': np.random.randint(1000, 20000, num_policies),
        'Risk_Score': np.random.rand(num_policies)
    })
    return data

# ------------------- Risk Aggregation -------------------

def calculate_risk_aggregation(df):
    return df.groupby('Region')[['Coverage_Amount', 'Risk_Score']].agg(['sum', 'mean'])

# ------------------- Correlation Analysis -------------------

def plot_correlation(df):
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr()
    fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix")
    return fig

# ------------------- Risk Concentration Modeling -------------------

def cluster_risk(df):
    features = df[['Coverage_Amount', 'Risk_Score', 'Claim_History', 'Premium']]
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=42).fit(reduced)
    df['Cluster'] = kmeans.labels_
    return df, reduced

# ------------------- Stress Testing -------------------

def run_stress_test(df, shock_factor=1.5):
    df_stressed = df.copy()
    df_stressed['Stress_Risk_Score'] = df['Risk_Score'] * shock_factor
    df_stressed['Stress_Risk_Score'] = df_stressed['Stress_Risk_Score'].clip(0, 1)
    return df_stressed

# ------------------- Dashboard -------------------

def main():
    st.set_page_config(page_title="Insurance Risk Management Dashboard", layout="wide")
    st.title(" Insurance Portfolio Risk Management Dashboard")

    # Load and display data
    df = load_policy_data()
    st.sidebar.header("Configuration")
    stress_factor = st.sidebar.slider("Stress Test Factor", 1.0, 3.0, 1.5, 0.1)

    st.subheader(" Sample of Policy Data")
    st.dataframe(df.head())

    # Risk Aggregation
    st.subheader("Risk Aggregation by Region")
    agg_df = calculate_risk_aggregation(df)
    st.dataframe(agg_df)

    # Correlation Analysis
    st.subheader(" Risk Factor Correlation")
    corr_fig = plot_correlation(df)
    st.plotly_chart(corr_fig)

    # Risk Clustering
    st.subheader(" Risk Concentration Clustering")
    clustered_df, reduced = cluster_risk(df)
    cluster_fig = px.scatter(
        x=reduced[:,0], y=reduced[:,1], color=clustered_df['Cluster'].astype(str),
        labels={'x':'PCA 1', 'y':'PCA 2'}, title="Risk Clusters"
    )
    st.plotly_chart(cluster_fig)

    # Stress Testing
    st.subheader(" Stress Testing (Catastrophic Scenario)")
    stressed_df = run_stress_test(df, stress_factor)
    stress_fig = px.histogram(stressed_df, x='Stress_Risk_Score', nbins=30, title="Stressed Risk Score Distribution")
    st.plotly_chart(stress_fig)

    # Alerts
    st.subheader(" Alert System for High Risk")
    high_risk = stressed_df[stressed_df['Stress_Risk_Score'] > 0.85]
    st.warning(f"High Risk Policies Count: {len(high_risk)}")
    st.dataframe(high_risk[['Policy_ID', 'Policy_Type', 'Region', 'Stress_Risk_Score']])

if __name__ == "__main__":
    main()
