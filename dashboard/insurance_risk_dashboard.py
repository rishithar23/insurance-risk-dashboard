import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------ Sample Data ------------------------
def create_sample_data():
    data = {
        'policy_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'region': ['North', 'South', 'East', 'West', 'North'],
        'product_type': ['Life', 'Auto', 'Health', 'Property', 'Auto'],
        'insured_value': [100000, 50000, 200000, 300000, 75000],
        'exposure_score': [0.3, 0.5, 0.2, 0.4, 0.6]
    }
    df = pd.DataFrame(data)
    return df

# ------------------------ Risk Calculations ------------------------
def calculate_total_exposure(df):
    return df['insured_value'].sum()

def calculate_risk_by_region(df):
    return df.groupby('region')['insured_value'].sum()

def calculate_risk_by_product(df):
    return df.groupby('product_type')['insured_value'].sum()

# ------------------------ Correlation Analysis ------------------------
def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# ------------------------ Stress Testing ------------------------
def simulate_catastrophe(df, region_affected, impact_factor=0.5):
    df_copy = df.copy()
    df_copy.loc[df_copy['region'] == region_affected, 'insured_value'] *= impact_factor
    return df_copy

# ------------------------ Dashboard Visualization ------------------------
def plot_risk_distribution_by_region(risk_data):
    risk_data.plot(kind='bar', title='Risk by Region')
    plt.ylabel("Insured Value")
    plt.show()

def plot_risk_distribution_by_product(risk_data):
    risk_data.plot(kind='bar', color='orange', title='Risk by Product')
    plt.ylabel("Insured Value")
    plt.show()

# ------------------------ Main Execution ------------------------
def main():
    print("📊 Loading Insurance Portfolio...")
    df = create_sample_data()
    print(df)

    print("\n💰 Total Portfolio Exposure:", calculate_total_exposure(df))

    region_risk = calculate_risk_by_region(df)
    product_risk = calculate_risk_by_product(df)

    print("\n📌 Risk by Region:\n", region_risk)
    print("\n📌 Risk by Product:\n", product_risk)

    print("\n📉 Plotting Risk Distribution by Region...")
    plot_risk_distribution_by_region(region_risk)

    print("\n📉 Plotting Risk Distribution by Product...")
    plot_risk_distribution_by_product(product_risk)

    print("\n📊 Correlation Analysis...")
    plot_correlation_matrix(df)

    print("\n⚠️ Running Stress Test (Catastrophe in North region)...")
    stressed_df = simulate_catastrophe(df, region_affected="North", impact_factor=0.2)
    new_exposure = calculate_total_exposure(stressed_df)
    print("💣 Exposure after Stress Test:", new_exposure)

# ------------------------ Entry Point ------------------------
if __name__ == "__main__":
    main()
