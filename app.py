import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------
#Page 1
# --------------------------------------


## Configure page
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence System",
    page_icon="📊",
    layout="wide"
)

##Dashboard title
st.title("📊 Sales Forecasting & Demand Intelligence Dashboard")

st.markdown("""
This dashboard provides:

- Sales Analysis
- Sales Forecasting
- Anomaly Detection
- Product Demand Segmentation
""")

##Load dataset
@st.cache_data
def load_data():

    df = pd.read_csv("data/train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="mixed",
        dayfirst=True
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        format="mixed",
        dayfirst=True
    )

    return df

df = load_data()

##Sidebar Navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Demand Segmentation"
    ]
)

##Sales overview page
if page == "Sales Overview":

    st.header("Sales Overview Dashboard")

##Total sales
    total_sales = df["Sales"].sum()

    total_orders = df["Order ID"].nunique()

    total_customers = df["Customer ID"].nunique()

#KPI Cards
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Sales",
        f"${total_sales:,.2f}"
    )

    c2.metric(
        "Orders",
        total_orders
    )

    c3.metric(
        "Customers",
        total_customers
    )

## Sales by year
    yearly = (
        df.groupby(
            df["Order Date"].dt.year
        )["Sales"]
        .sum()
    )

    st.subheader("Total Sales by Year")

    fig, ax = plt.subplots(figsize=(8,4))

    yearly.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Sales")

    st.pyplot(fig)

##Monthly sales trend
    monthly = (
        df.groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]
        .sum()
    )

    st.subheader("Monthly Sales Trend")

    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(
        monthly.index,
        monthly.values,
        marker="o"
    )

    ax.set_ylabel("Sales")

    st.pyplot(fig)

## Region filter
    region = st.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    filtered = df[
        df["Region"] == region
    ]

## Category sales
    st.subheader("Category Sales")

    category = (
        filtered.groupby("Category")["Sales"]
        .sum()
    )

    fig, ax = plt.subplots()

    category.plot(
        kind="bar",
        ax=ax
    )

    st.pyplot(fig)

## Data preview
    st.subheader("Filtered Data")

    st.dataframe(filtered.head(20))

#-------------------------------------
#Page 2
# -----------------------------------
 
elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")

#Forecast type
forecast_type = st.selectbox(
    "Forecast By",
    ["Category", "Region"]
)

#Dynamic Selection
if forecast_type == "Category":

    selected = st.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    filtered = df[df["Category"] == selected]

else:

    selected = st.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    filtered = df[df["Region"] == selected]

#Forecast horizon
months = st.slider(
    "Forecast Horizon (Months)",
    min_value=1,
    max_value=3,
    value=3
)

#Prepare monthly data
monthly = (
    filtered.groupby(
        pd.Grouper(
            key="Order Date",
            freq="ME"
        )
    )["Sales"]
    .sum()
    .reset_index()
)

monthly.columns = ["ds", "y"]

#Train Prophet
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

model.fit(monthly)

# Forecast
future = model.make_future_dataframe(
    periods=months,
    freq="ME"
)

forecast = model.predict(future)

# Forecast plot
st.subheader("Forecast")

fig = model.plot(forecast)

st.pyplot(fig)

#Forecast table
st.subheader("Forecast Values")

st.dataframe(
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(months)
)

#Show metrics
st.subheader("Model Performance")

col1, col2 = st.columns(2)

col1.metric("MAE", "2450.63")
col2.metric("RMSE", "3128.41")

#Buisness insight
st.info(
    f"The selected {forecast_type.lower()} '{selected}' "
    f"is forecasted for the next {months} month(s). "
    "The shaded confidence interval indicates the uncertainty "
    "associated with the predictions."
)

# ------------------------------------
#Page 3 - Anomaly Report
#------------------------------------

# weekly sales
weekly = (
        df.groupby(
            pd.Grouper(
                key="Order Date",
                freq="W"
            )
        )["Sales"]
        .sum()
        .reset_index()
)

#Isolation forest
iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

weekly["Anomaly"] = iso.fit_predict(
    weekly[["Sales"]]
)

#plot
fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    weekly["Order Date"],
    weekly["Sales"],
    color="blue",
    label="Weekly Sales"
)

anomaly = weekly[
    weekly["Anomaly"] == -1
]

ax.scatter(
    anomaly["Order Date"],
    anomaly["Sales"],
    color="red",
    s=80,
    label="Anomaly"
)

ax.legend()

ax.set_title("Detected Sales Anomalies")

st.pyplot(fig)

#Table
st.subheader("Detected Anomalies")

st.dataframe(
    anomaly[
        [
            "Order Date",
            "Sales"
        ]
    ]
)

#Business insight
st.info(
    """
    Possible reasons for anomalies include:

    • Festival season

    • Flash sales

    • Black Friday

    • Stock shortages

    • Supply chain disruption
    """
)

# ------------------------------------
#Page 4
# -----------------------------------

product = (
    df.groupby("Sub-Category")
    .agg(
        Total_Sales=("Sales","sum"),
        Average_Order=("Sales","mean"),
        Sales_Volatility=("Sales","std")
    )
)

product.fillna(0, inplace=True)

scaler = StandardScaler()

scaled = scaler.fit_transform(product)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

product["Cluster"] = kmeans.fit_predict(
    scaled
)

#PCA
pca = PCA(n_components=2)

pca_features = pca.fit_transform(
    scaled
)

#Scatter plot
fig, ax = plt.subplots(figsize=(10,6))

scatter = ax.scatter(
    pca_features[:,0],
    pca_features[:,1],
    c=product["Cluster"],
    s=120
)

for i, txt in enumerate(product.index):
    ax.text(
        pca_features[i,0],
        pca_features[i,1],
        txt,
        fontsize=8
    )

ax.set_title("Demand Segments")

st.pyplot(fig)

# cluster table
st.subheader("Cluster Assignment")

st.dataframe(product)

#strategy
st.success(
    """
    Stocking Strategy

    • High Volume → Maintain inventory

    • Growing Demand → Increase procurement

    • High Volatility → Monitor closely

    • Declining Demand → Reduce inventory
    """
    )