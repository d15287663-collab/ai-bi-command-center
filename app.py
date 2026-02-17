import pandas as pd
import streamlit as st
import plotly.express as px
from prophet import Prophet
from sklearn.cluster import KMeans

st.set_page_config(page_title="AI BI Command Center", layout="wide")

st.title("🚀 AI-Powered Business Intelligence Command Center")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("data/superstore.csv")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    return df

df = load_data()

# ================= SIDEBAR FILTERS =================
st.sidebar.header("Filters")

region_filter = st.sidebar.multiselect(
    "Select Region",
    df["Region"].unique(),
    default=df["Region"].unique()
)

df = df[df["Region"].isin(region_filter)]

# ================= KPIs =================
total_sales = df['Sales'].sum()
total_profit = df['Profit'].sum()
orders = df.shape[0]

col1, col2, col3 = st.columns(3)
col1.metric("💰 Revenue", f"₹{total_sales:,.0f}")
col2.metric("📈 Profit", f"₹{total_profit:,.0f}")
col3.metric("🧾 Orders", orders)

st.divider()

# ================= SALES TREND =================
st.subheader("Monthly Sales Trend")

monthly = df.groupby(df['Order Date'].dt.to_period("M"))['Sales'].sum().reset_index()
monthly['Order Date'] = monthly['Order Date'].astype(str)

fig1 = px.line(monthly, x='Order Date', y='Sales', markers=True)
st.plotly_chart(fig1, use_container_width=True)

# ================= REGION PERFORMANCE =================
st.subheader("Region Performance")

region = df.groupby('Region')['Sales'].sum().reset_index()
fig2 = px.bar(region, x='Region', y='Sales', color='Region')
st.plotly_chart(fig2, use_container_width=True)

# ================= CUSTOMER SEGMENTATION =================
st.subheader("🧠 Customer Segmentation")

cust = df.groupby('Customer Name').agg({
    'Sales':'sum',
    'Profit':'sum'
}).reset_index()

kmeans = KMeans(n_clusters=3, random_state=42)
cust['Segment'] = kmeans.fit_predict(cust[['Sales','Profit']])

fig3 = px.scatter(
    cust,
    x='Sales',
    y='Profit',
    color=cust['Segment'].astype(str),
    title="Customer Segments"
)

st.plotly_chart(fig3, use_container_width=True)

# ================= FORECAST =================
st.subheader("🔮 AI Sales Forecast (6 Months)")

forecast_df = df[['Order Date','Sales']]
forecast_df.columns = ['ds','y']

model = Prophet()
model.fit(forecast_df)

future = model.make_future_dataframe(periods=180)
forecast = model.predict(future)

fig4 = px.line(forecast, x='ds', y='yhat', title="Sales Forecast")
st.plotly_chart(fig4, use_container_width=True)

st.success("AI Insight: Sales expected to grow next quarter 📊")
