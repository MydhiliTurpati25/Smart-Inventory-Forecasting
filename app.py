
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Smart Inventory AI",
    page_icon="📦",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4f7fb, #eaf2f8);
}
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dce6ef;
}
h1, h2, h3 { color: #12304a; }
</style>
""", unsafe_allow_html=True)

st.title("📦 Smart Inventory & Demand Forecasting")
st.caption("AI-powered inventory planning and demand intelligence")

@st.cache_data
def sample_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(),
                          periods=90, freq="D")
    products = {
        "Wireless Mouse": (22, 140, 799),
        "Keyboard": (15, 90, 1299),
        "USB Cable": (35, 180, 299),
        "Headphones": (12, 70, 1999),
        "Laptop Stand": (8, 45, 1499),
    }
    rows = []
    for product, (mean, stock, price) in products.items():
        for i, date in enumerate(dates):
            seasonal = 1 + 0.18 * np.sin(i * 2 * np.pi / 7)
            qty = max(0, int(rng.poisson(mean * seasonal)))
            rows.append({
                "date": date,
                "product": product,
                "quantity_sold": qty,
                "stock": stock,
                "unit_price": price,
                "lead_time_days": 5
            })
    return pd.DataFrame(rows)

def prepare_data(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_")
                  for c in df.columns]

    aliases = {
        "product_name": "product",
        "item": "product",
        "sales": "quantity_sold",
        "units_sold": "quantity_sold",
        "quantity": "quantity_sold",
        "inventory": "stock",
        "current_stock": "stock",
        "price": "unit_price",
        "selling_price": "unit_price",
        "sales_date": "date",
    }
    df = df.rename(columns={
        c: aliases[c] for c in df.columns if c in aliases
    })

    required = ["date", "product", "quantity_sold", "stock"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["product"] = df["product"].astype("string").str.strip()

    for c in ["quantity_sold", "stock"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "unit_price" not in df.columns:
        df["unit_price"] = 0.0
    df["unit_price"] = pd.to_numeric(
        df["unit_price"], errors="coerce"
    ).fillna(0)

    if "lead_time_days" not in df.columns:
        df["lead_time_days"] = 5
    df["lead_time_days"] = pd.to_numeric(
        df["lead_time_days"], errors="coerce"
    ).fillna(5).clip(lower=1)

    df = df.dropna(
        subset=["date", "product", "quantity_sold", "stock"]
    )
    df = df[df["product"].ne("")]
    df = df[
        (df["quantity_sold"] >= 0) &
        (df["stock"] >= 0) &
        (df["unit_price"] >= 0)
    ]
    if df.empty:
        raise ValueError("No valid records remain after data cleaning.")

    # One daily sales total per product and date.
    daily = df.groupby(
        ["date", "product"], as_index=False
    ).agg(
        quantity_sold=("quantity_sold", "sum"),
        stock=("stock", "last"),
        unit_price=("unit_price", "last"),
        lead_time_days=("lead_time_days", "max")
    )
    return daily

def forecast_product(history, days=7):
    history = history.sort_values("date").copy()
    history = history.set_index("date").asfreq("D")
    history["quantity_sold"] = history["quantity_sold"].fillna(0)

    y = history["quantity_sold"].clip(lower=0)
    future_dates = pd.date_range(
        history.index.max() + pd.Timedelta(days=1),
        periods=days, freq="D"
    )

    # Require sufficient history before fitting the ML model.
    if len(y) >= 21 and y.nunique() > 1:
        frame = pd.DataFrame({"y": y})
        frame["day"] = np.arange(len(frame))
        frame["weekday"] = frame.index.dayofweek
        frame["lag7"] = frame["y"].shift(7)
        frame = frame.dropna()

        if len(frame) >= 10:
            model = RandomForestRegressor(
                n_estimators=100, random_state=42, min_samples_leaf=2
            )
            model.fit(frame[["day", "weekday", "lag7"]], frame["y"])
            history_values = list(y.values)
            predictions = []
            for date in future_dates:
                lag7 = history_values[-7]
                x = pd.DataFrame([{
                    "day": len(history_values),
                    "weekday": date.dayofweek,
                    "lag7": lag7
                }])
                pred = max(0, float(model.predict(x)[0]))
                predictions.append(pred)
                history_values.append(pred)
            method = "Random Forest ML"
            return future_dates, predictions, method

    # Fallback for short or nearly constant histories.
    window = y.tail(min(7, len(y)))
    avg = max(0, float(window.mean()))
    predictions = [avg] * days
    return future_dates, predictions, "Recent average fallback"

st.sidebar.header("⚙️ Controls")
uploaded = st.sidebar.file_uploader(
    "Upload inventory CSV", type=["csv"]
)
forecast_days = st.sidebar.slider(
    "Forecast horizon (days)", 7, 30, 7
)
safety_days = st.sidebar.slider(
    "Safety stock (days of demand)", 1, 14, 3
)

try:
    raw = pd.read_csv(uploaded) if uploaded else sample_data()
    data = prepare_data(raw)
except Exception as exc:
    st.error(f"Could not process the data: {exc}")
    st.info(
        "Required CSV columns: date, product, quantity_sold, stock. "
        "Optional: unit_price, lead_time_days."
    )
    st.stop()

st.success(
    f"Loaded {len(data):,} valid daily product records across "
    f"{data['product'].nunique()} products."
)

latest = data.sort_values("date").groupby("product").tail(1).copy()
latest["inventory_value"] = latest["stock"] * latest["unit_price"]

forecast_rows = []
for product, hist in data.groupby("product"):
    dates, preds, method = forecast_product(hist, forecast_days)
    current = hist.sort_values("date").iloc[-1]
    avg_daily = float(np.mean(preds)) if preds else 0
    lead = int(current["lead_time_days"])
    safety_stock = avg_daily * safety_days
    reorder_point = avg_daily * lead + safety_stock
    stock = float(current["stock"])
    reorder_qty = max(
        0, int(np.ceil(avg_daily * (lead + safety_days) - stock))
    )

    for date, pred in zip(dates, preds):
        forecast_rows.append({
            "date": date,
            "product": product,
            "forecast_demand": round(pred, 2),
            "method": method
        })

forecasts = pd.DataFrame(forecast_rows)
summary = latest[[
    "product", "stock", "unit_price", "lead_time_days",
    "inventory_value"
]].merge(
    forecasts.groupby("product", as_index=False).agg(
        avg_forecast=("forecast_demand", "mean"),
        total_forecast=("forecast_demand", "sum"),
        method=("method", "first")
    ),
    on="product", how="left"
)

summary["reorder_point"] = (
    summary["avg_forecast"] *
    (summary["lead_time_days"] + safety_days)
)
summary["reorder_qty"] = np.maximum(
    0, np.ceil(summary["reorder_point"] - summary["stock"])
).astype(int)
summary["status"] = np.where(
    summary["stock"] <= summary["reorder_point"], "Reorder",
    np.where(
        summary["stock"] > summary["total_forecast"] * 2,
        "Possible Overstock", "Healthy"
    )
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products", f"{summary['product'].nunique():,}")
c2.metric("Inventory Units", f"{summary['stock'].sum():,.0f}")
c3.metric("Inventory Value", f"₹{summary['inventory_value'].sum():,.0f}")
c4.metric(
    "Products to Reorder",
    f"{(summary['status'] == 'Reorder').sum():,}"
)

st.divider()
st.subheader("📊 Demand Forecast")

selected = st.selectbox(
    "Choose a product", sorted(summary["product"].unique())
)
hist_selected = data[data["product"] == selected].sort_values("date")
future_selected = forecasts[forecasts["product"] == selected]

fig = px.line(
    hist_selected.tail(45), x="date", y="quantity_sold",
    title=f"{selected}: Historical Daily Sales",
    labels={"date": "Date", "quantity_sold": "Units sold"},
    markers=True
)
fig.add_scatter(
    x=future_selected["date"],
    y=future_selected["forecast_demand"],
    mode="lines+markers", name="Forecast"
)
fig.update_layout(
    template="plotly_white", legend_title_text="",
    margin=dict(l=15, r=15, t=55, b=15)
)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("📦 Stock by Product")
    stock_fig = px.bar(
        summary.sort_values("stock"),
        x="stock", y="product", orientation="h",
        color="status",
        color_discrete_map={
            "Reorder": "#e45756",
            "Healthy": "#26a69a",
            "Possible Overstock": "#f2b134"
        }
    )
    stock_fig.update_layout(template="plotly_white")
    st.plotly_chart(stock_fig, use_container_width=True)

with right:
    st.subheader("🚦 Inventory Status")
    counts = summary["status"].value_counts().rename_axis(
        "status"
    ).reset_index(name="products")
    status_fig = px.pie(
        counts, names="status", values="products", hole=0.55
    )
    status_fig.update_layout(template="plotly_white")
    st.plotly_chart(status_fig, use_container_width=True)

st.divider()
st.subheader("🧠 Inventory Recommendations")
display = summary[[
    "product", "stock", "avg_forecast", "total_forecast",
    "reorder_point", "reorder_qty", "status", "method"
]].rename(columns={
    "product": "Product",
    "stock": "Current Stock",
    "avg_forecast": "Avg Daily Forecast",
    "total_forecast": "Forecast Total",
    "reorder_point": "Reorder Threshold",
    "reorder_qty": "Suggested Reorder Qty",
    "status": "Status",
    "method": "Forecast Method"
})
st.dataframe(display, use_container_width=True, hide_index=True)

csv = summary.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Inventory Recommendations CSV",
    data=csv,
    file_name="inventory_recommendations.csv",
    mime="text/csv"
)

st.caption(
    "Forecasts are estimates based on historical sales. "
    "The sample dataset is synthetic. Validate results before making "
    "real purchasing decisions."
)
