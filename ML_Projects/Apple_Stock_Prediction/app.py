import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

import onnxruntime as ort

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AAPL Stock Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }
    .sub-title  { font-size: 1rem; color: #6c757d; margin-bottom: 2rem; }
    div[data-testid="stSidebar"] { background: #f0f4f8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ASSET PATHS
# ─────────────────────────────────────────────
# Resolve paths relative to this script's location
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR   = os.path.join(SCRIPT_DIR, "assets")
MODEL_PATH   = os.path.join(ASSETS_DIR, "final_gru_model.onnx")
SCALER_PATH  = os.path.join(ASSETS_DIR, "final_scaler_gru.pkl")
DATA_PATH    = os.path.join(ASSETS_DIR, "apple_df_cleaned.pkl")

# ─────────────────────────────────────────────
# LOAD ASSETS (cached — loads only once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    """Load pre-trained ONNX model, scaler, and cleaned dataframe from disk."""
    missing = []
    for p in [MODEL_PATH, SCALER_PATH, DATA_PATH]:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        return None, None, None, missing

    session = ort.InferenceSession(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    df = pd.read_pickle(DATA_PATH)
    return session, scaler, df, []


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">📈 Apple (AAPL) Stock Price Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">GRU Deep Learning Model · 30-Day Forecast · P668 Project by Avinash</div>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────
with st.spinner("Loading pre-trained model and data..."):
    gru_session, final_scaler_gru, apple_df_cleaned, missing_files = load_assets()

if missing_files:
    st.error("❌ Required asset files not found. Please make sure these files are in the `assets/` folder:")
    for f in missing_files:
        st.code(f)
    st.info("""
**How to generate these files:**
Run this in your Google Colab notebook at the end:

```python
import pickle, os
os.makedirs('/content/streamlit_assets', exist_ok=True)

# Save model as ONNX (requires: !pip install tf2onnx onnx)
import tf2onnx, onnx, tensorflow as tf
saved_model_path = "/content/streamlit_assets/temp_saved_model"
final_gru_model.export(saved_model_path)
!python -m tf2onnx.convert --saved-model {saved_model_path} --output /content/streamlit_assets/final_gru_model.onnx --opset 13

# Save scaler and dataframe
with open('/content/streamlit_assets/final_scaler_gru.pkl', 'wb') as f:
    pickle.dump(final_scaler_gru, f)
apple_df_cleaned.to_pickle('/content/streamlit_assets/apple_df_cleaned.pkl')
```
Then download all 3 files and place them in the `assets/` folder.
    """)
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR — USER INPUTS
# ─────────────────────────────────────────────
data_min = apple_df_cleaned.index.min().date()
data_max = apple_df_cleaned.index.max().date()

with st.sidebar:
    st.header("⚙️ Forecast Settings")
    st.markdown("---")

    st.subheader("📅 Historical View Range")
    st.caption(f"Available data: {data_min} to {data_max}")

    view_start = st.date_input(
        "View history from",
        value=data_min,
        min_value=data_min,
        max_value=data_max
    )
    view_end = st.date_input(
        "View history to",
        value=data_max,
        min_value=data_min,
        max_value=data_max
    )

    st.markdown("---")
    st.subheader("🔮 Forecast Settings")
    forecast_days = st.slider("Forecast horizon (days)", min_value=7, max_value=30, value=30, step=1)
    time_steps    = st.selectbox(
        "Time steps used during training",
        [30, 60, 90],
        index=1,
        help="Must match the time_steps used when training the model in your notebook (default: 60)"
    )

    st.markdown("---")
    run_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

    st.markdown("---")
    st.success(f"✅ Model loaded from `assets/`\n\nNo retraining needed!")
    st.caption(f"**Model:** GRU (ONNX format)\n**Data:** {len(apple_df_cleaned):,} trading days\n**Project:** P668 — Avinash")

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def plot_history(df):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), facecolor='white')
    fig.suptitle('Historical Price Analysis', fontsize=15, fontweight='bold', y=0.98)

    # Close + Bollinger Bands
    if 'Upper_Band' in df.columns and 'Lower_Band' in df.columns:
        axes[0].fill_between(df.index, df['Lower_Band'], df['Upper_Band'],
                             alpha=0.15, color='#0077b6', label='Bollinger Bands')
    axes[0].plot(df['Close'], color='#0077b6', linewidth=1.5, label='Close Price')
    if 'SMA_30' in df.columns:
        axes[0].plot(df['SMA_30'], color='#f4a261', linewidth=1, linestyle='--', label='SMA 30')
    axes[0].set_title('Close Price with Bollinger Bands')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor('#fafafa')

    # RSI
    if 'RSI' in df.columns:
        axes[1].plot(df['RSI'], color='#7b2d8b', linewidth=1)
        axes[1].axhline(70, color='red',   linestyle='--', linewidth=0.8, label='Overbought (70)')
        axes[1].axhline(30, color='green', linestyle='--', linewidth=0.8, label='Oversold (30)')
        axes[1].set_title('RSI (14-period)')
        axes[1].set_ylim(0, 100)
        axes[1].legend(fontsize=9)
    else:
        axes[1].set_title('RSI — not available')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor('#fafafa')

    # Volume
    axes[2].bar(df.index, df['Volume'], color='#adb5bd', width=1)
    axes[2].set_title('Trading Volume')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_facecolor('#fafafa')

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    return fig


def plot_forecast(df_clean, forecast_table, n_history=100):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('#fafafa')

    hist = df_clean['Close'].tail(n_history)
    ax.plot(hist.index, hist.values,
            color='#0077b6', linewidth=2, label=f'Historical (last {n_history} days)')

    lower = forecast_table['Predicted Close ($)'] * 0.95
    upper = forecast_table['Predicted Close ($)'] * 1.05
    ax.fill_between(forecast_table.index, lower, upper,
                    alpha=0.2, color='#e63946', label='±5% Confidence Band')

    ax.plot(forecast_table.index, forecast_table['Predicted Close ($)'],
            color='#e63946', linestyle='--', marker='o', markersize=4,
            linewidth=2, label=f'{len(forecast_table)}-Day GRU Forecast')

    last_date = df_clean.index[-1]
    ax.axvline(x=last_date, color='gray', linestyle=':', linewidth=1.5, label='Forecast Start')

    sp = forecast_table['Predicted Close ($)'].iloc[0]
    ep = forecast_table['Predicted Close ($)'].iloc[-1]

    ax.annotate(f'${sp:.2f}',
                xy=(forecast_table.index[0], sp),
                xytext=(8, 15), textcoords='offset points',
                fontsize=10, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred'))
    ax.annotate(f'${ep:.2f}',
                xy=(forecast_table.index[-1], ep),
                xytext=(-60, 15), textcoords='offset points',
                fontsize=10, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred'))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_title('Apple (AAPL) 30-Day Stock Price Forecast — GRU Model',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# ONNX PREDICTION HELPER
# ─────────────────────────────────────────────
def onnx_predict(session, input_data):
    """Run prediction using ONNX runtime session."""
    input_name = session.get_inputs()[0].name
    input_data = input_data.astype(np.float32)
    result = session.run(None, {input_name: input_data})
    return result[0]


# ─────────────────────────────────────────────
# WELCOME SCREEN (before button click)
# ─────────────────────────────────────────────
if not run_btn:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Step 1**\n\nSelect the date range you want to view historical data for.")
    with c2:
        st.info("**Step 2**\n\nSet the forecast horizon (up to 30 days) and time steps.")
    with c3:
        st.info("**Step 3**\n\nClick **Generate Forecast** — results appear instantly (no training wait!).")

    st.markdown("### About this app")
    st.markdown(f"""
This app loads a **pre-trained GRU model** (ONNX format) and generates a stock price forecast instantly.

| Detail | Value |
|---|---|
| Ticker | Apple Inc. (AAPL) |
| Training data | {data_min} to {data_max} |
| Total records | {len(apple_df_cleaned):,} trading days |
| Model | GRU (2-layer, 100 units, dropout 0.2) |
| Runtime | ONNX Runtime (lightweight, no TensorFlow needed) |
| Forecast method | Recursive multi-step prediction |

> ⚠️ For academic/educational purposes only. Not financial advice.
    """)

# ─────────────────────────────────────────────
# FORECAST LOGIC (after button click)
# ─────────────────────────────────────────────
else:
    if view_start >= view_end:
        st.error("View start date must be before view end date.")
        st.stop()

    # Filter dataframe for selected view range
    mask = (apple_df_cleaned.index.date >= view_start) & (apple_df_cleaned.index.date <= view_end)
    df_view = apple_df_cleaned.loc[mask]

    if len(df_view) < 60:
        st.warning("Selected view range is very short. Showing all available data for charts.")
        df_view = apple_df_cleaned

    # ── Dataset Summary ──
    st.subheader("📊 Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total trading days (full dataset)", f"{len(apple_df_cleaned):,}")
    c2.metric("Viewing range", f"{view_start} – {view_end}")
    c3.metric("Min Close (selected)", f"${df_view['Close'].min():.2f}")
    c4.metric("Max Close (selected)", f"${df_view['Close'].max():.2f}")

    # ── Historical Charts ──
    st.subheader("📉 Historical Price Analysis")
    with st.spinner("Plotting..."):
        fig_hist = plot_history(df_view)
        st.pyplot(fig_hist)
        plt.close()

    # ── Recursive Forecast (instant — no retraining) ──
    st.subheader("🔮 Generating 30-Day Forecast")

    with st.spinner("Running recursive forecast..."):
        close_vals  = apple_df_cleaned[['Close']].values
        scaled_data = final_scaler_gru.transform(close_vals)   # transform only — scaler already fitted!

        if len(scaled_data) < time_steps:
            st.error(f"Not enough data ({len(scaled_data)} rows) for time_steps={time_steps}.")
            st.stop()

        last_seq     = scaled_data[-time_steps:].copy()
        preds_scaled = []

        for _ in range(forecast_days):
            inp  = last_seq.reshape(1, time_steps, 1)
            pred = onnx_predict(gru_session, inp)
            preds_scaled.append(pred[0, 0])
            last_seq = np.append(last_seq[1:], pred[0, 0]).reshape(-1, 1)

        predictions = final_scaler_gru.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        )

    last_date    = apple_df_cleaned.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df  = pd.DataFrame(
        {'Predicted Close ($)': predictions.flatten()},
        index=future_dates
    )

    st.success("✅ Forecast generated instantly from pre-trained model!")

    # ── Metrics ──
    sp         = forecast_df['Predicted Close ($)'].iloc[0]
    ep         = forecast_df['Predicted Close ($)'].iloc[-1]
    change_pct = ((ep - sp) / sp) * 100
    last_price = apple_df_cleaned['Close'].iloc[-1]

    st.subheader("📈 Forecast Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Known Price",          f"${last_price:.2f}")
    c2.metric("Forecast Day 1",            f"${sp:.2f}")
    c3.metric(f"Forecast Day {forecast_days}", f"${ep:.2f}", delta=f"{change_pct:+.2f}%")
    c4.metric("Forecast End Date",         future_dates[-1].strftime('%d %b %Y'))

    # ── Forecast Chart ──
    fig_fc = plot_forecast(apple_df_cleaned, forecast_df)
    st.pyplot(fig_fc)
    plt.close()

    # ── Interpretation Note ──
    if abs(change_pct) > 20:
        st.warning(f"""
**⚠️ Forecast Interpretation Note**
The model predicts a **{change_pct:+.1f}%** change over {forecast_days} days.
Recursive deep learning forecasts are known to amplify trends (recursive drift).
Treat this as a **directional trend indicator**, not an exact price target.
        """)

    # ── Validation ──
    hist_vol = apple_df_cleaned['Close'].tail(60).pct_change().std()
    fore_vol = forecast_df['Predicted Close ($)'].pct_change().std()

    with st.expander("🔍 Forecast Validation Report"):
        st.markdown(f"""
| Metric | Value |
|---|---|
| Last known price | ${last_price:.2f} |
| Forecast start price | ${sp:.2f} |
| Forecast end price | ${ep:.2f} |
| Total expected change | {change_pct:+.2f}% |
| Historical daily volatility (60d) | {hist_vol:.4f} |
| Forecasted daily volatility | {fore_vol:.4f} |
| Model | Pre-trained GRU · ONNX runtime · loaded from `assets/` |
| Time steps | {time_steps} |
        """)
        if fore_vol > hist_vol * 2:
            st.warning("Forecasted volatility is significantly higher than historical levels.")
        elif fore_vol < hist_vol * 0.1:
            st.info("Forecasted volatility is very low — common in recursive deep learning models.")
        else:
            st.success("Forecasted volatility is within a reasonable range of historical volatility.")

    # ── Full Forecast Table ──
    st.subheader("📋 Full Forecast Table")
    display_df = forecast_df.copy()
    display_df.index = display_df.index.strftime('%A, %d %b %Y')
    display_df['Predicted Close ($)'] = display_df['Predicted Close ($)'].map('${:.2f}'.format)
    display_df.index.name = 'Date'
    st.dataframe(display_df, use_container_width=True)

    # ── CSV Download ──
    csv_df = forecast_df.copy()
    csv_df.index = csv_df.index.strftime('%Y-%m-%d')
    st.download_button(
        label="⬇️ Download Forecast as CSV",
        data=csv_df.to_csv().encode('utf-8'),
        file_name=f"AAPL_forecast_{future_dates[0].strftime('%Y%m%d')}.csv",
        mime='text/csv',
        use_container_width=True
    )

    st.markdown("---")
    st.caption("📌 P668 Project — Apple Stock Price Prediction | GRU Deep Learning Model (ONNX) | For academic purposes only")
