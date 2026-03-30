import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="ExDark LLIE Demo", page_icon="EA", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif;
}

.block-container {
    padding-top: 1.2rem;
}

.hero {
    border-radius: 18px;
    padding: 18px 24px;
    background: linear-gradient(120deg, #102a43 0%, #1f7a8c 55%, #f4a261 100%);
    color: #ffffff;
    box-shadow: 0 10px 30px rgba(16, 42, 67, 0.2);
}

.pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    margin-right: 6px;
    font-size: 0.78rem;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.28);
}

.kpi-card {
    border: 1px solid #d8e2dc;
    border-radius: 14px;
    padding: 12px 14px;
    background: #f8fbfa;
}

.section-title {
    margin-top: 0.4rem;
    margin-bottom: 0.2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

SCENARIOS = [
    "S1_Raw",
    "S2_HVI_CIDNet",
    "S3_RetinexFormer",
    "S4_LYT_Net",
]

MOCK_METRICS = pd.DataFrame(
    [
        {"Scenario": "S1_Raw", "mAP50": 0.516, "Precision": 0.642, "Recall": 0.571, "NIQE": 4.81, "BRISQUE": 33.4, "LOE": 0.0, "Latency_ms": 12.3, "GFLOPs": 6.5},
        {"Scenario": "S2_HVI_CIDNet", "mAP50": 0.553, "Precision": 0.671, "Recall": 0.602, "NIQE": 3.72, "BRISQUE": 27.8, "LOE": 178.2, "Latency_ms": 24.1, "GFLOPs": 9.8},
        {"Scenario": "S3_RetinexFormer", "mAP50": 0.561, "Precision": 0.680, "Recall": 0.611, "NIQE": 3.54, "BRISQUE": 26.9, "LOE": 151.0, "Latency_ms": 29.7, "GFLOPs": 11.2},
        {"Scenario": "S4_LYT_Net", "mAP50": 0.548, "Precision": 0.667, "Recall": 0.600, "NIQE": 3.66, "BRISQUE": 28.5, "LOE": 165.7, "Latency_ms": 18.5, "GFLOPs": 8.1},
    ]
)


def make_placeholder(label: str, width: int = 540, height: int = 300) -> Image.Image:
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)
    r = (35 + 120 * xv).astype(np.uint8)
    g = (50 + 140 * yv).astype(np.uint8)
    b = (90 + 90 * (1 - xv * yv)).astype(np.uint8)
    arr = np.stack([r, g, b], axis=2)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.rectangle((16, 16, width - 16, height - 16), outline=(245, 245, 245), width=3)
    draw.text((26, 26), label, fill=(255, 255, 255))
    return img


with st.sidebar:
    st.header("Control Panel")
    selected_scenario = st.selectbox("Scenario Focus", SCENARIOS, index=3)
    selected_image = st.selectbox("Sample Image", [f"test_{i:03d}.jpg" for i in range(1, 13)], index=2)
    conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    show_gt = st.toggle("Show Ground Truth Column", value=True)
    st.caption("UI mode: mock data")

st.markdown(
    f"""
<div class="hero">
  <h2 style="margin:0;">ExDark LLIE x YOLO Demo Dashboard</h2>
  <p style="margin:6px 0 10px 0;">UI prototype for scenario comparison and qualitative inspection.</p>
  <span class="pill">Scenario Focus: {selected_scenario}</span>
  <span class="pill">Image: {selected_image}</span>
  <span class="pill">Conf: {conf:.2f}</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### Quick KPIs")
focus_row = MOCK_METRICS[MOCK_METRICS["Scenario"] == selected_scenario].iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='kpi-card'><b>mAP@0.5</b><br><span style='font-size:1.5rem'>{focus_row['mAP50']:.3f}</span></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi-card'><b>Precision</b><br><span style='font-size:1.5rem'>{focus_row['Precision']:.3f}</span></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='kpi-card'><b>NIQE</b><br><span style='font-size:1.5rem'>{focus_row['NIQE']:.2f}</span></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='kpi-card'><b>Total Latency</b><br><span style='font-size:1.5rem'>{focus_row['Latency_ms']:.1f} ms</span></div>", unsafe_allow_html=True)

st.markdown("### Scenario Leaderboard")
leaderboard = MOCK_METRICS.copy()
leaderboard["Rank"] = leaderboard["mAP50"].rank(ascending=False, method="min").astype(int)
leaderboard = leaderboard.sort_values("Rank")
st.dataframe(leaderboard[["Rank", "Scenario", "mAP50", "Precision", "Recall", "NIQE", "BRISQUE", "LOE", "Latency_ms", "GFLOPs"]], use_container_width=True, hide_index=True)

st.markdown("### Image Comparison Grid")
col_titles = ["Original", "Enhanced", "Prediction"]
if show_gt:
    col_titles.append("Ground Truth")

cols = st.columns(len(col_titles))
for idx, title in enumerate(col_titles):
    label = f"{title} | {selected_scenario} | {selected_image}"
    cols[idx].image(make_placeholder(label), use_container_width=True)

st.markdown("### Scenario Detail")
chart_col, comp_col = st.columns([1.3, 1])

with chart_col:
    curve_df = pd.DataFrame(
        {
            "Epoch": np.arange(1, 26),
            "mAP50": np.clip(0.28 + 0.018 * np.log1p(np.arange(1, 26)) + 0.06, 0, 1),
            "Precision": np.clip(0.34 + 0.015 * np.log1p(np.arange(1, 26)) + 0.05, 0, 1),
        }
    )
    fig = px.line(curve_df, x="Epoch", y=["mAP50", "Precision"],
                  title=f"Training Trend (Mock) - {selected_scenario}")
    fig.update_layout(legend_title_text="Metric", margin=dict(l=8, r=8, t=44, b=8))
    st.plotly_chart(fig, use_container_width=True)

with comp_col:
    st.markdown("#### Compute Profile")
    st.metric("Enhancer GFLOPs", f"{focus_row['GFLOPs'] * 0.38:.2f}")
    st.metric("YOLO GFLOPs", f"{focus_row['GFLOPs'] * 0.62:.2f}")
    st.metric("Total GFLOPs", f"{focus_row['GFLOPs']:.2f}")
    st.markdown("#### Notes")
    st.write("- Lower is better: NIQE, BRISQUE, LOE")
    st.write("- Higher is better: mAP, Precision, Recall")

st.caption("Prototype only. Backend parser and real artifact binding will be added next.")
