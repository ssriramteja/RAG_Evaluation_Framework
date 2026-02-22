import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

st.set_page_config(
    page_title = "RAG Evaluation Dashboard",
    page_icon  = "📊",
    layout     = "wide"
)

st.title("RAG Evaluation Dashboard")
st.caption("Clinical RAG Performance Analysis | ClinicalBERT + FAISS + Groq Llama 3.3")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    csv_path  = RESULTS_DIR / "eval_results.csv"
    json_path = RESULTS_DIR / "eval_report.json"
    df, report = None, None
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    if json_path.exists():
        with open(json_path) as f:
            report = json.load(f)
    return df, report

df, report = load_results()

if df is None:
    st.warning("No evaluation results found.")
    st.info("To populate this dashboard, execute the evaluation pipeline:\n```bash\npython src/rag_runner.py\npython src/evaluator.py\npython src/report.py\n```")
    st.stop()

# ── RAGAS metric columns ──────────────────────────────────────────────────────
ragas_cols = [
    c for c in [
        "faithfulness", "answer_relevancy",
        "context_recall", "context_precision", "answer_correctness"
    ] if c in df.columns
]
custom_cols = [
    c for c in [
        "token_overlap_vs_gt", "fuzzy_similarity_vs_gt",
        "context_hit_rate", "answer_length_score", "grounding_proxy"
    ] if c in df.columns
]

# ── Top KPI metrics ───────────────────────────────────────────────────────────
st.markdown("## Aggregate Performance")
kpi_cols = st.columns(len(ragas_cols))
for col, metric in zip(kpi_cols, ragas_cols):
    score = df[metric].mean()
    delta_color = "normal" if score >= 0.7 else "inverse"
    col.metric(
        label      = metric.replace("_", " ").title(),
        value      = f"{score:.3f}",
        delta      = "Target Achieved" if score >= 0.7 else "Below Threshold",
        delta_color= delta_color
    )

st.divider()

# ── Row 1: Bar chart + Radar chart ────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Average RAGAS Scores")
    avg_scores = {m: round(df[m].mean(), 4) for m in ragas_cols}
    bar_fig = px.bar(
        x      = list(avg_scores.keys()),
        y      = list(avg_scores.values()),
        labels = {"x": "Metric", "y": "Score"},
        color  = list(avg_scores.values()),
        color_continuous_scale = "Blues",
        range_y= [0, 1]
    )
    bar_fig.update_layout(
        showlegend    = False,
        coloraxis_showscale = False,
        plot_bgcolor  = "white",
        height        = 350
    )
    bar_fig.add_hline(
        y=0.7, line_dash="dash", line_color="green",
        annotation_text="Target: 0.7"
    )
    st.plotly_chart(bar_fig, use_container_width=True)

with col2:
    st.markdown("### Pipeline Characterization")
    all_metrics = ragas_cols + custom_cols[:3]
    values      = [df[m].mean() for m in all_metrics]

    radar_fig = go.Figure(go.Scatterpolar(
        r    = values + [values[0]],
        theta= all_metrics + [all_metrics[0]],
        fill = "toself",
        line = dict(color="#1f77b4", width=2)
    ))
    radar_fig.update_layout(
        polar  = dict(radialaxis=dict(visible=True, range=[0, 1])),
        height = 350
    )
    st.plotly_chart(radar_fig, use_container_width=True)

st.divider()

# ── Row 2: Per-question heatmap ───────────────────────────────────────────────
st.markdown("### Individual Query Heatmap")

heatmap_df = df[["question"] + ragas_cols].copy()
heatmap_df["question"] = heatmap_df["question"].str[:50] + "..."
heatmap_df = heatmap_df.set_index("question")

heatmap_fig = px.imshow(
    heatmap_df.T,
    color_continuous_scale = "RdYlGn",
    zmin  = 0, zmax = 1,
    aspect= "auto",
    labels= dict(color="Score")
)
heatmap_fig.update_layout(height=350)
st.plotly_chart(heatmap_fig, use_container_width=True)

st.divider()

# ── Row 3: Per-question drill-down ────────────────────────────────────────────
st.markdown("### Case Investigation")

selected_q = st.selectbox(
    "Select query for deep-dive:",
    options=df["question"].tolist()
)

row = df[df["question"] == selected_q].iloc[0]

dcol1, dcol2 = st.columns(2)
with dcol1:
    st.markdown("**User Query:**")
    st.info(row["question"])

    st.markdown("**Model Response:**")
    st.success(row.get("answer", "N/A"))

with dcol2:
    st.markdown("**Expected Ground Truth:**")
    st.warning(row.get("ground_truth", "N/A"))

    st.markdown("**Performance Scores:**")
    for m in ragas_cols:
        score = row[m]
        status_point = "●" if score >= 0.7 else "○"
        st.markdown(f"{status_point} **{m}**: `{score:.4f}`")

st.divider()

# ── Row 4: Custom metrics ─────────────────────────────────────────────────────
st.markdown("### Lexical Proxy Metrics")
cust_avg = {m: round(df[m].mean(), 4) for m in custom_cols}

c_cols = st.columns(len(custom_cols))
for col, (metric, score) in zip(c_cols, cust_avg.items()):
    col.metric(
        label = metric.replace("_", " ").title(),
        value = f"{score:.3f}"
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Engine: RAGAS Framework | Models: Groq Llama 3.3-70b (Judge), ClinicalBERT (Embeddings)"
)
