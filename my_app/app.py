import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from io import BytesIO

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import skew

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RFM Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background-color: #0d0f14; color: #e8e6e0; }

[data-testid="stSidebar"] {
    background-color: #13161e;
    border-right: 1px solid #2a2d38;
}

.hero {
    background: linear-gradient(135deg, #1a1d28 0%, #0d0f14 60%);
    border: 1px solid #2a2d38;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,180,50,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 { font-size: 2.4rem; font-weight: 800; color: #f5c842; margin: 0 0 0.4rem 0; letter-spacing: -0.5px; }
.hero p  { color: #8a8fa0; font-size: 1rem; margin: 0; font-family: 'DM Mono', monospace; }

.metric-card {
    background: #13161e;
    border: 1px solid #2a2d38;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
}
.metric-card .label { font-size: 0.72rem; font-family: 'DM Mono', monospace; color: #6b7080; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: #f5c842; }
.metric-card .sub   { font-size: 0.75rem; color: #5a5f70; font-family: 'DM Mono', monospace; }

.section-header { display: flex; align-items: center; gap: 0.8rem; margin: 2rem 0 1rem 0; padding-bottom: 0.6rem; border-bottom: 1px solid #2a2d38; }
.section-header h2 { font-size: 1.15rem; font-weight: 700; color: #e8e6e0; margin: 0; }
.section-badge { background: #f5c84220; color: #f5c842; border: 1px solid #f5c84240; border-radius: 6px; padding: 2px 10px; font-size: 0.7rem; font-family: 'DM Mono', monospace; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }

.info-box { background: #13161e; border-left: 3px solid #f5c842; border-radius: 0 8px 8px 0; padding: 0.8rem 1.2rem; font-size: 0.85rem; color: #9a9fae; font-family: 'DM Mono', monospace; margin: 0.8rem 0 1.2rem 0; }

.stButton > button { background: #f5c842 !important; color: #0d0f14 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; padding: 0.5rem 2rem !important; font-size: 0.95rem !important; }
.stButton > button:hover { background: #ffd85a !important; box-shadow: 0 4px 20px rgba(245,200,66,0.3) !important; }

.stDownloadButton > button { background: transparent !important; color: #f5c842 !important; border: 1px solid #f5c84250 !important; font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; border-radius: 8px !important; }
.stDownloadButton > button:hover { border-color: #f5c842 !important; background: #f5c84210 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Matplotlib dark theme
# ─────────────────────────────────────────────
DARK_BG  = "#0d0f14"
CARD_BG  = "#13161e"
ACCENT   = "#f5c842"
TEXT_CLR = "#e8e6e0"
MUTED    = "#6b7080"
GRID_CLR = "#2a2d38"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   CARD_BG,
    "axes.edgecolor":   GRID_CLR,
    "axes.labelcolor":  TEXT_CLR,
    "axes.titlecolor":  TEXT_CLR,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT_CLR,
    "grid.color":       GRID_CLR,
    "grid.linewidth":   0.6,
    "axes.grid":        True,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.titlepad":    10,
    "figure.dpi":       110,
    "font.family":      "monospace",
})

CLUSTER_PALETTE = ["#f5c842", "#4ecdc4", "#ff6b6b", "#a29bfe",
                   "#fd79a8", "#00b894", "#e17055", "#74b9ff"]

# ─────────────────────────────────────────────
#  Exact column names from online_retail_II.xlsx
# ─────────────────────────────────────────────
COL_CUSTOMER = "Customer ID"   # float column, space in name
COL_INVOICE  = "Invoice"       # object  (was "InvoiceNo" in old dataset)
COL_DATE     = "InvoiceDate"   # datetime
COL_PRICE    = "Price"         # float   (was "UnitPrice" in old dataset)
COL_QUANTITY = "Quantity"      # int
COL_COUNTRY  = "Country"       # object

# ─────────────────────────────────────────────
#  Core functions
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes, filename):
    """Load XLS/XLSX and clean using the exact schema of online_retail_II.xlsx."""

    # Load file
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
    else:
        df = pd.read_csv(BytesIO(file_bytes), encoding="ISO-8859-1")

    raw_rows = len(df)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Validate required columns
    required = [COL_CUSTOMER, COL_INVOICE, COL_DATE, COL_PRICE, COL_QUANTITY]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns: {missing_cols}\n"
            f"Columns found: {list(df.columns)}\n\n"
            "Expected schema: Invoice, StockCode, Description, Quantity, "
            "InvoiceDate, Price, Customer ID, Country\n"
            "Please upload 'online_retail_II.xlsx' from Kaggle."
        )

    # Drop rows where Customer ID is missing
    df.dropna(subset=[COL_CUSTOMER], inplace=True)

    # Remove cancellations / returns
    df = df[df[COL_QUANTITY] > 0]
    df = df[df[COL_PRICE]    > 0]

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Parse InvoiceDate
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])

    # TotalPrice = Quantity × Price
    df["TotalPrice"] = df[COL_QUANTITY] * df[COL_PRICE]

    # Customer ID → string for groupby
    df[COL_CUSTOMER] = df[COL_CUSTOMER].astype(str).str.strip()

    return df, raw_rows


@st.cache_data(show_spinner=False)
def build_rfm(_df):
    """Compute Recency / Frequency / Monetary per customer."""
    snapshot_date = _df[COL_DATE].max() + pd.Timedelta(days=1)

    rfm = _df.groupby(COL_CUSTOMER).agg(
        Recency   = (COL_DATE,      lambda x: (snapshot_date - x.max()).days),
        Frequency = (COL_INVOICE,   "nunique"),
        Monetary  = ("TotalPrice",  "sum")
    ).reset_index()

    return rfm, snapshot_date


def scale_rfm(rfm):
    """Log1p-transform skewed features then StandardScale."""
    rfm_log = rfm[["Recency", "Frequency", "Monetary"]].copy()
    for col in ["Recency", "Frequency", "Monetary"]:
        if abs(skew(rfm_log[col])) > 1:
            rfm_log[col] = np.log1p(rfm_log[col])
    scaled = StandardScaler().fit_transform(rfm_log)
    return scaled


def run_kmeans(scaled, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    return labels, silhouette_score(scaled, labels), km.inertia_


def run_hierarchical(scaled, k):
    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(scaled)
    return labels, silhouette_score(scaled, labels)


def run_dbscan(scaled, eps, min_samples):
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
    unique = set(labels)
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    mask = labels != -1
    sil = (silhouette_score(scaled[mask], labels[mask])
           if mask.sum() > 1 and n_clusters > 1 else float("nan"))
    return labels, sil, n_clusters, int((labels == -1).sum())


def pca_2d(scaled):
    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(scaled)
    return comps, pca.explained_variance_ratio_


def tsne_2d(scaled, max_samples=2000):
    idx = np.random.choice(len(scaled), size=min(max_samples, len(scaled)), replace=False)
    comps = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=1000).fit_transform(scaled[idx])
    return comps, idx


def scatter_2d(x, y, labels, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(DARK_BG)
    for i, lbl in enumerate(sorted(set(labels))):
        mask  = labels == lbl
        color = "#555566" if lbl == -1 else CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        name  = "Noise"         if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(x[mask], y[mask], c=color, label=name,
                   alpha=0.55, s=18, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT_CLR,
              facecolor=CARD_BG, edgecolor=GRID_CLR)
    fig.tight_layout()
    return fig


def boxplot_clusters(rfm, cluster_col):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor(DARK_BG)
    for ax, col in zip(axes, ["Recency", "Frequency", "Monetary"]):
        for i, grp in rfm.groupby(cluster_col)[col]:
            ax.boxplot(grp, positions=[i], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)], alpha=0.7),
                       medianprops=dict(color="white", linewidth=2),
                       whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                       flierprops=dict(marker=".", color=MUTED, markersize=3))
        ax.set_title(f"{col} per Cluster")
        ax.set_xlabel("Cluster")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem 0;'>
        <div style='font-size:1.4rem;font-weight:800;color:#f5c842;letter-spacing:-0.5px;'>⬡ RFM Studio</div>
        <div style='font-size:0.72rem;font-family:"DM Mono",monospace;color:#6b7080;margin-top:2px;'>Customer Segmentation Engine</div>
    </div>
    <hr style='border-color:#2a2d38;margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload Dataset",
        type=["xlsx", "xls", "csv"],
        help="Upload online_retail_II.xlsx from Kaggle"
    )

    st.markdown("<div style='margin-top:1.2rem;font-size:0.8rem;font-weight:700;color:#9a9fae;text-transform:uppercase;letter-spacing:1px;'>K-Means</div>", unsafe_allow_html=True)
    k_value = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=4)

    st.markdown("<div style='margin-top:1rem;font-size:0.8rem;font-weight:700;color:#9a9fae;text-transform:uppercase;letter-spacing:1px;'>DBSCAN</div>", unsafe_allow_html=True)
    dbscan_eps     = st.slider("Epsilon (eps)",  min_value=0.1, max_value=2.0, value=0.5, step=0.05)
    dbscan_minsamp = st.slider("Min Samples",    min_value=2,   max_value=20,  value=5)

    st.markdown("<div style='margin-top:1rem;font-size:0.8rem;font-weight:700;color:#9a9fae;text-transform:uppercase;letter-spacing:1px;'>t-SNE</div>", unsafe_allow_html=True)
    tsne_max = st.slider("Max Samples", min_value=500, max_value=3000, value=2000, step=500)

    st.markdown("<hr style='border-color:#2a2d38;margin:1.2rem 0 0.6rem 0;'>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Analysis", use_container_width=True)

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛒 RFM Customer Segmentation</h1>
    <p>Upload online_retail_II.xlsx → Clean → RFM → K-Means · Hierarchical · DBSCAN → PCA · t-SNE</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  GATES
# ─────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class="info-box">
        ① Upload <strong>online_retail_II.xlsx</strong> from Kaggle in the sidebar, then click <strong>▶ Run Analysis</strong>.<br>
        Dataset: <a href="https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset"
                   target="_blank" style="color:#f5c842;">kaggle.com → Online Retail Dataset</a>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not run_btn:
    st.markdown("""
    <div class="info-box">✅ File uploaded. Adjust parameters in the sidebar, then click <strong>▶ Run Analysis</strong>.</div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  STEP 1 — LOAD & CLEAN
# ─────────────────────────────────────────────
with st.spinner("Loading and cleaning data…"):
    try:
        file_bytes = uploaded.read()
        df, raw_rows = load_and_clean(file_bytes, uploaded.name)
    except ValueError as e:
        st.error(f"❌ Column mismatch:\n\n{e}")
        st.stop()

st.markdown("""
<div class="section-header">
    <span class="section-badge">Step 1</span>
    <h2>Data Overview</h2>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col_w, label, value, sub in zip(
    [c1, c2, c3, c4],
    ["Raw Rows", "Clean Rows", "Customers", "Countries"],
    [f"{raw_rows:,}", f"{len(df):,}",
     f"{df[COL_CUSTOMER].nunique():,}",
     f"{df[COL_COUNTRY].nunique() if COL_COUNTRY in df.columns else '—'}"],
    ["before cleaning", "after cleaning", "unique IDs", "in dataset"]
):
    with col_w:
        st.markdown(f"""<div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

with st.expander("Preview cleaned data (first 50 rows)"):
    st.dataframe(df.head(50), use_container_width=True)

# ─────────────────────────────────────────────
#  STEP 2 — RFM
# ─────────────────────────────────────────────
with st.spinner("Computing RFM features…"):
    rfm, snapshot_date = build_rfm(df)
    rfm_scaled = scale_rfm(rfm)

st.markdown("""
<div class="section-header">
    <span class="section-badge">Step 2</span>
    <h2>RFM Feature Engineering</h2>
</div>
""", unsafe_allow_html=True)

rc1, rc2, rc3 = st.columns(3)
for w, col_name, emoji in zip([rc1, rc2, rc3],
                               ["Recency", "Frequency", "Monetary"],
                               ["⏱", "🔁", "💰"]):
    with w:
        st.markdown(f"""<div class="metric-card">
            <div class="label">{emoji} {col_name}</div>
            <div class="value" style="font-size:1.3rem;">{rfm[col_name].median():.1f}</div>
            <div class="sub">median · skew {skew(rfm[col_name]):.2f}</div>
        </div>""", unsafe_allow_html=True)

fig_rfm, axes = plt.subplots(1, 3, figsize=(14, 4))
fig_rfm.patch.set_facecolor(DARK_BG)
for ax, col, color in zip(axes,
                           ["Recency", "Frequency", "Monetary"],
                           [ACCENT, "#4ecdc4", "#ff6b6b"]):
    sns.histplot(rfm[col], kde=True, ax=ax, color=color, alpha=0.75, edgecolor="none")
    ax.set_title(f"{col} Distribution")
    ax.set_xlabel(col, fontsize=9)
fig_rfm.tight_layout()
st.pyplot(fig_rfm, use_container_width=True)

# ─────────────────────────────────────────────
#  STEP 3 — CLUSTERING
# ─────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-badge">Step 3</span>
    <h2>Clustering</h2>
</div>
""", unsafe_allow_html=True)

tab_km, tab_hier, tab_db = st.tabs(["🔵  K-Means", "🌳  Hierarchical", "🔴  DBSCAN"])

# ── K-MEANS ──────────────────────────────────
with tab_km:
    with st.spinner("Running K-Means…"):
        inertia_vals, sil_vals = [], []
        for k in range(2, 11):
            _, s, inr = run_kmeans(rfm_scaled, k)
            inertia_vals.append(inr)
            sil_vals.append(s)
        km_labels, km_sil, _ = run_kmeans(rfm_scaled, k_value)
        rfm["KMeans_Cluster"] = km_labels

    col_e, col_s = st.columns(2)
    with col_e:
        fig_e, ax = plt.subplots(figsize=(6, 4))
        fig_e.patch.set_facecolor(DARK_BG)
        ax.plot(range(2, 11), inertia_vals, marker="o", color=ACCENT, linewidth=2.5, markersize=7)
        ax.axvline(k_value, color="#ff6b6b", linestyle="--", linewidth=1.5, label=f"K={k_value}")
        ax.set_title("Elbow Method (WCSS)")
        ax.set_xlabel("K"); ax.set_ylabel("Inertia"); ax.legend(fontsize=9)
        fig_e.tight_layout(); st.pyplot(fig_e, use_container_width=True)

    with col_s:
        fig_s, ax = plt.subplots(figsize=(6, 4))
        fig_s.patch.set_facecolor(DARK_BG)
        ax.plot(range(2, 11), sil_vals, marker="s", color="#4ecdc4", linewidth=2.5, markersize=7)
        ax.axvline(k_value, color="#ff6b6b", linestyle="--", linewidth=1.5, label=f"K={k_value}")
        ax.set_title("Silhouette Score vs K")
        ax.set_xlabel("K"); ax.set_ylabel("Silhouette Score"); ax.legend(fontsize=9)
        fig_s.tight_layout(); st.pyplot(fig_s, use_container_width=True)

    st.markdown(f"""<div class="info-box">K-Means  K={k_value}  →  Silhouette Score: <strong style="color:#f5c842;">{km_sil:.4f}</strong></div>""", unsafe_allow_html=True)
    st.pyplot(boxplot_clusters(rfm, "KMeans_Cluster"), use_container_width=True)

# ── HIERARCHICAL ─────────────────────────────
with tab_hier:
    with st.spinner("Running Hierarchical Clustering…"):
        hier_labels, hier_sil = run_hierarchical(rfm_scaled, k_value)
        rfm["Hierarchical_Cluster"] = hier_labels

    sample_idx_d = np.random.choice(len(rfm_scaled), size=min(300, len(rfm_scaled)), replace=False)
    linked = linkage(rfm_scaled[sample_idx_d], method="ward")

    fig_dend, ax = plt.subplots(figsize=(13, 5))
    fig_dend.patch.set_facecolor(DARK_BG)
    dendrogram(linked, truncate_mode="lastp", p=20,
               leaf_rotation=90, leaf_font_size=9, show_contracted=True,
               color_threshold=0.7 * max(linked[:, 2]), ax=ax)
    ax.set_title("Dendrogram — Ward Linkage (300 samples)")
    ax.set_xlabel("Customer Samples"); ax.set_ylabel("Distance")
    fig_dend.tight_layout()
    st.pyplot(fig_dend, use_container_width=True)

    st.markdown(f"""<div class="info-box">Hierarchical (Ward)  K={k_value}  →  Silhouette Score: <strong style="color:#f5c842;">{hier_sil:.4f}</strong></div>""", unsafe_allow_html=True)
    st.pyplot(boxplot_clusters(rfm, "Hierarchical_Cluster"), use_container_width=True)

# ── DBSCAN ───────────────────────────────────
with tab_db:
    with st.spinner("Computing k-NN distances…"):
        nn = NearestNeighbors(n_neighbors=5).fit(rfm_scaled)
        distances, _ = nn.kneighbors(rfm_scaled)
        distances_sorted = np.sort(distances[:, 4])

    fig_knn, ax = plt.subplots(figsize=(9, 4))
    fig_knn.patch.set_facecolor(DARK_BG)
    ax.plot(distances_sorted, color="#fd79a8", linewidth=1.8)
    ax.axhline(dbscan_eps, color=ACCENT, linestyle="--", linewidth=1.5, label=f"eps = {dbscan_eps}")
    ax.set_title("k-NN Distance Plot — eps Estimation")
    ax.set_xlabel("Points (sorted)"); ax.set_ylabel("5th Nearest Neighbour Distance"); ax.legend(fontsize=9)
    fig_knn.tight_layout()
    st.pyplot(fig_knn, use_container_width=True)

    db_labels, db_sil, n_db_clusters, n_noise = run_dbscan(rfm_scaled, dbscan_eps, dbscan_minsamp)
    rfm["DBSCAN_Cluster"] = db_labels

    d1, d2, d3 = st.columns(3)
    with d1: st.markdown(f"""<div class="metric-card"><div class="label">Clusters Found</div><div class="value">{n_db_clusters}</div></div>""", unsafe_allow_html=True)
    with d2: st.markdown(f"""<div class="metric-card"><div class="label">Noise Points</div><div class="value">{n_noise}</div></div>""", unsafe_allow_html=True)
    with d3: st.markdown(f"""<div class="metric-card"><div class="label">Silhouette Score</div><div class="value" style="font-size:1.3rem;">{db_sil:.4f}</div></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STEP 4 — PCA
# ─────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-badge">Step 4</span>
    <h2>PCA — 2D Projection</h2>
</div>
""", unsafe_allow_html=True)

pca_comps, var_ratio = pca_2d(rfm_scaled)
st.markdown(f"""<div class="info-box">Variance explained — PC1: <strong style="color:#f5c842;">{var_ratio[0]*100:.1f}%</strong>  ·  PC2: <strong style="color:#f5c842;">{var_ratio[1]*100:.1f}%</strong>  ·  Total: <strong style="color:#f5c842;">{sum(var_ratio)*100:.1f}%</strong></div>""", unsafe_allow_html=True)

p1, p2, p3 = st.columns(3)
for w, lbl_col, title in zip([p1, p2, p3],
                               ["KMeans_Cluster", "Hierarchical_Cluster", "DBSCAN_Cluster"],
                               ["K-Means", "Hierarchical", "DBSCAN"]):
    with w:
        st.pyplot(scatter_2d(pca_comps[:, 0], pca_comps[:, 1],
                             rfm[lbl_col].values, f"PCA — {title}", "PC1", "PC2"),
                  use_container_width=True)

# ─────────────────────────────────────────────
#  STEP 5 — t-SNE
# ─────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-badge">Step 5</span>
    <h2>t-SNE — Non-Linear Projection</h2>
</div>
""", unsafe_allow_html=True)

with st.spinner(f"Running t-SNE on {min(tsne_max, len(rfm_scaled))} samples (~30s)…"):
    tsne_comps, tsne_idx = tsne_2d(rfm_scaled, max_samples=tsne_max)

rfm_tsne_sub = rfm.iloc[tsne_idx]

t1, t2, t3 = st.columns(3)
for w, lbl_col, title in zip([t1, t2, t3],
                               ["KMeans_Cluster", "Hierarchical_Cluster", "DBSCAN_Cluster"],
                               ["K-Means", "Hierarchical", "DBSCAN"]):
    with w:
        st.pyplot(scatter_2d(tsne_comps[:, 0], tsne_comps[:, 1],
                             rfm_tsne_sub[lbl_col].values, f"t-SNE — {title}", "Dim 1", "Dim 2"),
                  use_container_width=True)

# ─────────────────────────────────────────────
#  STEP 6 — EVALUATION SUMMARY
# ─────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-badge">Step 6</span>
    <h2>Evaluation & Summary</h2>
</div>
""", unsafe_allow_html=True)

summary_df = pd.DataFrame({
    "Algorithm":         ["K-Means", "Hierarchical (Ward)", "DBSCAN"],
    "Clusters":          [rfm["KMeans_Cluster"].nunique(),
                          rfm["Hierarchical_Cluster"].nunique(),
                          n_db_clusters],
    "Noise Points":      [0, 0, n_noise],
    "Silhouette Score":  [round(km_sil, 4), round(hier_sil, 4), round(db_sil, 4)],
    "Requires K?":       ["Yes", "Yes", "No"],
    "Handles Outliers?": ["No",  "No",  "Yes"],
})
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# RFM heatmap
st.markdown("**K-Means Cluster RFM Profile Heatmap**")
km_profile = rfm.groupby("KMeans_Cluster")[["Recency", "Frequency", "Monetary"]].mean()
km_norm = pd.DataFrame(
    MinMaxScaler().fit_transform(km_profile),
    index=km_profile.index, columns=km_profile.columns
)
fig_heat, ax = plt.subplots(figsize=(8, max(3, k_value * 0.7)))
fig_heat.patch.set_facecolor(DARK_BG)
sns.heatmap(km_norm, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "Normalised Value"}, annot_kws={"size": 10})
ax.set_title("Normalised RFM Profile per K-Means Cluster")
ax.set_xlabel("RFM Feature"); ax.set_ylabel("Cluster")
fig_heat.tight_layout()
st.pyplot(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────
#  STEP 7 — DOWNLOAD
# ─────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-badge">Step 7</span>
    <h2>Download Results</h2>
</div>
""", unsafe_allow_html=True)

csv_buf = BytesIO()
rfm.to_csv(csv_buf, index=False)
csv_buf.seek(0)

st.download_button(
    label="⬇  Download RFM + Cluster Labels (CSV)",
    data=csv_buf,
    file_name="rfm_clusters.csv",
    mime="text/csv",
)

with st.expander("Preview output data"):
    st.dataframe(
        rfm[[COL_CUSTOMER, "Recency", "Frequency", "Monetary",
             "KMeans_Cluster", "Hierarchical_Cluster", "DBSCAN_Cluster"]].head(50),
        use_container_width=True
    )

st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;font-family:"DM Mono",monospace;
            font-size:0.72rem;color:#3a3d4a;'>
    RFM Customer Segmentation Studio · Built with Streamlit
</div>
""", unsafe_allow_html=True)