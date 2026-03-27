import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛒",
    layout="wide"
)

# ─── Load Artifacts (cached — runs only once per session) ─────────────────────
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_dir, 'recommendation_artifacts.pkl'), 'rb') as f:
        arts = pickle.load(f)

    item_user_matrix = load_npz(os.path.join(base_dir, 'train_item_user_matrix.npz'))
    arts['item_similarity_matrix'] = item_user_matrix
    return arts

with st.spinner("Loading models... please wait"):
    artifacts = load_artifacts()

product_id_to_index = artifacts['product_id_to_index']
index_to_product_id = artifacts['index_to_product_id']
item_user_matrix    = artifacts['item_user_matrix']   # shape: (34044 items x 61129 users)
sampled_df          = artifacts['sampled_df']

# ─── Helper Functions ─────────────────────────────────────────────────────────
def get_item_recommendations(target_product_id, k=5, similarity_threshold=0.0):
    """
    Computes cosine similarity for ONE product row against all others.
    Uses ~2MB of RAM instead of 8.64GB (no full matrix precomputation).
    """
    if target_product_id not in product_id_to_index:
        return [], []

    target_idx = product_id_to_index[target_product_id]

    # Extract just the single row for the target product (shape: 1 x n_users)
    target_vector = item_user_matrix[target_idx]

    # Compute similarity of this 1 product vs all 34044 products
    # Result shape: (1, 34044) — lightweight and fast
    similarity_scores = cosine_similarity(target_vector, item_user_matrix).flatten()

    similar_indices = np.argsort(similarity_scores)[::-1]

    recommendations = []
    scores = []
    for idx in similar_indices:
        if idx == target_idx:
            continue
        score = similarity_scores[idx]
        if score >= similarity_threshold:
            recommendations.append(index_to_product_id[idx])
            scores.append(round(float(score), 4))
        if len(recommendations) >= k:
            break
    return recommendations, scores

def get_cluster_info(product_id):
    rows = sampled_df[sampled_df['productid'] == product_id]
    if rows.empty:
        return None, None
    row = rows.iloc[0]
    return int(row['dbscan_cluster_tuned']), row

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.markdown("""
**Project:** P655 — Product Recommendation System

**Technique:** Item-Item Collaborative Filtering
**Clustering:** Tuned DBSCAN (eps=1.0, min_samples=5)
**Dataset:** rating_short.csv
(76K users x 40K products)

**Best Model:** Tuned DBSCAN
(Highest Silhouette Score)
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Avinash** | ExcelR Data Science Programme")
st.sidebar.markdown("---")
st.sidebar.markdown("**Try these sample Product IDs:**")
for pid in ['B0074BW614', 'B00004NKIQ', 'B000QUUFRW', 'B001TH7GUU']:
    st.sidebar.code(pid)

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.title("Product Recommendation System")
st.markdown("Item-Item Collaborative Filtering · DBSCAN Clustering")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Search")
    product_input = st.text_input("Enter Product ID", placeholder="e.g. B0074BW614")
    num_recs  = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)
    threshold = st.select_slider(
        "Similarity Threshold",
        options=[0.0, 0.1, 0.3, 0.5, 0.7],
        value=0.0,
        help="Higher = stricter match. Start with 0.0 if you get no results."
    )
    search_btn = st.button("Get Recommendations", type="primary", use_container_width=True)

with col2:
    if search_btn:
        pid = product_input.strip()

        if not pid:
            st.warning("Please enter a Product ID.")

        elif pid not in product_id_to_index:
            st.error(f"Product ID `{pid}` not found in the dataset.")
            st.info("Try one of the sample IDs from the sidebar.")

        else:
            st.subheader(f"Results for: `{pid}`")

            # Cluster Info
            cluster_label, product_row = get_cluster_info(pid)
            if cluster_label is not None and cluster_label != -1:
                st.success(f"DBSCAN Cluster: **{cluster_label}**")
                st.table(pd.DataFrame({
                    "Metric": ["Avg Product Rating", "Avg User Rating",
                               "Log Product Rating Count", "Log User Rating Count"],
                    "Value":  [round(product_row['product_avg_rating'], 2),
                               round(product_row['user_avg_rating'], 2),
                               round(product_row['product_num_ratings_log'], 2),
                               round(product_row['user_num_ratings_log'], 2)]
                }))
            elif cluster_label == -1:
                st.warning("This product was classified as noise by DBSCAN (not in any cluster).")
            else:
                st.info("Cluster info not available — product not in DBSCAN sampled data.")

            # Recommendations
            st.subheader("Similar Products (Item-Item CF)")
            with st.spinner("Computing recommendations..."):
                recs, scores = get_item_recommendations(pid, k=num_recs, similarity_threshold=threshold)

            if recs:
                st.dataframe(
                    pd.DataFrame({'Rank': range(1, len(recs)+1),
                                  'Recommended Product ID': recs,
                                  'Similarity Score': scores}),
                    use_container_width=True, hide_index=True
                )
                st.caption(f"Showing {len(recs)} recommendation(s) with similarity >= {threshold}")
            else:
                st.warning("No recommendations found. Try lowering the Similarity Threshold to 0.0.")

    else:
        st.info("Enter a Product ID on the left and click Get Recommendations.")
        st.markdown("""
        **How it works:**
        - Enter any Product ID from the dataset
        - The app finds similar products using cosine similarity between item rating vectors
        - It also shows which DBSCAN cluster the product belongs to
        - Adjust the threshold slider to control how strict the similarity match is
        """)
