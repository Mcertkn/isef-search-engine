import streamlit as st
import pandas as pd
from src.search import SemanticSearcher


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ISEF Search",
    page_icon="🔬",
    layout="wide"
)


# --------------------------------------------------
# Load search engine
# --------------------------------------------------

@st.cache_resource
def load_searcher():
    return SemanticSearcher(
        model_name="all-MiniLM-L6-v2",
        embeddings_path="models/embeddings.npy",
        data_path="data/processed/search_data.csv"
    )


searcher = load_searcher()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🔬 ISEF Search")

st.markdown(
    """
    **Explore ISEF research projects using semantic search.**

    Search by idea, topic, method, problem, or research area.
    """
)

st.divider()


# --------------------------------------------------
# Search controls
# --------------------------------------------------

query = st.text_input(
    "Search projects",
    placeholder="e.g. machine learning for cancer detection",
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 5])

with col1:
    top_k = st.selectbox(
        "Results",
        options=[5, 10, 15, 20],
        index=1
    )

with col2:
    search_clicked = st.button(
        "🔎 Search",
        use_container_width=True
    )


# --------------------------------------------------
# Search
# --------------------------------------------------

if search_clicked:

    if not query.strip():
        st.warning("Please enter a search query.")

    else:

        results = searcher.search(
            query=query,
            top_k=top_k
        )

        st.subheader(
            f"Results for: `{query}`"
        )

        st.caption(
            f"Showing {len(results)} most semantically similar projects."
        )

        for _, row in results.iterrows():

            with st.container(border=True):

                st.markdown(
                    f"### {row['title']}"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.caption("YEAR")
                    st.write(row["year"])

                with col2:
                    st.caption("CATEGORY")
                    st.write(
                        row["category"]
                        if pd.notna(row["category"])
                        else "Unknown"
                    )

                with col3:
                    st.caption("COUNTRY")
                    st.write(
                        row["country"]
                        if pd.notna(row["country"])
                        else "Unknown"
                    )

                with col4:
                    st.caption("SIMILARITY")
                    st.write(
                        f"{row['similarity']:.3f}"
                    )

                st.progress(
                    min(float(row["similarity"]), 1.0)
                )