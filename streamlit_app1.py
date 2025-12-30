import os
import streamlit as st
import requests
import json
from typing import List

# =====================================================
# CONFIGURATION
# =====================================================

# Base URL of your FastAPI app 
# You can override it with an env variable when needed
API_BASE_URL = os.getenv(
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8000"
)

SEARCH_URL = f"{API_BASE_URL}/search"
HEALTH_URL = f"{API_BASE_URL}/deep-health"
# NEW: spelling correction endpoint
SPELLING_URL = f"{API_BASE_URL}/spellingcorrection"

st.set_page_config(
    page_title="Hybrid Search Demo",
    page_icon="🔍",
    layout="wide"
)

# =====================================================
# UI HEADER
# =====================================================
st.title("🔍 Hybrid Search Demo (Elasticsearch + SentenceTransformer)")
st.markdown(
    """
    This Streamlit app interacts with your **FastAPI hybrid search API**.  
    It performs BM25 or vector-based hybrid searches over your HS code index.
    """
)

# =====================================================
# INPUT FORM
# =====================================================
with st.form("search_form"):
    query = st.text_input("Enter your search query:", placeholder="e.g. aluminium sheets")
    size = st.slider("Number of results", min_value=1, max_value=50, value=10)
    use_vector = st.checkbox("Use Vector (Hybrid) Search", value=False)
    alpha = st.slider("Vector weight (alpha)", 0.0, 1.0, 0.5, step=0.1)
    highlight_matches = st.checkbox("Highlight matches", value=True)
    # NEW: control whether to run spelling correction or not
    use_spelling_correction = st.checkbox("Use spelling correction", value=False)
    show_debug = st.checkbox("Show debug info", value=False)
    submitted = st.form_submit_button("Run Search")

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def search_api(
    query: str,
    size: int,
    use_vector: bool,
    alpha: float,
    highlight_matches: bool,
    show_debug: bool = False
):
    payload = {
        "query": query,
        "size": size,
        "use_vector": use_vector,
        "alpha": alpha,
        "use_highlight": highlight_matches,
        "filter": {
            "trading_types": [],
            "in_vehicle_ids": [],
            "out_vehicle_ids": []
        }
    }
    
    if show_debug:
        st.write("**Debug: Sending payload to /search:**")
        st.json(payload)

    try:
        response = requests.post(SEARCH_URL, json=payload, timeout=30)
        
        if show_debug:
            st.write(f"**Search response status:** {response.status_code}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The model encoding might be taking too long.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Request failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            st.error(f"**Response status code:** {e.response.status_code}")
            st.error(f"**Response text:** {e.response.text}")
        return None

# NEW: spelling correction helper
def spelling_correction_api(
    query: str,
    show_debug: bool = False
) -> str:
    """
    Send the user query to the spelling correction endpoint.
    Adjust payload / response keys based on your FastAPI implementation.
    """
    payload = {"query": query}

    if show_debug:
        st.write("**Debug: Sending payload to /spellingcorrection:**")
        st.json(payload)

    try:
        response = requests.post(SPELLING_URL, json=payload, timeout=10)

        if show_debug:
            st.write(f"**Spelling correction status:** {response.status_code}")

        response.raise_for_status()
        data = response.json()

        # Adjust this part to match the actual response schema of your API.
        corrected = (
            data.get("corrected_query")
            or data.get("corrected")
            or data.get("query")
        )

        if not corrected:
            # If the API did not return a corrected form, fall back to original
            if show_debug:
                st.write("No corrected query in response, using original.")
            return query

        return corrected
    except requests.exceptions.RequestException as e:
        st.warning(f"Spelling correction failed, using original query. Error: {e}")
        return query

# =====================================================
# RUN SEARCH AND DISPLAY RESULTS
# =====================================================
if submitted:
    if not query.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        # Decide which query to send to /search
        search_query = query

        if use_spelling_correction:
            with st.spinner("Running spelling correction..."):
                corrected_query = spelling_correction_api(query, show_debug=show_debug)
            search_query = corrected_query

            # Show the user what is actually being searched
            if corrected_query != query:
                st.info(f"Using corrected query: `{corrected_query}`")
            else:
                st.info("Spelling correction did not change the query.")

        with st.spinner("Searching..."):
            result = search_api(search_query, size, use_vector, alpha, highlight_matches, show_debug)

        if result:
            st.success(f"✅ Found {result.get('total-hits', 0)} hits")

            hits: List[dict] = result.get("Ranked-objects", [])
            if not hits:
                st.info("No results found.")
            else:
                for i, hit in enumerate(hits, start=1):
                    with st.expander(f"#{i} - {hit.get('name_ru_d4', hit.get('code', 'No Name'))}"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Code:** `{hit.get('code')}`")
                            if hit.get("score"):
                                st.markdown(f"**Score:** {hit.get('score'):.4f}")

                            highlight = hit.get("highlight") if highlight_matches else None

                            def render_field(label: str, field_name: str):
                                if highlight and highlight.get(field_name):
                                    snippet = highlight[field_name][0]
                                    st.markdown(f"**{label}:** {snippet}", unsafe_allow_html=True)
                                elif hit.get(field_name):
                                    st.markdown(f"**{label}:** {hit[field_name]}")

                            render_field("Category (D1)", "name_ru_d1")
                            render_field("Subcategory (D2)", "name_ru_d2")
                            render_field("Sub-subcategory (D3)", "name_ru_d3")
                            render_field("Product (D4)", "name_ru_d4")

                            st.markdown(f"**Full Path:** {hit.get('Path') or '-'}")
                        
                        with col2:
                            tradings = hit.get("tradings", [])
                            if tradings:
                                st.markdown("**Tradings:**")
                                for t in tradings:
                                    trade_type = t.get("tradeType", "N/A")
                                    trade_name = t.get("tradeName", "N/A")
                                    st.write(f"• {trade_name} ({trade_type})")
                                    
                                    if t.get("inVehicleId"):
                                        st.write(f"  In: Vehicle {t.get('inVehicleId')}")
                                    if t.get("outVehicleId"):
                                        st.write(f"  Out: Vehicle {t.get('outVehicleId')}")
                            else:
                                st.markdown("*No trading info*")
        else:
            st.error("❌ No valid response received from the API.")

# =====================================================
# SIDEBAR - API HEALTH CHECK
# =====================================================
with st.sidebar:
    st.header("API Status")
    
    if st.button("Check API Health"):
        try:
            health_response = requests.get(HEALTH_URL, timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                st.success("✅ API is reachable")
                st.json(health_data)
            else:
                st.error(f"⚠️ API returned status: {health_response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Cannot connect to API: {e}")
    
    st.markdown("---")
    st.markdown("### Search Tips")
    st.markdown("""
    - **BM25 only**: Uncheck "Use Vector Search"
    - **Hybrid search**: Check "Use Vector Search" and adjust alpha
    - **Alpha = 0.0**: Pure BM25
    - **Alpha = 1.0**: Pure vector similarity
    - **Alpha = 0.5**: Balanced hybrid
    """)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("© 2025 Hybrid Search Demo - FastAPI + Streamlit + Elasticsearch")
