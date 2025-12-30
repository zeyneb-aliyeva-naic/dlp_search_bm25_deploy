import streamlit as st
import requests
import json
from typing import Optional

# Configuration
API_BASE_URL = "http://localhost:7860"

st.set_page_config(
    page_title="Hybrid Search & Organization Finder",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Hybrid Search & Organization Finder")
st.markdown("---")

# Sidebar for API health check
with st.sidebar:
    st.header("API Status")
    if st.button("Check API Health"):
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                st.success("✅ API is running")
                
                # Try deep health check
                deep_health = requests.get(f"{API_BASE_URL}/deep-health")
                if deep_health.status_code == 200:
                    health_data = deep_health.json()
                    st.json(health_data)
            else:
                st.error("❌ API is not responding")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure it's running on port 7860")
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Make sure FastAPI is running:
       ```
       docker run -p 7860:7860 fastapi
       ```
    2. Use the tabs above to search
    """)

# Create tabs for different search types
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Hybrid Search AZE","Hybrid Search EN", "Hybrid Search RU","🏢 Organization Search"])

# Tab 1: Hybrid Search AZE
with tab1:
    st.header("Hybrid Search AZE")
    st.markdown("Search using BM25 and vector embeddings")
    
    query = st.text_input("Enter your search query:", key="hybrid_query")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        size = st.number_input("Number of results:", min_value=1, max_value=100, value=10, key="hybrid_size")
    with col2:
        alpha = st.slider("Alpha (BM25 vs Vector weight):", 0.0, 1.0, 0.5, 0.1, key="hybrid_alpha")
    with col3:
        use_vector = st.checkbox("Use vector search", value=True, key="hybrid_vector")
    with col4:
        use_spelling = st.checkbox("Spelling correction", value=False, key="hybrid_spelling")
    
    if st.button("Search", key="hybrid_search_btn"):
        if query:
            search_query = query
            
            # Apply spelling correction if enabled
            if use_spelling:
                with st.spinner("Running spelling correction..."):
                    try:
                        spell_response = requests.post(
                            f"{API_BASE_URL}/spellingcorrection",
                            json={"query": query},
                            headers={"Content-Type": "application/json"}
                        )
                        if spell_response.status_code == 200:
                            spell_data = spell_response.json()
                            corrected = spell_data.get("corrected_query", query)
                            if corrected != query:
                                st.info(f"Using corrected query: `{corrected}`")
                                search_query = corrected
                            else:
                                st.info("Spelling correction did not change the query.")
                        else:
                            st.warning("Spelling correction failed, using original query.")
                    except Exception as e:
                        st.warning(f"Spelling correction error: {e}, using original query.")
            
            with st.spinner("Searching..."):
                try:
                    payload = {
                        "query": search_query,
                        "size": size,
                        "alpha": alpha,
                        "use_vector": use_vector
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/search/az",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Found {data.get('total-hits', 0)} results")
                        
                        # Display results
                        results = data.get('Ranked-objects', [])
                        for result in results:
                            score = result.get('_score')
                            score_str = f"{score:.4f}" if score is not None else "N/A"
                            
                            # Try different possible structures for the data
                            source = result.get('_source', result)
                            
                            # Get name_az_d4 from wherever it exists
                            name_az = source.get('name_az_d4') or result.get('name_az_d4', 'N/A')
                            
                            with st.expander(f"{name_az}"):
                                # DEBUG: Show all keys that contain 'expanded'
                                expanded_keys = [k for k in source.keys() if 'expanded' in k.lower()]
                                if expanded_keys:
                                    st.info(f"Found expanded fields: {expanded_keys}")
                                
                                # Display name_az fields
                                st.markdown(f"**name_az_d1:** {source.get('name_az_d1', 'N/A')}")
                                st.markdown(f"**name_az_d2:** {source.get('name_az_d2', 'N/A')}")
                                st.markdown(f"**name_az_d3:** {source.get('name_az_d3', 'N/A')}")
                                st.markdown(f"**name_az_d4:** {source.get('name_az_d4', 'N/A')}")
                                
                                st.markdown("---")

                                st.markdown(f"**name_az_d1_expanded:** {source.get('name_az_d1_expanded', 'N/A')}")
                                st.markdown(f"**name_az_d2_expanded:** {source.get('name_az_d2_expanded', 'N/A')}")
                                st.markdown(f"**name_az_d3_expanded:** {source.get('name_az_d3_expanded', 'N/A')}")
                                st.markdown(f"**name_az_d4_expanded:** {source.get('name_az_d4_expanded', 'N/A')}")
                                
                                
                                # Display all other fields
                                for key, value in source.items():
                                    if not key.startswith('_') and not key.startswith('name_az'):
                                        st.markdown(f"**{key}:** {value}")
                                
                                st.markdown("---")
                                st.markdown(f"**Score:** {score_str}")
                                
                                # Show full data with toggle
                                if st.checkbox("Show Full JSON", key=f"json_{result.get('_id', hash(str(result)))}"):
                                    st.json(result)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure it's running on port 7860")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a search query")


with tab2:
    st.header("Hybrid Search EN")
    st.markdown("Search using BM25 and vector embeddings")
    
    query = st.text_input("Enter your search query:", key="hybrid_query_en")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        size = st.number_input("Number of results:", min_value=1, max_value=100, value=10, key="hybrid_size_en")
    with col2:
        alpha = st.slider("Alpha (BM25 vs Vector weight):", 0.0, 1.0, 0.5, 0.1, key="hybrid_alpha_en")
    with col3:
        use_vector = st.checkbox("Use vector search", value=True, key="hybrid_vector_en")
    with col4:
        use_spelling_en = st.checkbox("Spelling correction", value=False, key="hybrid_spelling_en")
    
    if st.button("Search", key="hybrid_search_btn_en"):
        if query:
            search_query = query
            
            # Apply spelling correction if enabled
            if use_spelling_en:
                with st.spinner("Running spelling correction..."):
                    try:
                        spell_response = requests.post(
                            f"{API_BASE_URL}/spellingcorrection",
                            json={"query": query},
                            headers={"Content-Type": "application/json"}
                        )
                        if spell_response.status_code == 200:
                            spell_data = spell_response.json()
                            corrected = spell_data.get("corrected_query", query)
                            if corrected != query:
                                st.info(f"Using corrected query: `{corrected}`")
                                search_query = corrected
                            else:
                                st.info("Spelling correction did not change the query.")
                        else:
                            st.warning("Spelling correction failed, using original query.")
                    except Exception as e:
                        st.warning(f"Spelling correction error: {e}, using original query.")
            
            with st.spinner("Searching..."):
                try:
                    payload = {
                        "query": search_query,
                        "size": size,
                        "alpha": alpha,
                        "use_vector": use_vector
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/search/en",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Found {data.get('total-hits', 0)} results")
                        
                        # Display results
                        results = data.get('Ranked-objects', [])
                        for result in results:
                            score = result.get('_score')
                            score_str = f"{score:.4f}" if score is not None else "N/A"
                            
                            # Try different possible structures for the data
                            source = result.get('_source', result)
                            
                            # Get name_en_d4 from wherever it exists
                            name_en = source.get('name_en_d4') or result.get('name_en_d4', 'N/A')
                            
                            with st.expander(f"{name_en}"):
                                # DEBUG: Show all keys that contain 'expanded'
                                expanded_keys = [k for k in source.keys() if 'expanded' in k.lower()]
                                if expanded_keys:
                                    st.info(f"Found expanded fields: {expanded_keys}")
                                
                                # Display name_en fields
                                st.markdown(f"**name_en_d1:** {source.get('name_en_d1', 'N/A')}")
                                st.markdown(f"**name_en_d2:** {source.get('name_en_d2', 'N/A')}")
                                st.markdown(f"**name_en_d3:** {source.get('name_en_d3', 'N/A')}")
                                st.markdown(f"**name_en_d4:** {source.get('name_en_d4', 'N/A')}")
                                
                                st.markdown("---")

                                st.markdown(f"**name_en_d1_expanded:** {source.get('name_en_d1_expanded', 'N/A')}")
                                st.markdown(f"**name_en_d2_expanded:** {source.get('name_en_d2_expanded', 'N/A')}")
                                st.markdown(f"**name_en_d3_expanded:** {source.get('name_en_d3_expanded', 'N/A')}")
                                st.markdown(f"**name_en_d4_expanded:** {source.get('name_en_d4_expanded', 'N/A')}")
                                
                                
                                # Display all other fields
                                for key, value in source.items():
                                    if not key.startswith('_') and not key.startswith('name_en'):
                                        st.markdown(f"**{key}:** {value}")
                                
                                st.markdown("---")
                                st.markdown(f"**Score:** {score_str}")
                                
                                # Show full data with toggle
                                if st.checkbox("Show Full JSON", key=f"json_{result.get('_id', hash(str(result)))}"):
                                    st.json(result)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure it's running on port 7860")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a search query")

with tab3:
    st.header("Hybrid Search RU")
    st.markdown("Search using BM25 and vector embeddings")
    
    query = st.text_input("Enter your search query:", key="hybrid_query_ru")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        size = st.number_input("Number of results:", min_value=1, max_value=100, value=10, key="hybrid_size_ru")
    with col2:
        alpha = st.slider("Alpha (BM25 vs Vector weight):", 0.0, 1.0, 0.5, 0.1, key="hybrid_alpha_ru")
    with col3:
        use_vector = st.checkbox("Use vector search", value=True, key="hybrid_vector_ru")
    with col4:
        use_spelling_ru = st.checkbox("Spelling correction", value=False, key="hybrid_spelling_ru")
    
    if st.button("Search", key="hybrid_search_btn_ru"):
        if query:
            search_query = query
            
            # Apply spelling correction if enabled
            if use_spelling_ru:
                with st.spinner("Running spelling correction..."):
                    try:
                        spell_response = requests.post(
                            f"{API_BASE_URL}/spellingcorrection",
                            json={"query": query},
                            headers={"Content-Type": "application/json"}
                        )
                        if spell_response.status_code == 200:
                            spell_data = spell_response.json()
                            corrected = spell_data.get("corrected_query", query)
                            if corrected != query:
                                st.info(f"Using corrected query: `{corrected}`")
                                search_query = corrected
                            else:
                                st.info("Spelling correction did not change the query.")
                        else:
                            st.warning("Spelling correction failed, using original query.")
                    except Exception as e:
                        st.warning(f"Spelling correction error: {e}, using original query.")
            
            with st.spinner("Searching..."):
                try:
                    payload = {
                        "query": search_query,
                        "size": size,
                        "alpha": alpha,
                        "use_vector": use_vector
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/search/ru",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Found {data.get('total-hits', 0)} results")
                        
                        # Display results
                        results = data.get('Ranked-objects', [])
                        for result in results:
                            score = result.get('_score')
                            score_str = f"{score:.4f}" if score is not None else "N/A"
                            
                            # Try different possible structures for the data
                            source = result.get('_source', result)
                            
                            # Get name_en_d4 from wherever it exists
                            name_en = source.get('name_ru_d4') or result.get('name_ru_d4', 'N/A')
                            
                            with st.expander(f"{name_en}"):
                                # DEBUG: Show all keys that contain 'expanded'
                                expanded_keys = [k for k in source.keys() if 'expanded' in k.lower()]
                                if expanded_keys:
                                    st.info(f"Found expanded fields: {expanded_keys}")
                                
                                # Display name_en fields
                                st.markdown(f"**name_ru_d1:** {source.get('name_ru_d1', 'N/A')}")
                                st.markdown(f"**name_ru_d2:** {source.get('name_ru_d2', 'N/A')}")
                                st.markdown(f"**name_ru_d3:** {source.get('name_ru_d3', 'N/A')}")
                                st.markdown(f"**name_ru_d4:** {source.get('name_ru_d4', 'N/A')}")
                                
                                st.markdown("---")

                                st.markdown(f"**name_ru_d1_expanded:** {source.get('name_ru_d1_expanded', 'N/A')}")
                                st.markdown(f"**name_ru_d2_expanded:** {source.get('name_ru_d2_expanded', 'N/A')}")
                                st.markdown(f"**name_ru_d3_expanded:** {source.get('name_ru_d3_expanded', 'N/A')}")
                                st.markdown(f"**name_ru_d4_expanded:** {source.get('name_ru_d4_expanded', 'N/A')}")
                                
                                
                                # Display all other fields
                                for key, value in source.items():
                                    if not key.startswith('_') and not key.startswith('name_ru'):
                                        st.markdown(f"**{key}:** {value}")
                                
                                st.markdown("---")
                                st.markdown(f"**Score:** {score_str}")
                                
                                # Show full data with toggle
                                if st.checkbox("Show Full JSON", key=f"json_{result.get('_id', hash(str(result)))}"):
                                    st.json(result)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure it's running on port 7860")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a search query")

# Tab 4: Organization Search
with tab4:
    st.header("Organization Search")
    st.markdown("Search for organizations by name or abbreviation")
    
    search_term = st.text_input("Enter organization name or abbreviation:", key="org_query")
    
    col1, col2 = st.columns(2)
    with col1:
        org_size = st.number_input("Number of results:", min_value=1, max_value=100, value=10, key="org_size")
    with col2:
        org_index = st.text_input("Index name:", value="organizations_v3", key="org_index")
    
    if st.button("Search Organizations", key="org_search_btn"):
        if search_term:
            with st.spinner("Searching organizations..."):
                try:
                    payload = {
                        "search_term": search_term,
                        "index": org_index,
                        "size": org_size
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/organizations",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Found {data.get('total-hits', 0)} organizations")
                        
                        # Display results in a more structured way
                        results = data.get('results', [])
                        for idx, org in enumerate(results, 1):
                            with st.expander(f"Organization {idx} - {org.get('name', 'N/A')}"):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.markdown(f"**Name:** {org.get('name', 'N/A')}")
                                    st.markdown(f"**Abbreviation:** {org.get('abbreviation', 'N/A')}")
                                    st.markdown(f"**Score:** {org.get('score', 'N/A')}")
                                with col_b:
                                    st.markdown(f"**ID:** {org.get('id', 'N/A')}")
                                    if org.get('additional_info'):
                                        st.markdown("**Additional Info:**")
                                        st.json(org.get('additional_info'))
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure it's running on port 7860")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a search term")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit 🎈 | Powered by FastAPI ⚡")
