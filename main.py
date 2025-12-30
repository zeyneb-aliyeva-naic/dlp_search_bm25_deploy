import logging
from fastapi import FastAPI, HTTPException
from elasticsearch import AsyncElasticsearch

from models import SearchRequest, HybridRetrievedResponseSet, SpellingRequest, SpellingResponse,HybridRetrievedResponseSetEng, HybridRetrievedResponseSetRu,SearchRequestRu, OrganizationSearchRequest, OrganizationSearchResponse, OrganizationDocument, SearchRequestEng, SpellingRequestRu
from services import SearchService, SearchServiceRu, SearchServiceEn
from config import ES_URL, ES_API_KEY, ES_INDEX, MODEL_DIR, SPELLING_MODEL_DIR, ORGANIZATIONS_INDEX, SOURCE_ES_URL, SOURCE_ES_API_KEY,ES_INDEX_EN, ES_INDEX_RU,load_model, setup_logging
from use_model import load_model as load_spelling_model, predict as spelling_predict
from query_builder import build_organization_query

import os

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Hybrid Search API with Vector Search",
    version="2.0.0",
    description="FastAPI service for hybrid search with Elasticsearch and vector embeddings"
)


@app.on_event("startup")
async def startup():
    """Initialize Elasticsearch connection and load model on startup"""
    app.state.es = AsyncElasticsearch(hosts=[ES_URL], api_key=ES_API_KEY)
    logger.info(f"✅ Connected to Elasticsearch at {ES_URL}")

    app.state.es_source = AsyncElasticsearch(hosts=[SOURCE_ES_URL], api_key=SOURCE_ES_API_KEY)
    print(f"✅ Connected to Source Elasticsearch at {SOURCE_ES_URL}")
    
    # Load the embedding model
    app.state.model = load_model()
    
    # Initialize search service
    app.state.search_service = SearchService(
        es_client=app.state.es,
        model=app.state.model,
        index=ES_INDEX
    )
    app.state.search_service_ru = SearchServiceRu(
        es_client=app.state.es,
        model=app.state.model,
        index=ES_INDEX_RU
    )
    app.state.search_service_en = SearchServiceEn(
        es_client=app.state.es,
        model=app.state.model,
        index=ES_INDEX_EN
    )

    # Load spelling correction model
    try:
        spelling_model_path = SPELLING_MODEL_DIR
        app.state.spell_model, app.state.spell_tokenizer, app.state.spell_config = load_spelling_model(
            spelling_model_path,
            device="cpu"
        )
        logger.info("✅ Spelling correction model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load spelling model: {e}")
        app.state.spell_model = None
        app.state.spell_tokenizer = None

@app.on_event("shutdown")
async def shutdown():
    """Close Elasticsearch connection on shutdown"""
    await app.state.es.close()
    logger.info("Elasticsearch connection closed")


@app.post(
    "/search/az",
    response_model=HybridRetrievedResponseSet,
    response_model_by_alias=True,
    summary="Hybrid Search with Vector Embeddings",
    description="Perform hybrid search combining BM25 and vector similarity"
)
async def search(req: SearchRequest) -> HybridRetrievedResponseSet:
    """
    Execute a hybrid search query against Elasticsearch.
    
    Args:
        req: SearchRequest containing query text, filters, size/top_k, alpha, and use_vector flag
        
    Returns:
        HybridRetrievedResponseSet with ranked search results
        
    Raises:
        HTTPException: If Elasticsearch query fails
    """
    try:
        ranked, total_count = await app.state.search_service.search(req)
        
        return HybridRetrievedResponseSet(
            **{
                "query-text": req.query, 
                "total-hits": total_count,
                "Ranked-objects": ranked
            }
        )
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Search error: {e}"
        )
    
@app.post(
    "/search/en",
    response_model=HybridRetrievedResponseSetEng,
    response_model_by_alias=True,
    summary="Hybrid Search with Vector Embeddings",
    description="Perform hybrid search combining BM25 and vector similarity"
)
async def search(req: SearchRequestEng) -> HybridRetrievedResponseSetEng:
    """
    Execute a hybrid search query against Elasticsearch.
    
    Args:
        req: SearchRequest containing query text, filters, size/top_k, alpha, and use_vector flag
        
    Returns:
        HybridRetrievedResponseSet with ranked search results
        
    Raises:
        HTTPException: If Elasticsearch query fails
    """
    try:
        ranked, total_count = await app.state.search_service.search(req)
        
        return HybridRetrievedResponseSetEng(
            **{
                "query-text": req.query, 
                "total-hits": total_count,
                "Ranked-objects": ranked
            }
        )
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Search error: {e}"
        )
    
@app.post(
    "/search/ru",
    response_model=HybridRetrievedResponseSetRu,
    response_model_by_alias=True,
    summary="Hybrid Search with Vector Embeddings",
    description="Perform hybrid search combining BM25 and vector similarity"
)
async def search(req: SearchRequestRu) -> HybridRetrievedResponseSetRu:
    """
    Execute a hybrid search query against Elasticsearch.
    
    Args:
        req: SearchRequest containing query text, filters, size/top_k, alpha, and use_vector flag
        
    Returns:
        HybridRetrievedResponseSet with ranked search results
        
    Raises:
        HTTPException: If Elasticsearch query fails
    """
    try:
        ranked, total_count = await app.state.search_service_ru.search(req)
        
        return HybridRetrievedResponseSetRu(
            **{
                "query-text": req.query, 
                "total-hits": total_count,
                "Ranked-objects": ranked
            }
        )
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Search error: {e}"
        )


@app.post(
    "/organizations",
    response_model=OrganizationSearchResponse,
    response_model_by_alias=True,
    summary="Search Organizations",
    description="Search for organizations by name or abbreviation"
)
async def search_organizations(req: OrganizationSearchRequest) -> OrganizationSearchResponse:
    """
    Search for organizations in Elasticsearch.
    
    Args:
        req: OrganizationSearchRequest containing search term, index, and size
        
    Returns:
        OrganizationSearchResponse with ranked search results
    """
    # Build Elasticsearch query
    es_query = build_organization_query(req.search_term)
    
    print(f"Organization search query for '{req.search_term}':", es_query)
    
    # Execute search against source Elasticsearch
    try:
        resp = await app.state.es_source.search(
            index=req.index,
            query=es_query,
            size=req.size
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Elasticsearch error: {e}"
        )
    
    # Parse response
    hits_data = resp.get("hits", {})
    hits = hits_data.get("hits", []) or []
    total_hits = hits_data.get("total", {})
    
    # Extract total count
    if isinstance(total_hits, dict):
        total_count = total_hits.get("value", 0)
    else:
        total_count = total_hits or 0
    
    # Convert hits to OrganizationDocument objects
    results = [OrganizationDocument.from_es_hit(hit) for hit in hits]
    
    # Return structured response
    return OrganizationSearchResponse(
        **{
            "search-term": req.search_term,
            "total-hits": total_count,
            "results": results
        }
    )


@app.get("/deep-health", summary="Health Check")
async def health_check():
    """Check if the service, Elasticsearch connection, and model are healthy"""
    try:
        es_health = await app.state.es.info()
        model_status = "loaded" if app.state.model is not None else "not loaded"
        
        return {
            "status": "healthy",
            "elasticsearch": {
                "status": "connected",
                "cluster_name": es_health.get("cluster_name"),
                "url": ES_URL,
                "index": ES_INDEX
            },
            "model": {
                "status": model_status,
                "path": MODEL_DIR
            },
            "spelling_model": "loaded" if app.state.spell_model else "not loaded",
            "vector_search": "enabled" if app.state.model else "disabled"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "model": "unknown"
        }


@app.get("/health")
async def health():
    """Basic health check endpoint"""
    return {"status": "ok"}


