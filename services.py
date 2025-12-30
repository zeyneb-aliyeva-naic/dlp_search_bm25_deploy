import logging
import asyncio
from typing import List, Tuple
from elasticsearch import AsyncElasticsearch
from sentence_transformers import SentenceTransformer

from models import SearchRequest, ElasticDocument, SearchRequestEng, SearchRequestRu, ElasticDocumentEng, ElasticDocumentRu
from query_builder import (
    build_es_bool_query, 
    build_hybrid_vector_query, 
    build_highlight_config,
    build_es_bool_query_ru,
    build_hybrid_vector_query_ru,
    build_highlight_config_ru,
    build_es_bool_query_en,
    build_hybrid_vector_query_en,
    build_highlight_config_en
)

logger = logging.getLogger(__name__)


class SearchService:
    """Service for executing hybrid search queries"""
    
    def __init__(self, es_client: AsyncElasticsearch, model: SentenceTransformer, index: str):
        self.es = es_client
        self.model = model
        self.index = index
    
    async def search(self, req: SearchRequest) -> Tuple[List[ElasticDocument], int]:
        """
        Execute hybrid or BM25-only search based on request parameters.
        
        Args:
            req: SearchRequest with query parameters
            
        Returns:
            Tuple of (ranked documents, total hit count)
        """
        logger.info(f"Search request: query='{req.query}', size={req.size}, "
                    f"use_vector={req.use_vector}, alpha={req.alpha}")
        
        use_vector_search = req.use_vector and self.model is not None
        
        if req.use_vector and self.model is None:
            logger.warning("Vector search requested but model is not loaded. Falling back to BM25.")
        
        if use_vector_search:
            return await self._hybrid_search(req)
        else:
            return await self._bm25_search(req)
    
    async def _hybrid_search(self, req: SearchRequest) -> Tuple[List[ElasticDocument], int]:
        """Execute hybrid search combining BM25 and vector similarity"""
        logger.info("Using HYBRID search mode (BM25 + Vector)")
        
        # Generate query embedding
        loop = asyncio.get_event_loop()
        query_vector = await loop.run_in_executor(
            None, 
            lambda: self.model.encode(req.query).tolist()
        )
        
        logger.info(f"Generated query embedding with {len(query_vector)} dimensions")
        
        # Build and execute hybrid query
        es_query = build_hybrid_vector_query(req, query_vector)
        hits, total_hits = await self._execute_search(req, es_query, "hybrid")
        
        return hits, total_hits
    
    async def _bm25_search(self, req: SearchRequest) -> Tuple[List[ElasticDocument], int]:
        """Execute BM25-only search"""
        logger.info("Using BM25-only search mode")
        
        # Build and execute BM25 query
        es_query = build_es_bool_query(req)
        hits, total_hits = await self._execute_search(req, es_query, "BM25")
        
        return hits, total_hits
    
    async def _execute_search(
        self, 
        req: SearchRequest, 
        es_query: dict, 
        mode: str
    ) -> Tuple[List[ElasticDocument], int]:
        """
        Execute Elasticsearch search and parse results.
        
        Args:
            req: SearchRequest with parameters
            es_query: Elasticsearch query dictionary
            mode: Search mode description for logging
            
        Returns:
            Tuple of (document list, total count)
        """
        highlight_conf = build_highlight_config() if req.use_highlight else None
        
        search_kwargs = {
            "index": self.index,
            "query": es_query,
            "size": req.size,
        }
        
        if highlight_conf:
            search_kwargs["highlight"] = highlight_conf
            logger.info("Highlighting enabled")
        
        logger.info(f"Executing {mode} search on index: {self.index}")
        resp = await self.es.search(**search_kwargs)
        logger.info(f"{mode} search completed successfully")
        
        # Parse response
        hits_data = (resp or {}).get("hits", {})
        hits = hits_data.get("hits", []) or []
        total_hits = hits_data.get("total", {})
        
        # Extract total count
        if isinstance(total_hits, dict):
            total_count = total_hits.get("value", 0)
        else:
            total_count = total_hits or 0
        
        logger.info(f"Retrieved {len(hits)} hits from {mode} search")
        logger.info(f"Total hits available: {total_count}")
        
        # Convert to documents
        ranked = [ElasticDocument.from_es_hit(h) for h in hits]
        
        # Log top results
        self._log_top_results(ranked)
        
        return ranked, total_count
    
    def _log_top_results(self, ranked: List[ElasticDocument]):
        """Log information about top search results"""
        if ranked:
            logger.info(f"Top result: code={ranked[0].code}, score={ranked[0].score:.4f}")
            if len(ranked) > 1:
                logger.info(f"2nd result: code={ranked[1].code}, score={ranked[1].score:.4f}")
            if len(ranked) > 2:
                logger.info(f"3rd result: code={ranked[2].code}, score={ranked[2].score:.4f}")


#RU
class SearchServiceRu:
    """Service for executing hybrid search queries"""
    
    def __init__(self, es_client: AsyncElasticsearch, model: SentenceTransformer, index: str):
        self.es = es_client
        self.model = model
        self.index = index
    
    async def search(self, req: SearchRequestRu) -> Tuple[List[ElasticDocumentRu], int]:
        """
        Execute hybrid or BM25-only search based on request parameters.
        
        Args:
            req: SearchRequest with query parameters
            
        Returns:
            Tuple of (ranked documents, total hit count)
        """
        logger.info(f"Search request: query='{req.query}', size={req.size}, "
                    f"use_vector={req.use_vector}, alpha={req.alpha}")
        
        use_vector_search = req.use_vector and self.model is not None
        
        if req.use_vector and self.model is None:
            logger.warning("Vector search requested but model is not loaded. Falling back to BM25.")
        
        if use_vector_search:
            return await self._hybrid_search(req)
        else:
            return await self._bm25_search(req)
    
    async def _hybrid_search(self, req: SearchRequestRu) -> Tuple[List[ElasticDocumentRu], int]:
        """Execute hybrid search combining BM25 and vector similarity"""
        logger.info("Using HYBRID search mode (BM25 + Vector)")
        
        # Generate query embedding
        loop = asyncio.get_event_loop()
        query_vector = await loop.run_in_executor(
            None, 
            lambda: self.model.encode(req.query).tolist()
        )
        
        logger.info(f"Generated query embedding with {len(query_vector)} dimensions")
        
        # Build and execute hybrid query
        es_query = build_hybrid_vector_query_ru(req, query_vector)
        hits, total_hits = await self._execute_search(req, es_query, "hybrid")
        
        return hits, total_hits
    
    async def _bm25_search(self, req: SearchRequestRu) -> Tuple[List[ElasticDocumentRu], int]:
        """Execute BM25-only search"""
        logger.info("Using BM25-only search mode")
        
        # Build and execute BM25 query
        es_query = build_es_bool_query_ru(req)
        hits, total_hits = await self._execute_search(req, es_query, "BM25")
        
        return hits, total_hits
    
    async def _execute_search(
        self, 
        req: SearchRequestRu, 
        es_query: dict, 
        mode: str
    ) -> Tuple[List[ElasticDocumentRu], int]:
        """
        Execute Elasticsearch search and parse results.
        
        Args:
            req: SearchRequest with parameters
            es_query: Elasticsearch query dictionary
            mode: Search mode description for logging
            
        Returns:
            Tuple of (document list, total count)
        """
        highlight_conf = build_highlight_config_ru() if req.use_highlight else None
        
        search_kwargs = {
            "index": self.index,
            "query": es_query,
            "size": req.size,
        }
        
        if highlight_conf:
            search_kwargs["highlight"] = highlight_conf
            logger.info("Highlighting enabled")
        
        logger.info(f"Executing {mode} search on index: {self.index}")
        resp = await self.es.search(**search_kwargs)
        logger.info(f"{mode} search completed successfully")
        
        # Parse response
        hits_data = (resp or {}).get("hits", {})
        hits = hits_data.get("hits", []) or []
        total_hits = hits_data.get("total", {})
        
        # Extract total count
        if isinstance(total_hits, dict):
            total_count = total_hits.get("value", 0)
        else:
            total_count = total_hits or 0
        
        logger.info(f"Retrieved {len(hits)} hits from {mode} search")
        logger.info(f"Total hits available: {total_count}")
        
        # Convert to documents
        ranked = [ElasticDocumentRu.from_es_hit(h) for h in hits]
        
        # Log top results
        self._log_top_results(ranked)
        
        return ranked, total_count
    
    def _log_top_results(self, ranked: List[ElasticDocumentRu]):
        """Log information about top search results"""
        if ranked:
            logger.info(f"Top result: code={ranked[0].code}, score={ranked[0].score:.4f}")
            if len(ranked) > 1:
                logger.info(f"2nd result: code={ranked[1].code}, score={ranked[1].score:.4f}")
            if len(ranked) > 2:
                logger.info(f"3rd result: code={ranked[2].code}, score={ranked[2].score:.4f}")



#EN
class SearchServiceEn:
    """Service for executing hybrid search queries"""
    
    def __init__(self, es_client: AsyncElasticsearch, model: SentenceTransformer, index: str):
        self.es = es_client
        self.model = model
        self.index = index
    
    async def search(self, req: SearchRequestEng) -> Tuple[List[ElasticDocumentEng], int]:
        """
        Execute hybrid or BM25-only search based on request parameters.
        
        Args:
            req: SearchRequest with query parameters
            
        Returns:
            Tuple of (ranked documents, total hit count)
        """
        logger.info(f"Search request: query='{req.query}', size={req.size}, "
                    f"use_vector={req.use_vector}, alpha={req.alpha}")
        
        use_vector_search = req.use_vector and self.model is not None
        
        if req.use_vector and self.model is None:
            logger.warning("Vector search requested but model is not loaded. Falling back to BM25.")
        
        if use_vector_search:
            return await self._hybrid_search(req)
        else:
            return await self._bm25_search(req)
    
    async def _hybrid_search(self, req: SearchRequestEng) -> Tuple[List[ElasticDocumentEng], int]:
        """Execute hybrid search combining BM25 and vector similarity"""
        logger.info("Using HYBRID search mode (BM25 + Vector)")
        
        # Generate query embedding
        loop = asyncio.get_event_loop()
        query_vector = await loop.run_in_executor(
            None, 
            lambda: self.model.encode(req.query).tolist()
        )
        
        logger.info(f"Generated query embedding with {len(query_vector)} dimensions")
        
        # Build and execute hybrid query
        es_query = build_hybrid_vector_query_en(req, query_vector)
        hits, total_hits = await self._execute_search(req, es_query, "hybrid")
        
        return hits, total_hits
    
    async def _bm25_search(self, req: SearchRequestEng) -> Tuple[List[ElasticDocumentEng], int]:
        """Execute BM25-only search"""
        logger.info("Using BM25-only search mode")
        
        # Build and execute BM25 query
        es_query = build_es_bool_query_en(req)
        hits, total_hits = await self._execute_search(req, es_query, "BM25")
        
        return hits, total_hits
    
    async def _execute_search(
        self, 
        req: SearchRequestEng, 
        es_query: dict, 
        mode: str
    ) -> Tuple[List[ElasticDocumentEng], int]:
        """
        Execute Elasticsearch search and parse results.
        
        Args:
            req: SearchRequest with parameters
            es_query: Elasticsearch query dictionary
            mode: Search mode description for logging
            
        Returns:
            Tuple of (document list, total count)
        """
        highlight_conf = build_highlight_config_en() if req.use_highlight else None
        
        search_kwargs = {
            "index": self.index,
            "query": es_query,
            "size": req.size,
        }
        
        if highlight_conf:
            search_kwargs["highlight"] = highlight_conf
            logger.info("Highlighting enabled")
        
        logger.info(f"Executing {mode} search on index: {self.index}")
        resp = await self.es.search(**search_kwargs)
        logger.info(f"{mode} search completed successfully")
        
        # Parse response
        hits_data = (resp or {}).get("hits", {})
        hits = hits_data.get("hits", []) or []
        total_hits = hits_data.get("total", {})
        
        # Extract total count
        if isinstance(total_hits, dict):
            total_count = total_hits.get("value", 0)
        else:
            total_count = total_hits or 0
        
        logger.info(f"Retrieved {len(hits)} hits from {mode} search")
        logger.info(f"Total hits available: {total_count}")
        
        # Convert to documents
        ranked = [ElasticDocumentEng.from_es_hit(h) for h in hits]
        
        # Log top results
        self._log_top_results(ranked)
        
        return ranked, total_count
    
    def _log_top_results(self, ranked: List[ElasticDocumentEng]):
        """Log information about top search results"""
        if ranked:
            logger.info(f"Top result: code={ranked[0].code}, score={ranked[0].score:.4f}")
            if len(ranked) > 1:
                logger.info(f"2nd result: code={ranked[1].code}, score={ranked[1].score:.4f}")
            if len(ranked) > 2:
                logger.info(f"3rd result: code={ranked[2].code}, score={ranked[2].score:.4f}")