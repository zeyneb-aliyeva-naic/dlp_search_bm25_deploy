import re
import logging
from typing import List
from models import SearchRequest, SearchRequestEng, SearchRequestRu

logger = logging.getLogger(__name__)


def build_es_bool_query(req: SearchRequest) -> dict:
    """
    Build Elasticsearch bool query with BM25 scoring.
    
    Args:
        req: SearchRequest containing query text and filters
        
    Returns:
        Elasticsearch bool query dictionary
    """
    f = req.filter
    query_text = req.query.strip()

    numeric_code = re.sub(r"\D", "", query_text)
    has_code = len(numeric_code) >= 6
    is_only_code = query_text.isdigit() and len(query_text) in [6, 8, 10]

    should_clauses = []

    if is_only_code:
        # Code-only query: use prefix matching with boosting
        should_clauses.extend(_build_code_clauses(query_text))
    else:
        # Text query: use multi_match and combined_fields
        should_clauses.extend(_build_text_clauses(query_text))
        
        # Add embedded code logic if query contains numbers
        if has_code:
            should_clauses.extend(_build_code_clauses(numeric_code))

    bool_query = {
        "should": should_clauses,
        "minimum_should_match": 1,
    }

    # Add filters if specified
    filters = _build_filters(f)
    if filters:
        bool_query["filter"] = filters

    query = {"bool": bool_query}
    logger.info(f"Constructed ES bool query: {query}")

    return query


def _build_code_clauses(code_text: str) -> List[dict]:
    """Build prefix matching clauses for product codes"""
    clauses = []
    code_len = len(code_text)

    if code_len >= 6:
        clauses.append({
            "prefix": {
                "code": {"value": code_text[:6], "boost": 10.0}
            }
        })
    if code_len >= 8:
        clauses.append({
            "prefix": {
                "code": {"value": code_text[:8], "boost": 20.0}
            }
        })
    if code_len >= 10:
        clauses.append({
            "prefix": {
                "code": {"value": code_text[:10], "boost": 30.0}
            }
        })
    
    return clauses


def _build_text_clauses(query_text: str) -> List[dict]:
    """Build text search clauses using multi_match and combined_fields"""
    return [
        {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "name_az_d4_expanded^3.5",
                    "name_az_d3_expanded^2.5",
                    "name_az_d2_expanded^1.5",
                    "name_az_d1_expanded^0.5",
                ],
                "fuzziness": "1",
            }
        },
        {
            "combined_fields": {
                "query": query_text,
                "fields": [
                    "name_az_d4_expanded^1",
                    "name_az_d3_expanded^2",
                    "name_az_d2_expanded^3",
                    "name_az_d1_expanded^4",
                ],
                "operator": "or",
                "boost": 5,
            }
        }
    ]


def _build_filters(filter_obj) -> List[dict]:
    """Build filter clauses for tradings"""
    filters = []
    
    if filter_obj.trade_type or filter_obj.in_vehicle_ids or filter_obj.out_vehicle_ids:
        nested_must = []

        if filter_obj.trade_type:
            nested_must.append({
                "term": {"tradings.tradeType": filter_obj.trade_type.value}
            })

        if filter_obj.in_vehicle_ids:
            nested_must.append({
                "terms": {
                    "tradings.inVehicleId": [v.to_id() for v in filter_obj.in_vehicle_ids]
                }
            })

        if filter_obj.out_vehicle_ids:
            nested_must.append({
                "terms": {
                    "tradings.outVehicleId": [v.to_id() for v in filter_obj.out_vehicle_ids]
                }
            })

        if nested_must:
            filters.append({
                "nested": {
                    "path": "tradings",
                    "query": {
                        "bool": {
                            "must": nested_must
                        }
                    }
                }
            })

    return filters


def build_hybrid_vector_query(req: SearchRequest, query_vector: List[float]) -> dict:
    """
    Build hybrid query combining BM25 and vector similarity with filters.
    
    Args:
        req: SearchRequest containing search parameters
        query_vector: Query embedding vector
        
    Returns:
        Elasticsearch script_score query dictionary
    """
    logger.info(f"Building hybrid vector query with alpha={req.alpha}")
    
    # Build the base BM25 query
    base_query = build_es_bool_query(req)
    bool_query = base_query["bool"]

    # Parent weights for hierarchical embeddings
    parent_coefficient = 0.3
    parent_fields = {
        "embedding_d1": 1.0 * parent_coefficient,
        "embedding_d2": 1.5 * parent_coefficient,
        "embedding_d3": 2.0 * parent_coefficient,
        "embedding_d4": 1.0  # main leaf, full weight
    }
    
    logger.info(f"Embedding weights: d1={parent_fields['embedding_d1']}, "
                f"d2={parent_fields['embedding_d2']}, d3={parent_fields['embedding_d3']}, "
                f"d4={parent_fields['embedding_d4']}")
    
    # Build script_score query for hybrid search with null checks
    query = {
        "script_score": {
            "query": {"bool": bool_query},
            "script": {
                "source": """
                double cosine = 0.0;
                
                // Level embeddings with null/empty checks
                if (doc.containsKey('embedding_d1') && doc['embedding_d1'].size() > 0) {
                    cosine += params.alpha_level1 * (cosineSimilarity(params.query_vector, 'embedding_d1') + 1.0)/2.0;
                }
                if (doc.containsKey('embedding_d2') && doc['embedding_d2'].size() > 0) {
                    cosine += params.alpha_level2 * (cosineSimilarity(params.query_vector, 'embedding_d2') + 1.0)/2.0;
                }
                if (doc.containsKey('embedding_d3') && doc['embedding_d3'].size() > 0) {
                    cosine += params.alpha_level3 * (cosineSimilarity(params.query_vector, 'embedding_d3') + 1.0)/2.0;
                }
                if (doc.containsKey('embedding_d4') && doc['embedding_d4'].size() > 0) {
                    cosine += params.alpha_level4 * (cosineSimilarity(params.query_vector, 'embedding_d4') + 1.0)/2.0;
                }
                
                // BM25 score normalization
                double bm25 = _score / (_score + 10.0);
                
                // Hybrid score calculation
                return params.alpha * cosine + (1 - params.alpha) * bm25;
                """,
                "params": {
                    "query_vector": query_vector,
                    "alpha": req.alpha,
                    "alpha_level1": parent_fields["embedding_d1"],
                    "alpha_level2": parent_fields["embedding_d2"],
                    "alpha_level3": parent_fields["embedding_d3"],
                    "alpha_level4": parent_fields["embedding_d4"],
                }
            }
        }
    }
    
    logger.info(f"Built script_score query with vector dimension: {len(query_vector)}")
    
    return query


def build_highlight_config() -> dict:
    """Build Elasticsearch highlight configuration"""
    return {
        "pre_tags": ["<mark>"],
        "post_tags": ["</mark>"],
        "fields": {
            "name_az_d1": {},
            "name_az_d2": {},
            "name_az_d3": {},
            "name_az_d4": {},
        },
    }
import re
import logging
from typing import List
from models import SearchRequest

logger = logging.getLogger(__name__)


def build_es_bool_query_ru(req: SearchRequestRu) -> dict:
    """
    Build Elasticsearch bool query with BM25 scoring.
    
    Args:
        req: SearchRequest containing query text and filters
        
    Returns:
        Elasticsearch bool query dictionary
    """
    f = req.filter
    query_text = req.query.strip()

    numeric_code = re.sub(r"\D", "", query_text)
    has_code = len(numeric_code) >= 6
    is_only_code = query_text.isdigit() and len(query_text) in [6, 8, 10]

    should_clauses = []

    if is_only_code:
        # Code-only query: use prefix matching with boosting
        should_clauses.extend(_build_code_clauses(query_text))
    else:
        # Text query: use multi_match and combined_fields
        should_clauses.extend(_build_text_clauses(query_text))
        
        # Add embedded code logic if query contains numbers
        if has_code:
            should_clauses.extend(_build_code_clauses(numeric_code))

    bool_query = {
        "should": should_clauses,
        "minimum_should_match": 1,
    }

    # Add filters if specified
    filters = _build_filters(f)
    if filters:
        bool_query["filter"] = filters

    query = {"bool": bool_query}
    logger.info(f"Constructed ES bool query: {query}")

    return query


def _build_code_clauses_ru(code_text: str) -> List[dict]:
    """Build prefix matching clauses for product codes"""
    clauses = []
    code_len = len(code_text)

    if code_len >= 6:
        clauses.append({
            "prefix": {
                "code": {"value": code_text[:6], "boost": 10.0}
            }
        })
    if code_len >= 8:
        clauses.append({
            "prefix": {
                "code": {"value": code_text[:8], "boost": 20.0}
            }
        })
    if code_len >= 10:
        clauses.append({
            "prefix": {
                "code": {"value": code_text[:10], "boost": 30.0}
            }
        })
    
    return clauses


def _build_text_clauses_ru(query_text: str) -> List[dict]:
    """Build text search clauses using multi_match and combined_fields"""
    return [
        {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "name_ru_d4_expanded^3.5",
                    "name_ru_d3_expanded^2.5",
                    "name_ru_d2_expanded^1.5",
                    "name_ru_d1_expanded^0.5",
                ],
                "fuzziness": "1",
            }
        },
        {
            "combined_fields": {
                "query": query_text,
                "fields": [
                    "name_ru_d4_expanded^1",
                    "name_ru_d3_expanded^2",
                    "name_ru_d2_expanded^3",
                    "name_ru_d1_expanded^4",
                ],
                "operator": "or",
                "boost": 5,
            }
        }
    ]


def _build_filters_ru(filter_obj) -> List[dict]:
    """Build filter clauses for tradings"""
    filters = []
    
    if filter_obj.trade_type or filter_obj.in_vehicle_ids or filter_obj.out_vehicle_ids:
        nested_must = []

        if filter_obj.trade_type:
            nested_must.append({
                "term": {"tradings.tradeType": filter_obj.trade_type.value}
            })

        if filter_obj.in_vehicle_ids:
            nested_must.append({
                "terms": {
                    "tradings.inVehicleId": [v.to_id() for v in filter_obj.in_vehicle_ids]
                }
            })

        if filter_obj.out_vehicle_ids:
            nested_must.append({
                "terms": {
                    "tradings.outVehicleId": [v.to_id() for v in filter_obj.out_vehicle_ids]
                }
            })

        if nested_must:
            filters.append({
                "nested": {
                    "path": "tradings",
                    "query": {
                        "bool": {
                            "must": nested_must
                        }
                    }
                }
            })

    return filters


def build_hybrid_vector_query_ru(req: SearchRequestRu, query_vector: List[float]) -> dict:
    """
    Build hybrid query combining BM25 and vector similarity with filters.
    
    Args:
        req: SearchRequest containing search parameters
        query_vector: Query embedding vector
        
    Returns:
        Elasticsearch script_score query dictionary
    """
    logger.info(f"Building hybrid vector query with alpha={req.alpha}")
    
    # Build the base BM25 query
    base_query = build_es_bool_query(req)
    bool_query = base_query["bool"]

    # Parent weights for hierarchical embeddings
    parent_coefficient = 0.3
    parent_fields = {
        "embedding_d1": 1.0 * parent_coefficient,
        "embedding_d2": 1.5 * parent_coefficient,
        "embedding_d3": 2.0 * parent_coefficient,
        "embedding_d4": 1.0  # main leaf, full weight
    }
    
    logger.info(f"Embedding weights: d1={parent_fields['embedding_d1']}, "
                f"d2={parent_fields['embedding_d2']}, d3={parent_fields['embedding_d3']}, "
                f"d4={parent_fields['embedding_d4']}")
    
    # Build script_score query for hybrid search with null checks
    query = {
        "script_score": {
            "query": {"bool": bool_query},
            "script": {
                "source": """
                double cosine = 0.0;
                
                // Level embeddings with null/empty checks
                if (doc.containsKey('embedding_d1') && doc['embedding_d1'].size() > 0) {
                    cosine += params.alpha_level1 * (cosineSimilarity(params.query_vector, 'embedding_d1') + 1.0)/2.0;
                }
                if (doc.containsKey('embedding_d2') && doc['embedding_d2'].size() > 0) {
                    cosine += params.alpha_level2 * (cosineSimilarity(params.query_vector, 'embedding_d2') + 1.0)/2.0;
                }
                if (doc.containsKey('embedding_d3') && doc['embedding_d3'].size() > 0) {
                    cosine += params.alpha_level3 * (cosineSimilarity(params.query_vector, 'embedding_d3') + 1.0)/2.0;
                }
                if (doc.containsKey('embedding_d4') && doc['embedding_d4'].size() > 0) {
                    cosine += params.alpha_level4 * (cosineSimilarity(params.query_vector, 'embedding_d4') + 1.0)/2.0;
                }
                
                // BM25 score normalization
                double bm25 = _score / (_score + 10.0);
                
                // Hybrid score calculation
                return params.alpha * cosine + (1 - params.alpha) * bm25;
                """,
                "params": {
                    "query_vector": query_vector,
                    "alpha": req.alpha,
                    "alpha_level1": parent_fields["embedding_d1"],
                    "alpha_level2": parent_fields["embedding_d2"],
                    "alpha_level3": parent_fields["embedding_d3"],
                    "alpha_level4": parent_fields["embedding_d4"],
                }
            }
        }
    }
    
    logger.info(f"Built script_score query with vector dimension: {len(query_vector)}")
    logger.info(f"Hybrid query: {query}")
    
    return query


def build_highlight_config_ru() -> dict:
    """Build Elasticsearch highlight configuration"""
    return {
        "pre_tags": ["<mark>"],
        "post_tags": ["</mark>"],
        "fields": {
            "name_ru_d1": {},
            "name_ru_d2": {},
            "name_ru_d3": {},
            "name_ru_d4": {},
        },
    }


#En
def build_es_bool_query_en(req: SearchRequestEng) -> dict:
    """
    Build Elasticsearch bool query with BM25 scoring.
    
    Args:
        req: SearchRequest containing query text and filters
        
    Returns:
        Elasticsearch bool query dictionary
    """
    f = req.filter
    query_text = req.query.strip()

    numeric_code = re.sub(r"\D", "", query_text)
    has_code = len(numeric_code) >= 6
    is_only_code = query_text.isdigit() and len(query_text) in [6, 8, 10]

    should_clauses = []

    if is_only_code:
        # Code-only query: use prefix matching with boosting
        should_clauses.extend(_build_code_clauses(query_text))
    else:
        # Text query: use multi_match and combined_fields
        should_clauses.extend(_build_text_clauses(query_text))
        
        # Add embedded code logic if query contains numbers
        if has_code:
            should_clauses.extend(_build_code_clauses(numeric_code))

    bool_query = {
        "should": should_clauses,
        "minimum_should_match": 1,
    }

    # Add filters if specified
    filters = _build_filters(f)
    if filters:
        bool_query["filter"] = filters

    query = {"bool": bool_query}
    logger.info(f"Constructed ES bool query: {query}")

    return query


def _build_code_clauses_en(code_text: str) -> List[dict]:
    """Build prefix matching clauses for product codes"""
    clauses = []
    code_len = len(code_text)

    if code_len >= 6:
        clauses.append({
            "prefix": {
                "code": {"value": code_text[:6], "boost": 10.0}
            }
        })
    if code_len >= 8:
        clauses.append({
            "prefix": {
                "code": {"value": code_text[:8], "boost": 20.0}
            }
        })
    if code_len >= 10:
        clauses.append({
            "prefix": {
                "code": {"value": code_text[:10], "boost": 30.0}
            }
        })
    
    return clauses


def _build_text_clauses_en(query_text: str) -> List[dict]:
    """Build text search clauses using multi_match and combined_fields"""
    return [
        {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "name_en_d4_expanded^3.5",
                    "name_en_d3_expanded^2.5",
                    "name_en_d2_expanded^1.5",
                    "name_en_d1_expanded^0.5",
                ],
                "fuzziness": "1",
            }
        },
        {
            "combined_fields": {
                "query": query_text,
                "fields": [
                    "name_ru_d4_expanded^1",
                    "name_ru_d3_expanded^2",
                    "name_ru_d2_expanded^3",
                    "name_ru_d1_expanded^4",
                ],
                "operator": "or",
                "boost": 5,
            }
        }
    ]


def _build_filters_en(filter_obj) -> List[dict]:
    """Build filter clauses for tradings"""
    filters = []
    
    if filter_obj.trade_type or filter_obj.in_vehicle_ids or filter_obj.out_vehicle_ids:
        nested_must = []

        if filter_obj.trade_type:
            nested_must.append({
                "term": {"tradings.tradeType": filter_obj.trade_type.value}
            })

        if filter_obj.in_vehicle_ids:
            nested_must.append({
                "terms": {
                    "tradings.inVehicleId": [v.to_id() for v in filter_obj.in_vehicle_ids]
                }
            })

        if filter_obj.out_vehicle_ids:
            nested_must.append({
                "terms": {
                    "tradings.outVehicleId": [v.to_id() for v in filter_obj.out_vehicle_ids]
                }
            })

        if nested_must:
            filters.append({
                "nested": {
                    "path": "tradings",
                    "query": {
                        "bool": {
                            "must": nested_must
                        }
                    }
                }
            })

    return filters


def build_hybrid_vector_query_en(req: SearchRequestEng, query_vector: List[float]) -> dict:
    """
    Build hybrid query combining BM25 and vector similarity with filters.
    
    Args:
        req: SearchRequest containing search parameters
        query_vector: Query embedding vector
        
    Returns:
        Elasticsearch script_score query dictionary
    """
    logger.info(f"Building hybrid vector query with alpha={req.alpha}")
    
    # Build the base BM25 query
    base_query = build_es_bool_query(req)
    bool_query = base_query["bool"]

    # Parent weights for hierarchical embeddings
    parent_coefficient = 0.3
    parent_fields = {
        "embedding_d1": 1.0 * parent_coefficient,
        "embedding_d2": 1.5 * parent_coefficient,
        "embedding_d3": 2.0 * parent_coefficient,
        "embedding_d4": 1.0  # main leaf, full weight
    }
    
    logger.info(f"Embedding weights: d1={parent_fields['embedding_d1']}, "
                f"d2={parent_fields['embedding_d2']}, d3={parent_fields['embedding_d3']}, "
                f"d4={parent_fields['embedding_d4']}")
    
    # Build script_score query for hybrid search with null checks
    query = {
        "script_score": {
            "query": {"bool": bool_query},
            "script": {
                "source": """
                double cosine = 0.0;
                
                // Level embeddings with null/empty checks
                if (doc.containsKey('embedding_d1') && doc['embedding_d1'].size() > 0) {
                    cosine += params.alpha_level1 * (cosineSimilarity(params.query_vector, 'embedding_d1') + 1.0)/2.0;
                }
                if (doc.containsKey('embedding_d2') && doc['embedding_d2'].size() > 0) {
                    cosine += params.alpha_level2 * (cosineSimilarity(params.query_vector, 'embedding_d2') + 1.0)/2.0;
                }
                if (doc.containsKey('embedding_d3') && doc['embedding_d3'].size() > 0) {
                    cosine += params.alpha_level3 * (cosineSimilarity(params.query_vector, 'embedding_d3') + 1.0)/2.0;
                }
                if (doc.containsKey('embedding_d4') && doc['embedding_d4'].size() > 0) {
                    cosine += params.alpha_level4 * (cosineSimilarity(params.query_vector, 'embedding_d4') + 1.0)/2.0;
                }
                
                // BM25 score normalization
                double bm25 = _score / (_score + 10.0);
                
                // Hybrid score calculation
                return params.alpha * cosine + (1 - params.alpha) * bm25;
                """,
                "params": {
                    "query_vector": query_vector,
                    "alpha": req.alpha,
                    "alpha_level1": parent_fields["embedding_d1"],
                    "alpha_level2": parent_fields["embedding_d2"],
                    "alpha_level3": parent_fields["embedding_d3"],
                    "alpha_level4": parent_fields["embedding_d4"],
                }
            }
        }
    }
    
    logger.info(f"Built script_score query with vector dimension: {len(query_vector)}")
    logger.info(f"Hybrid query: {query}")
    
    return query


def build_highlight_config_en() -> dict:
    """Build Elasticsearch highlight configuration"""
    return {
        "pre_tags": ["<mark>"],
        "post_tags": ["</mark>"],
        "fields": {
            "name_ru_d1": {},
            "name_ru_d2": {},
            "name_ru_d3": {},
            "name_ru_d4": {},
        },
    }


# Organization abbreviation mapping
ORGANIZATION_ABBREVIATIONS = {
    "RGEM": "Səhiyyə Nazirliyi Respublika Gigiyena və Epidemiologiya Mərkəzi",
    "ADY": "Azərbaycan Dəmir Yolları",
    "ISB": "İcbari Sığorta Bürosu",
    "İSB": "İcbari Sığorta Bürosu",
    "AZRSKM": 'Səhiyyə Nazirliyi "Respublika Sanitariya-Karantin Mərkəzi" publik hüquqi şəxsi (PHŞ)',
    "ABADA": "Azərbaycan Beynəlxalq Avtomobil Daşıyıcıları Assosiyası",
    "IITKM": "Azərbaycan Respublikası İqtisadi İslahatların Təhlili və Kommunikasiya Mərkəzi",
    "İİTKM": "Azərbaycan Respublikası İqtisadi İslahatların Təhlili və Kommunikasiya Mərkəzi",
    "ASCO": "Azərbaycan Xəzər Dəniz Gəmiçiliyi QSC",
    "DSX": "Azərbaycan Respublikası Dövlət Sərhəd Xidməti",
    "AEM": "Azərbaycan Respublikası Səhiyyə Nazirliyi Analitik Ekspertiza Mərkəzi",
    "DGK": "Azərbaycan Respublikası Dövlət Gömrük Komitəsi",
    "AYNA": "Azərbaycan Yerüstü Nəqliyyat Agentliyi",
    "DVX": "Azərbaycan İqtisadiyyat Nazirliyi yanında Dövlət Vergi Xidməti",
    "IMEM": "Azərbaycan Respublikasının Prezidenti yanında Antiinhisar və İstehlak Bazarına Nəzarət Dövlət Agentliyinin İstehlak Mallarının Ekspertizası Mərkəzi",
    "İMEM": "Azərbaycan Respublikasının Prezidenti yanında Antiinhisar və İstehlak Bazarına Nəzarət Dövlət Agentliyinin İstehlak Mallarının Ekspertizası Mərkəzi",
    "BMFM": "Azərbaycan Respublikasının Kənd Təsərrüfatı Nazirliyi yanında Aqrar Xidmətlər Agentliyinin Bitki Mühafizəsi və Fumiqasiya Mərkəzi",
    "AQTA": "Azərbaycan Respublikasının Qida Təhlükəsizliyi Agentliyi",
    "PTX": "Azərbaycan Respublikası Prezidentinin Təhlükəsizlik Xidməti",
    "XRITDX": "Azərbaycan Respublikasının Xüsusi Rabitə və İnformasiya Təhlükəsizliyi",
    "XRİTDX": "Azərbaycan Respublikasının Xüsusi Rabitə və İnformasiya Təhlükəsizliyi",
    "AIBND": "Azərbaycan Respublikasının Prezidenti yanında Antiinhisar və İstehlak Bazarına Nəzarət Dövlət Agentliyi",
    "AMEA": "Azərbaycan Milli Elmlər Akademiyası",
    "ETN": "Azərbaycan Respublikası Elm və Təhsil Nazirliyi",
    "AMN": "Azərbaycan Respublikası Mədəniyyət Nazirliyi",
    "NK": "Azərbaycan Respublikasının Nazirlər Kabineti",
    "MSN": "Azərbaycan Respublikası Müdafiə Sənayesi Nazirliyi",
    "DTX": "Azərbaycan Respublikası Dövlət Təhlükəsizlik Xidməti",
    "AEN": "Azərbaycan Respublikası Energetika Nazirliyi",
    "ETSN": "Azərbaycan Respublikası Ekologiya və Təbii Sərvətlər Nazirliyi",
    "ARƏMA": "Azərbaycan Respublikası Əqli Mülkiyyət Agentliyi",
    "DQIDK": "Azərbaycan Respublikası Dini Qurumlarla İş üzrə Dövlət Komitəsi",
    "FHN": "Azərbaycan Respublikasının Fövqəladə Hallar Nazirliyi",
    "RINN": "Rəqəmsal İnkişaf və Nəqliyyat Nazirliyi",
    "RİNN": "Rəqəmsal İnkişaf və Nəqliyyat Nazirliyi",
    "SN":"Azərbaycan Respublikası Səhiyyə Nazirliyi"
}


def build_organization_query(search_term: str) -> dict:
    """
    Build Elasticsearch query for organization search.
    
    Args:
        search_term: The search term (can be full name or abbreviation)
        
    Returns:
        Elasticsearch query dict
    """
    # Check if search term is a known abbreviation
    if search_term.upper() in ORGANIZATION_ABBREVIATIONS.keys():
        search_term = ORGANIZATION_ABBREVIATIONS[search_term.upper()]
    
    # Extract words from search term (split on spaces and periods)
    words = re.findall(r'([^\s.]+)(?=\.|$)', search_term)
    
    if not words:
        return {"match_none": {}}
    
    # Single word search: use fuzzy match or prefix
    
    if len(words) <= 1:
        return {
            "bool": {
                "should": [
                    {
                        "match": {
                            "name": {
                                "query": search_term,
                                "fuzziness": "AUTO",
                                "boost": 2.0
                            }
                        }
                    },
                    {
                        "prefix": {
                            "name": {
                                "value": search_term,
                                "boost": 3.0
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        }
    
    
    # Multiple words: require ALL tokens to match (AND logic)
    must_clauses = []
    for token in words:
        token_should = []
        
        # Use fuzziness for tokens >= 3 characters
        if len(token) >= 3:
            token_should.append({
                "match": {
                    "name": {
                        "query": token,
                        "fuzziness": "AUTO"
                    }
                }
            })
        else:
            token_should.append({
                "match": {
                    "name": {
                        "query": token
                    }
                }
            })
        
        # Add prefix match for better results
        token_should.append({
            "prefix": {
                "name": {
                    "value": token
                }
            }
        })
        
        must_clauses.append({
            "bool": {
                "should": token_should,
                "minimum_should_match": 1
            }
        })
    
    return {
        "bool": {
            "must": must_clauses
        }
    }
