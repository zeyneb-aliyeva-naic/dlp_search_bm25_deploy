from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict
from config import ORGANIZATIONS_INDEX

class TradingType(str, Enum):
    """Trading type enumeration"""
    IMPORT = "IMPORT"
    EXPORT = "EXPORT"
    TRANSIT = "TRANSIT"


class VehicleType(str, Enum):
    """Vehicle type enumeration with ID mapping"""
    demiryolu = "demiryolu"  # ID: 3 - Dəmiryolu nəqliyyatı
    deniz = "deniz"          # ID: 1 - Dəniz nəqliyyatı
    avtomobil = "avtomobil"  # ID: 2 - Avtomobil nəqliyyatı
    hava = "hava"            # ID: 4 - Hava nəqliyyatı
    
    def to_id(self) -> int:
        """Convert vehicle type to numeric ID for Elasticsearch"""
        mapping = {
            "deniz": 1,
            "avtomobil": 2,
            "demiryolu": 3,
            "hava": 4
        }
        return mapping[self.value]


class Trading(BaseModel):
    """Trading information model"""
    id: Optional[str] = None
    tradeType: str
    tradeName: str
    inVehicleId: Optional[int] = None
    outVehicleId: Optional[int] = None


class Filter(BaseModel):
    """Search filter model"""
    model_config = ConfigDict(extra="ignore")
    
    trade_type: Optional[TradingType] = Field(
        default=None, 
        description="Single trade type filter"
    )
    in_vehicle_ids: List[VehicleType] = Field(
        default_factory=list, 
        description="Incoming vehicle IDs (applies with trade_type)"
    )
    out_vehicle_ids: List[VehicleType] = Field(
        default_factory=list, 
        description="Outgoing vehicle IDs (applies with trade_type)"
    )


class ElasticDocument(BaseModel):
    """Elasticsearch document model"""
    id: str
    code: Optional[str] = None
    score: Optional[float] = None
    name_az_d1: Optional[str] = None
    name_az_d2: Optional[str] = None
    name_az_d3: Optional[str] = None
    name_az_d4: Optional[str] = None
    tradings: List[Trading] = Field(default_factory=list)
    Path: Optional[str] = None
    highlight: Optional[Dict[str, List[str]]] = None 

    @staticmethod
    def build_path(p1, p2, p3):
        """Build hierarchical path from components"""
        parts = [p for p in (p1, p2, p3) if p]
        return " / ".join(parts)

    @staticmethod
    def from_es_hit(hit: dict) -> "ElasticDocument":
        """Create ElasticDocument from Elasticsearch hit"""
        src = hit.get("_source") or {}
        score = hit.get("_score")
        tradings_data = src.get("tradings", [])

        tradings = []
        if tradings_data:
            for t in tradings_data:
                if isinstance(t, dict):
                    try:
                        tradings.append(Trading(**t))
                    except Exception:
                        continue
        
        path = ElasticDocument.build_path(
            src.get("name_az_d1"), 
            src.get("name_az_d2"), 
            src.get("name_az_d3")
        )
        
        return ElasticDocument(
            id=src.get("id") or hit.get("_id"),
            code=src.get("code"),
            score=score,
            name_az_d1=src.get("name_az_d1"),
            name_az_d2=src.get("name_az_d2"),
            name_az_d3=src.get("name_az_d3"),
            name_az_d4=src.get("name_az_d4"),
            tradings=tradings,
            Path=path or src.get("Path"),
            highlight=hit.get("highlight"),
        )


class HybridRetrievedResponseSet(BaseModel):
    """Response model for hybrid search results"""
    model_config = ConfigDict(populate_by_name=True)
    
    query_text: str = Field(alias="query-text")
    total_hits: int = Field(alias="total-hits")
    ranked_objects: List[ElasticDocument] = Field(
        default_factory=list, 
        alias="Ranked-objects"
    )


class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(min_length=1)
    filter: Filter = Field(default_factory=Filter)
    size: int = Field(
        default=10, 
        ge=1, 
        le=200, 
        description="Number of results to return (same as top_k)"
    )
    top_k: Optional[int] = Field(
        default=None, 
        ge=1, 
        le=200, 
        description="Alias for size parameter (number of top results)"
    )
    alpha: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0, 
        description="Weight for vector similarity (0=BM25 only, 1=vector only)"
    )
    use_vector: bool = Field(
        default=True, 
        description="Enable vector search in hybrid mode"
    )
    use_highlight: bool = Field(
        default=False, 
        description="Return Elasticsearch highlight snippets"
    )
    
    def model_post_init(self, __context):
        """Use top_k as size if provided"""
        if self.top_k is not None:
            self.size = self.top_k


class SpellingRequest(BaseModel):
    query: str


class SpellingResponse(BaseModel):
    original_query: str
    corrected_query: str


# En search

class TradingTypeEng(str, Enum):
    """Trading type enumeration"""
    IMPORT = "IMPORT"
    EXPORT = "EXPORT"
    TRANSIT = "TRANSIT"


class VehicleTypeEng(str, Enum):
    """Vehicle type enumeration with ID mapping"""
    demiryolu = "railway"  # ID: 3 - Dəmiryolu nəqliyyatı
    deniz = "sea"          # ID: 1 - Dəniz nəqliyyatı
    avtomobil = "car"  # ID: 2 - Avtomobil nəqliyyatı
    hava = "air"            # ID: 4 - Hava nəqliyyatı
    
    def to_id(self) -> int:
        """Convert vehicle type to numeric ID for Elasticsearch"""
        mapping = {
            "sea": 1,
            "car": 2,
            "railway": 3,
            "air": 4
        }
        return mapping[self.value]


class TradingEng(BaseModel):
    """Trading information model"""
    id: Optional[str] = None
    tradeType: str
    tradeName: str
    inVehicleId: Optional[int] = None
    outVehicleId: Optional[int] = None


class FilterEng(BaseModel):
    """Search filter model"""
    model_config = ConfigDict(extra="ignore")
    
    trade_type: Optional[TradingTypeEng] = Field(
        default=None, 
        description="Single trade type filter"
    )
    in_vehicle_ids: List[VehicleTypeEng] = Field(
        default_factory=list, 
        description="Incoming vehicle IDs (applies with trade_type)"
    )
    out_vehicle_ids: List[VehicleTypeEng] = Field(
        default_factory=list, 
        description="Outgoing vehicle IDs (applies with trade_type)"
    )


class ElasticDocumentEng(BaseModel):
    """Elasticsearch document model"""
    id: str
    code: Optional[str] = None
    score: Optional[float] = None
    name_en_d1: Optional[str] = None
    name_en_d2: Optional[str] = None
    name_en_d3: Optional[str] = None
    name_en_d4: Optional[str] = None
    tradings: List[TradingEng] = Field(default_factory=list)
    Path: Optional[str] = None
    highlight: Optional[Dict[str, List[str]]] = None 

    @staticmethod
    def build_path(p1, p2, p3):
        """Build hierarchical path from components"""
        parts = [p for p in (p1, p2, p3) if p]
        return " / ".join(parts)

    @staticmethod
    def from_es_hit(hit: dict) -> "ElasticDocument":
        """Create ElasticDocument from Elasticsearch hit"""
        src = hit.get("_source") or {}
        score = hit.get("_score")
        tradings_data = src.get("tradings", [])

        tradings = []
        if tradings_data:
            for t in tradings_data:
                if isinstance(t, dict):
                    try:
                        tradings.append(Trading(**t))
                    except Exception:
                        continue
        
        path = ElasticDocument.build_path(
            src.get("name_en_d1"), 
            src.get("name_en_d2"), 
            src.get("name_en_d3")
        )
        
        return ElasticDocument(
            id=src.get("id") or hit.get("_id"),
            code=src.get("code"),
            score=score,
            name_az_d1=src.get("name_en_d1"),
            name_az_d2=src.get("name_en_d2"),
            name_az_d3=src.get("name_en_d3"),
            name_az_d4=src.get("name_en_d4"),
            tradings=tradings,
            Path=path or src.get("Path"),
            highlight=hit.get("highlight"),
        )


class HybridRetrievedResponseSetEng(BaseModel):
    """Response model for hybrid search results"""
    model_config = ConfigDict(populate_by_name=True)
    
    query_text: str = Field(alias="query-text")
    total_hits: int = Field(alias="total-hits")
    ranked_objects: List[ElasticDocumentEng] = Field(
        default_factory=list, 
        alias="Ranked-objects"
    )


class SearchRequestEng(BaseModel):
    """Search request model"""
    query: str = Field(min_length=1)
    filter: Filter = Field(default_factory=Filter)
    size: int = Field(
        default=10, 
        ge=1, 
        le=200, 
        description="Number of results to return (same as top_k)"
    )
    top_k: Optional[int] = Field(
        default=None, 
        ge=1, 
        le=200, 
        description="Alias for size parameter (number of top results)"
    )
    alpha: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0, 
        description="Weight for vector similarity (0=BM25 only, 1=vector only)"
    )
    use_vector: bool = Field(
        default=True, 
        description="Enable vector search in hybrid mode"
    )
    use_highlight: bool = Field(
        default=False, 
        description="Return Elasticsearch highlight snippets"
    )
    
    def model_post_init(self, __context):
        """Use top_k as size if provided"""
        if self.top_k is not None:
            self.size = self.top_k


class SpellingRequestEng(BaseModel):
    query: str


class SpellingResponseEng(BaseModel):
    original_query: str
    corrected_query: str

#ru search
class TradingTypeRu(str, Enum):
    """Trading type enumeration"""
    IMPORT = "IMPORT"
    EXPORT = "EXPORT"
    TRANSIT = "TRANSIT"


class VehicleTypeRu(str, Enum):
    """Vehicle type enumeration with ID mapping"""
    demiryolu = "железная дорога"  # ID: 3 - Dəmiryolu nəqliyyatı
    deniz = "море"          # ID: 1 - Dəniz nəqliyyatı
    avtomobil = "автомобиль"  # ID: 2 - Avtomobil nəqliyyatı
    hava = "воздух"            # ID: 4 - Hava nəqliyyatı
    
    def to_id(self) -> int:
        """Convert vehicle type to numeric ID for Elasticsearch"""
        mapping = {
            "море": 1,
            "автомобиль": 2,
            "железная дорога": 3,
            "воздух": 4
        }
        return mapping[self.value]


class TradingRu(BaseModel):
    """Trading information model"""
    id: Optional[str] = None
    tradeType: str
    tradeName: str
    inVehicleId: Optional[int] = None
    outVehicleId: Optional[int] = None


class FilterEng(BaseModel):
    """Search filter model"""
    model_config = ConfigDict(extra="ignore")
    
    trade_type: Optional[TradingTypeRu] = Field(
        default=None, 
        description="Single trade type filter"
    )
    in_vehicle_ids: List[VehicleTypeRu] = Field(
        default_factory=list, 
        description="Incoming vehicle IDs (applies with trade_type)"
    )
    out_vehicle_ids: List[VehicleTypeRu] = Field(
        default_factory=list, 
        description="Outgoing vehicle IDs (applies with trade_type)"
    )


class ElasticDocumentRu(BaseModel):
    """Elasticsearch document model"""
    id: str
    code: Optional[str] = None
    score: Optional[float] = None
    name_en_d1: Optional[str] = None
    name_en_d2: Optional[str] = None
    name_en_d3: Optional[str] = None
    name_en_d4: Optional[str] = None
    tradings: List[TradingRu] = Field(default_factory=list)
    Path: Optional[str] = None
    highlight: Optional[Dict[str, List[str]]] = None 

    @staticmethod
    def build_path(p1, p2, p3):
        """Build hierarchical path from components"""
        parts = [p for p in (p1, p2, p3) if p]
        return " / ".join(parts)

    @staticmethod
    def from_es_hit(hit: dict) -> "ElasticDocumentRu":
        """Create ElasticDocument from Elasticsearch hit"""
        src = hit.get("_source") or {}
        score = hit.get("_score")
        tradings_data = src.get("tradings", [])

        tradings = []
        if tradings_data:
            for t in tradings_data:
                if isinstance(t, dict):
                    try:
                        tradings.append(Trading(**t))
                    except Exception:
                        continue
        
        path = ElasticDocumentRu.build_path(
            src.get("name_ru_d1"), 
            src.get("name_ru_d2"), 
            src.get("name_ru_d3")
        )
        
        return ElasticDocumentRu(
            id=src.get("id") or hit.get("_id"),
            code=src.get("code"),
            score=score,
            name_az_d1=src.get("name_ru_d1"),
            name_az_d2=src.get("name_ru_d2"),
            name_az_d3=src.get("name_ru_d3"),
            name_az_d4=src.get("name_ru_d4"),
            tradings=tradings,
            Path=path or src.get("Path"),
            highlight=hit.get("highlight"),
        )


class HybridRetrievedResponseSetRu(BaseModel):
    """Response model for hybrid search results"""
    model_config = ConfigDict(populate_by_name=True)
    
    query_text: str = Field(alias="query-text")
    total_hits: int = Field(alias="total-hits")
    ranked_objects: List[ElasticDocumentRu] = Field(
        default_factory=list, 
        alias="Ranked-objects"
    )


class SearchRequestRu(BaseModel):
    """Search request model"""
    query: str = Field(min_length=1)
    filter: Filter = Field(default_factory=Filter)
    size: int = Field(
        default=10, 
        ge=1, 
        le=200, 
        description="Number of results to return (same as top_k)"
    )
    top_k: Optional[int] = Field(
        default=None, 
        ge=1, 
        le=200, 
        description="Alias for size parameter (number of top results)"
    )
    alpha: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0, 
        description="Weight for vector similarity (0=BM25 only, 1=vector only)"
    )
    use_vector: bool = Field(
        default=True, 
        description="Enable vector search in hybrid mode"
    )
    use_highlight: bool = Field(
        default=False, 
        description="Return Elasticsearch highlight snippets"
    )
    
    def model_post_init(self, __context):
        """Use top_k as size if provided"""
        if self.top_k is not None:
            self.size = self.top_k


class SpellingRequestRu(BaseModel):
    query: str


class SpellingResponseRu(BaseModel):
    original_query: str
    corrected_query: str

#org search
class OrganizationDocument(BaseModel):
    """Model for organization search results"""
    id: Optional[str] = None
    name: Optional[str] = None
    code: Optional[str] = None
    score: Optional[float] = None
    
    @staticmethod
    def from_es_hit(hit: dict) -> "OrganizationDocument":
        """Convert Elasticsearch hit to OrganizationDocument"""
        src = hit.get("_source") or {}
        score = hit.get("_score")
        
        return OrganizationDocument(
            id=src.get("id") or hit.get("_id"),
            name=src.get("name"),
            code=src.get("code"),
            score=score
        )


class OrganizationSearchRequest(BaseModel):
    """Request model for organization search"""
    search_term: str = Field(min_length=1, description="Organization name or abbreviation to search")
    index: str = Field(default=ORGANIZATIONS_INDEX, description="Elasticsearch index name")
    size: int = Field(default=10, ge=1, le=200, description="Number of results to return")


class OrganizationSearchResponse(BaseModel):
    """Response model for organization search"""
    model_config = ConfigDict(populate_by_name=True)
    search_term: str = Field(alias="search-term")
    total_hits: int = Field(alias="total-hits")
    results: List[OrganizationDocument] = Field(default_factory=list)
