"""
OSINT API Endpoints

REST API endpoints for Open Source Intelligence operations including
automated data collection, analysis, and correlation from public sources.

Security Level: DEFENSIVE_ONLY
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, status
from pydantic import BaseModel, Field, validator

from ...modules import get_module
from ...modules.osint import OSINTQuery, OSINTSource, DataType, ConfidenceLevel, OSINTResult, IntelligenceData
from . import STANDARD_RESPONSES, APIResponseStatus

router = APIRouter()

# Pydantic models
class OSINTSourceEnum(str, Enum):
    """OSINT source enumeration"""
    SEARCH_ENGINES = "search_engines"
    SOCIAL_MEDIA = "social_media"
    DOMAIN_RECORDS = "domain_records"
    CERTIFICATE_TRANSPARENCY = "certificate_transparency"
    THREAT_INTELLIGENCE = "threat_intelligence"
    CODE_REPOSITORIES = "code_repositories"
    BREACH_DATABASES = "breach_databases"
    NETWORK_SCANNING = "network_scanning"
    METADATA_EXTRACTION = "metadata_extraction"
    GEOSPATIAL = "geospatial"

class DataTypeEnum(str, Enum):
    """Data type enumeration"""
    EMAIL = "email"
    DOMAIN = "domain"
    IP_ADDRESS = "ip_address"
    PHONE_NUMBER = "phone_number"
    USERNAME = "username"
    ORGANIZATION = "organization"
    PERSON = "person"
    CERTIFICATE = "certificate"
    VULNERABILITY = "vulnerability"
    BREACH_DATA = "breach_data"

class ConfidenceLevelEnum(str, Enum):
    """Confidence level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"

class OSINTQueryRequest(BaseModel):
    """OSINT query request model"""
    target: str = Field(..., description="Target for OSINT collection")
    data_type: DataTypeEnum = Field(..., description="Type of data to collect")
    sources: List[OSINTSourceEnum] = Field(..., description="OSINT sources to query")
    depth: int = Field(default=1, ge=1, le=3, description="Collection depth level")
    timeout: int = Field(default=300, ge=30, le=1800, description="Operation timeout in seconds")
    options: Dict[str, Any] = Field(default_factory=dict, description="Source-specific options")
    
    @validator('target')
    def validate_target(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Target cannot be empty')
        return v.strip()
    
    @validator('sources')
    def validate_sources(cls, v):
        if len(v) == 0:
            raise ValueError('At least one source is required')
        return v

class IntelligenceDataResponse(BaseModel):
    """Intelligence data response model"""
    data_type: DataTypeEnum
    value: str
    source: OSINTSourceEnum
    timestamp: datetime
    confidence: ConfidenceLevelEnum
    metadata: Dict[str, Any]
    related_data: List[str] = []
    tags: List[str] = []

class OSINTResultResponse(BaseModel):
    """OSINT result response model"""
    target: str
    query_type: DataTypeEnum
    timestamp: datetime
    duration: float
    intelligence_data: List[IntelligenceDataResponse]
    sources_queried: List[OSINTSourceEnum]
    correlation_score: float
    metadata: Dict[str, Any] = {}

class OSINTOperationResponse(BaseModel):
    """OSINT operation response model"""
    status: APIResponseStatus
    operation_id: str
    target: str
    result: OSINTResultResponse
    metadata: Dict[str, Any] = {}

class OSINTStatusResponse(BaseModel):
    """OSINT module status response"""
    module: str
    status: str
    version: str
    queries_performed: int
    total_intelligence_points: int
    data_type_breakdown: Dict[str, int]
    source_breakdown: Dict[str, int]
    last_query: Optional[str]

# Dependency to get OSINT module
async def get_osint_module():
    """Get OSINT module instance"""
    module = get_module("osint")
    if not module:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OSINT module not available"
        )
    return module

# Endpoints
@router.get("/status", response_model=OSINTStatusResponse, responses=STANDARD_RESPONSES)
async def get_osint_status(module=Depends(get_osint_module)):
    """Get OSINT module status and statistics"""
    try:
        status_info = await module.get_status()
        return OSINTStatusResponse(**status_info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )

@router.post("/query", response_model=OSINTOperationResponse, responses=STANDARD_RESPONSES)
async def execute_osint_query(
    query_request: OSINTQueryRequest,
    module=Depends(get_osint_module)
):
    """Execute OSINT query against target"""
    try:
        # Convert request to OSINTQuery
        osint_query = OSINTQuery(
            target=query_request.target,
            data_type=DataType(query_request.data_type.value),
            sources=[OSINTSource(src.value) for src in query_request.sources],
            depth=query_request.depth,
            timeout=query_request.timeout,
            options=query_request.options
        )
        
        # Execute OSINT query
        start_time = datetime.utcnow()
        osint_result = await module.execute_osint_query(osint_query)
        
        # Convert intelligence data to response format
        intelligence_responses = []
        for intel_data in osint_result.intelligence_data:
            intelligence_responses.append(IntelligenceDataResponse(
                data_type=DataTypeEnum(intel_data.data_type.value),
                value=intel_data.value,
                source=OSINTSourceEnum(intel_data.source.value),
                timestamp=intel_data.timestamp,
                confidence=ConfidenceLevelEnum(intel_data.confidence.value),
                metadata=intel_data.metadata,
                related_data=intel_data.related_data,
                tags=intel_data.tags
            ))
        
        # Convert result to response format
        result_response = OSINTResultResponse(
            target=osint_result.target,
            query_type=DataTypeEnum(osint_result.query_type.value),
            timestamp=osint_result.timestamp,
            duration=osint_result.duration.total_seconds(),
            intelligence_data=intelligence_responses,
            sources_queried=[OSINTSourceEnum(src.value) for src in osint_result.sources_queried],
            correlation_score=osint_result.correlation_score,
            metadata=osint_result.metadata
        )
        
        return OSINTOperationResponse(
            status=APIResponseStatus.COMPLETED,
            operation_id=f"osint_{int(start_time.timestamp())}",
            target=query_request.target,
            result=result_response,
            metadata={
                "data_type": query_request.data_type.value,
                "sources_count": len(query_request.sources),
                "intelligence_points": len(intelligence_responses),
                "correlation_score": osint_result.correlation_score
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OSINT query failed: {str(e)}"
        )

@router.get("/search", response_model=List[IntelligenceDataResponse], responses=STANDARD_RESPONSES)
async def search_intelligence_database(
    query: str = Query(..., description="Search query"),
    data_type: Optional[DataTypeEnum] = Query(None, description="Filter by data type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
    module=Depends(get_osint_module)
):
    """Search intelligence database"""
    try:
        # Search intelligence database
        data_type_filter = DataType(data_type.value) if data_type else None
        results = module.search_intelligence_database(query, data_type_filter)
        
        # Limit results
        if limit:
            results = results[:limit]
        
        # Convert to response format
        intelligence_responses = []
        for intel_data in results:
            intelligence_responses.append(IntelligenceDataResponse(
                data_type=DataTypeEnum(intel_data.data_type.value),
                value=intel_data.value,
                source=OSINTSourceEnum(intel_data.source.value),
                timestamp=intel_data.timestamp,
                confidence=ConfidenceLevelEnum(intel_data.confidence.value),
                metadata=intel_data.metadata,
                related_data=intel_data.related_data,
                tags=intel_data.tags
            ))
        
        return intelligence_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intelligence search failed: {str(e)}"
        )

@router.get("/related/{target}", response_model=List[IntelligenceDataResponse], responses=STANDARD_RESPONSES)
async def get_related_intelligence(
    target: str = Path(..., description="Target to find related intelligence for"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
    module=Depends(get_osint_module)
):
    """Get intelligence data related to target"""
    try:
        # Get related intelligence
        results = module.get_related_intelligence(target)
        
        # Limit results
        if limit:
            results = results[:limit]
        
        # Convert to response format
        intelligence_responses = []
        for intel_data in results:
            intelligence_responses.append(IntelligenceDataResponse(
                data_type=DataTypeEnum(intel_data.data_type.value),
                value=intel_data.value,
                source=OSINTSourceEnum(intel_data.source.value),
                timestamp=intel_data.timestamp,
                confidence=ConfidenceLevelEnum(intel_data.confidence.value),
                metadata=intel_data.metadata,
                related_data=intel_data.related_data,
                tags=intel_data.tags
            ))
        
        return intelligence_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get related intelligence: {str(e)}"
        )

@router.get("/export", responses=STANDARD_RESPONSES)
async def export_intelligence_database(
    format: str = Query("json", regex="^(json|csv)$", description="Export format"),
    module=Depends(get_osint_module)
):
    """Export intelligence database"""
    try:
        exported_data = module.export_intelligence(format=format)
        
        if format == "json":
            return {"data": exported_data}
        elif format == "csv":
            return {"data": exported_data}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}"
        )

@router.get("/sources", responses=STANDARD_RESPONSES)
async def get_osint_sources():
    """Get available OSINT sources"""
    return {
        "sources": [
            {
                "name": "search_engines",
                "description": "Google, Bing, and other search engines",
                "data_types": ["organization", "person", "email", "domain"],
                "rate_limited": True
            },
            {
                "name": "social_media",
                "description": "LinkedIn, GitHub, and social platforms",
                "data_types": ["person", "organization", "username"],
                "rate_limited": True
            },
            {
                "name": "domain_records",
                "description": "WHOIS, DNS, and domain registration data",
                "data_types": ["domain", "ip_address", "email", "organization"],
                "rate_limited": False
            },
            {
                "name": "certificate_transparency",
                "description": "SSL certificate transparency logs",
                "data_types": ["domain", "certificate", "organization"],
                "rate_limited": False
            },
            {
                "name": "threat_intelligence",
                "description": "Shodan, Censys, and threat intelligence feeds",
                "data_types": ["ip_address", "domain", "vulnerability"],
                "rate_limited": True
            },
            {
                "name": "code_repositories",
                "description": "GitHub, GitLab, and code repositories",
                "data_types": ["username", "email", "organization"],
                "rate_limited": True
            },
            {
                "name": "breach_databases",
                "description": "Have I Been Pwned and breach databases",
                "data_types": ["email", "breach_data"],
                "rate_limited": True
            }
        ]
    }

@router.get("/data-types", responses=STANDARD_RESPONSES)
async def get_data_types():
    """Get available data types"""
    return {
        "data_types": [
            {
                "name": "email",
                "description": "Email addresses",
                "sources": ["search_engines", "domain_records", "breach_databases"]
            },
            {
                "name": "domain",
                "description": "Domain names and subdomains",
                "sources": ["domain_records", "certificate_transparency", "threat_intelligence"]
            },
            {
                "name": "ip_address",
                "description": "IP addresses and network information",
                "sources": ["domain_records", "threat_intelligence"]
            },
            {
                "name": "phone_number",
                "description": "Phone numbers and contact information",
                "sources": ["search_engines", "domain_records"]
            },
            {
                "name": "username",
                "description": "Usernames and social media handles",
                "sources": ["social_media", "code_repositories"]
            },
            {
                "name": "organization",
                "description": "Organization and company information",
                "sources": ["search_engines", "domain_records", "social_media"]
            },
            {
                "name": "person",
                "description": "Personal information and profiles",
                "sources": ["search_engines", "social_media"]
            },
            {
                "name": "certificate",
                "description": "SSL certificates and cryptographic data",
                "sources": ["certificate_transparency"]
            },
            {
                "name": "vulnerability",
                "description": "Vulnerability information",
                "sources": ["threat_intelligence"]
            },
            {
                "name": "breach_data",
                "description": "Data breach information",
                "sources": ["breach_databases"]
            }
        ]
    }