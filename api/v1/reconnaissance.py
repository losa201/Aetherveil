"""
Reconnaissance API Endpoints

REST API endpoints for reconnaissance operations including passive and active
information gathering, DNS enumeration, and infrastructure discovery.

Security Level: DEFENSIVE_ONLY
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, status
from pydantic import BaseModel, Field, validator

from ...modules import get_module
from ...modules.reconnaissance import ReconTarget, ReconMode, TargetType, ReconResult
from . import STANDARD_RESPONSES, APIResponseStatus

router = APIRouter()

# Pydantic models
class ReconModeEnum(str, Enum):
    """Reconnaissance mode enumeration"""
    PASSIVE = "passive"
    ACTIVE = "active"
    STEALTH = "stealth"
    AGGRESSIVE = "aggressive"

class TargetTypeEnum(str, Enum):
    """Target type enumeration"""
    DOMAIN = "domain"
    IP_RANGE = "ip_range"
    SINGLE_IP = "single_ip"
    URL = "url"
    ORGANIZATION = "organization"

class ReconTargetRequest(BaseModel):
    """Reconnaissance target request model"""
    target: str = Field(..., description="Target for reconnaissance", example="example.com")
    target_type: TargetTypeEnum = Field(..., description="Type of target")
    mode: ReconModeEnum = Field(default=ReconModeEnum.PASSIVE, description="Reconnaissance mode")
    depth: int = Field(default=1, ge=1, le=5, description="Reconnaissance depth level")
    timeout: int = Field(default=30, ge=10, le=300, description="Operation timeout in seconds")
    
    @validator('target')
    def validate_target(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Target cannot be empty')
        if len(v) > 255:
            raise ValueError('Target length cannot exceed 255 characters')
        return v.strip()

class BulkReconRequest(BaseModel):
    """Bulk reconnaissance request model"""
    targets: List[ReconTargetRequest] = Field(..., description="List of targets to scan")
    concurrent_limit: int = Field(default=5, ge=1, le=10, description="Maximum concurrent operations")
    
    @validator('targets')
    def validate_targets(cls, v):
        if len(v) == 0:
            raise ValueError('At least one target is required')
        if len(v) > 100:
            raise ValueError('Maximum 100 targets allowed per request')
        return v

class ReconResultResponse(BaseModel):
    """Reconnaissance result response model"""
    target: str
    target_type: str
    technique: str
    timestamp: datetime
    data: Dict[str, Any]
    confidence: float
    source: str
    metadata: Dict[str, Any] = {}

class ReconOperationResponse(BaseModel):
    """Reconnaissance operation response model"""
    status: APIResponseStatus
    operation_id: str
    target: str
    results: List[ReconResultResponse]
    duration: float
    metadata: Dict[str, Any] = {}

class ReconStatusResponse(BaseModel):
    """Reconnaissance module status response"""
    module: str
    status: str
    version: str
    results_count: int
    last_activity: Optional[str]
    techniques_used: List[str]
    targets_scanned: List[str]

# Dependency to get reconnaissance module
async def get_recon_module():
    """Get reconnaissance module instance"""
    module = get_module("reconnaissance")
    if not module:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reconnaissance module not available"
        )
    return module

# Endpoints
@router.get("/status", response_model=ReconStatusResponse, responses=STANDARD_RESPONSES)
async def get_reconnaissance_status(module=Depends(get_recon_module)):
    """Get reconnaissance module status and statistics"""
    try:
        status_info = await module.get_status()
        return ReconStatusResponse(**status_info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )

@router.post("/scan", response_model=ReconOperationResponse, responses=STANDARD_RESPONSES)
async def execute_reconnaissance(
    target_request: ReconTargetRequest,
    module=Depends(get_recon_module)
):
    """Execute reconnaissance against a target"""
    try:
        # Convert request to ReconTarget
        recon_target = ReconTarget(
            target=target_request.target,
            target_type=TargetType(target_request.target_type.value),
            mode=ReconMode(target_request.mode.value),
            depth=target_request.depth,
            timeout=target_request.timeout
        )
        
        # Execute reconnaissance
        start_time = datetime.utcnow()
        results = await module.execute_reconnaissance(recon_target)
        end_time = datetime.utcnow()
        
        # Convert results to response format
        result_responses = []
        for result in results:
            result_responses.append(ReconResultResponse(
                target=result.target,
                target_type=result.target_type.value,
                technique=result.technique,
                timestamp=result.timestamp,
                data=result.data,
                confidence=result.confidence,
                source=result.source,
                metadata=result.metadata
            ))
        
        return ReconOperationResponse(
            status=APIResponseStatus.COMPLETED,
            operation_id=f"recon_{int(start_time.timestamp())}",
            target=target_request.target,
            results=result_responses,
            duration=(end_time - start_time).total_seconds(),
            metadata={
                "mode": target_request.mode.value,
                "depth": target_request.depth,
                "results_count": len(results)
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reconnaissance failed: {str(e)}"
        )

@router.post("/bulk-scan", response_model=List[ReconOperationResponse], responses=STANDARD_RESPONSES)
async def execute_bulk_reconnaissance(
    bulk_request: BulkReconRequest,
    module=Depends(get_recon_module)
):
    """Execute reconnaissance against multiple targets"""
    try:
        # Convert requests to ReconTargets
        recon_targets = []
        for target_request in bulk_request.targets:
            recon_target = ReconTarget(
                target=target_request.target,
                target_type=TargetType(target_request.target_type.value),
                mode=ReconMode(target_request.mode.value),
                depth=target_request.depth,
                timeout=target_request.timeout
            )
            recon_targets.append(recon_target)
        
        # Execute bulk reconnaissance
        start_time = datetime.utcnow()
        bulk_results = await module.bulk_reconnaissance(recon_targets)
        end_time = datetime.utcnow()
        
        # Convert results to response format
        operation_responses = []
        for target, results in bulk_results.items():
            result_responses = []
            for result in results:
                result_responses.append(ReconResultResponse(
                    target=result.target,
                    target_type=result.target_type.value,
                    technique=result.technique,
                    timestamp=result.timestamp,
                    data=result.data,
                    confidence=result.confidence,
                    source=result.source,
                    metadata=result.metadata
                ))
            
            operation_responses.append(ReconOperationResponse(
                status=APIResponseStatus.COMPLETED,
                operation_id=f"bulk_recon_{int(start_time.timestamp())}_{target}",
                target=target,
                results=result_responses,
                duration=(end_time - start_time).total_seconds(),
                metadata={
                    "bulk_operation": True,
                    "results_count": len(results)
                }
            ))
        
        return operation_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk reconnaissance failed: {str(e)}"
        )

@router.get("/results", response_model=List[ReconResultResponse], responses=STANDARD_RESPONSES)
async def get_reconnaissance_results(
    target: Optional[str] = Query(None, description="Filter by target"),
    technique: Optional[str] = Query(None, description="Filter by technique"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
    module=Depends(get_recon_module)
):
    """Get reconnaissance results with optional filtering"""
    try:
        # Get results from module
        results = module.get_results(target=target, technique=technique)
        
        # Limit results
        if limit:
            results = results[:limit]
        
        # Convert to response format
        result_responses = []
        for result in results:
            result_responses.append(ReconResultResponse(
                target=result.target,
                target_type=result.target_type.value,
                technique=result.technique,
                timestamp=result.timestamp,
                data=result.data,
                confidence=result.confidence,
                source=result.source,
                metadata=result.metadata
            ))
        
        return result_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get results: {str(e)}"
        )

@router.get("/export", responses=STANDARD_RESPONSES)
async def export_reconnaissance_results(
    format: str = Query("json", regex="^(json|csv)$", description="Export format"),
    module=Depends(get_recon_module)
):
    """Export reconnaissance results"""
    try:
        exported_data = module.export_results(format=format)
        
        if format == "json":
            return {"data": exported_data}
        elif format == "csv":
            return {"data": exported_data}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}"
        )

@router.delete("/results", responses=STANDARD_RESPONSES)
async def clear_reconnaissance_results(
    confirm: bool = Query(False, description="Confirm clearing all results"),
    module=Depends(get_recon_module)
):
    """Clear all reconnaissance results"""
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must confirm clearing results by setting confirm=true"
        )
    
    try:
        # Clear results (this would need to be implemented in the module)
        module.results.clear()
        return {"message": "All reconnaissance results cleared"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear results: {str(e)}"
        )

@router.get("/techniques", responses=STANDARD_RESPONSES)
async def get_available_techniques():
    """Get list of available reconnaissance techniques"""
    return {
        "techniques": [
            {
                "name": "dns_reconnaissance",
                "description": "DNS record enumeration and analysis",
                "passive": True,
                "active": False
            },
            {
                "name": "subdomain_enumeration",
                "description": "Subdomain discovery and enumeration",
                "passive": False,
                "active": True
            },
            {
                "name": "ssl_reconnaissance",
                "description": "SSL/TLS certificate analysis",
                "passive": True,
                "active": False
            },
            {
                "name": "web_reconnaissance",
                "description": "Web application reconnaissance",
                "passive": False,
                "active": True
            },
            {
                "name": "network_discovery",
                "description": "Network infrastructure discovery",
                "passive": False,
                "active": True
            },
            {
                "name": "port_discovery",
                "description": "Port scanning and service detection",
                "passive": False,
                "active": True
            }
        ]
    }

@router.get("/modes", responses=STANDARD_RESPONSES)
async def get_reconnaissance_modes():
    """Get available reconnaissance modes"""
    return {
        "modes": [
            {
                "name": "passive",
                "description": "Passive information gathering without direct interaction",
                "risk_level": "low",
                "detection_risk": "minimal"
            },
            {
                "name": "active",
                "description": "Active information gathering with direct target interaction",
                "risk_level": "medium",
                "detection_risk": "medium"
            },
            {
                "name": "stealth",
                "description": "Stealthy reconnaissance with evasion techniques",
                "risk_level": "low",
                "detection_risk": "low"
            },
            {
                "name": "aggressive",
                "description": "Aggressive reconnaissance for comprehensive coverage",
                "risk_level": "high",
                "detection_risk": "high"
            }
        ]
    }