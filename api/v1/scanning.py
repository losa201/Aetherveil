"""
Scanning API Endpoints

REST API endpoints for vulnerability scanning operations including network scanning,
web application testing, and vulnerability detection.

Security Level: DEFENSIVE_ONLY
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, status
from pydantic import BaseModel, Field, validator

from ...modules import get_module
from ...modules.scanning import ScanTarget, ScanType, ScanIntensity, VulnerabilitySeverity, ScanResult, Vulnerability
from . import STANDARD_RESPONSES, APIResponseStatus

router = APIRouter()

# Pydantic models
class ScanTypeEnum(str, Enum):
    """Scan type enumeration"""
    NETWORK_DISCOVERY = "network_discovery"
    PORT_SCAN = "port_scan"
    SERVICE_DETECTION = "service_detection"
    VULNERABILITY_SCAN = "vulnerability_scan"
    WEB_APPLICATION_SCAN = "web_application_scan"
    SSL_SCAN = "ssl_scan"
    COMPLIANCE_SCAN = "compliance_scan"

class ScanIntensityEnum(str, Enum):
    """Scan intensity enumeration"""
    STEALTH = "stealth"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    COMPREHENSIVE = "comprehensive"

class VulnerabilitySeverityEnum(str, Enum):
    """Vulnerability severity enumeration"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ScanTargetRequest(BaseModel):
    """Scan target request model"""
    target: str = Field(..., description="Target for scanning", example="192.168.1.1")
    scan_type: ScanTypeEnum = Field(..., description="Type of scan to perform")
    intensity: ScanIntensityEnum = Field(default=ScanIntensityEnum.NORMAL, description="Scan intensity level")
    ports: Optional[str] = Field(None, description="Port specification (e.g., '1-1000', '80,443')")
    timeout: int = Field(default=300, ge=30, le=3600, description="Scan timeout in seconds")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional scan options")
    
    @validator('target')
    def validate_target(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Target cannot be empty')
        if len(v) > 255:
            raise ValueError('Target length cannot exceed 255 characters')
        return v.strip()

class BulkScanRequest(BaseModel):
    """Bulk scan request model"""
    targets: List[ScanTargetRequest] = Field(..., description="List of targets to scan")
    concurrent_limit: int = Field(default=3, ge=1, le=5, description="Maximum concurrent scans")
    
    @validator('targets')
    def validate_targets(cls, v):
        if len(v) == 0:
            raise ValueError('At least one target is required')
        if len(v) > 50:
            raise ValueError('Maximum 50 targets allowed per request')
        return v

class VulnerabilityResponse(BaseModel):
    """Vulnerability response model"""
    vuln_id: str
    name: str
    description: str
    severity: VulnerabilitySeverityEnum
    cvss_score: Optional[float]
    cve_id: Optional[str]
    affected_service: str
    evidence: Dict[str, Any]
    remediation: str
    references: List[str] = []

class ServiceResponse(BaseModel):
    """Service response model"""
    host: str
    port: int
    protocol: str
    service: str
    product: str = ""
    version: str = ""
    extrainfo: str = ""
    banner: Optional[str] = None

class ScanResultResponse(BaseModel):
    """Scan result response model"""
    target: str
    scan_type: ScanTypeEnum
    timestamp: datetime
    duration: float
    status: str
    vulnerabilities: List[VulnerabilityResponse]
    services: List[ServiceResponse]
    metadata: Dict[str, Any] = {}

class ScanOperationResponse(BaseModel):
    """Scan operation response model"""
    status: APIResponseStatus
    operation_id: str
    target: str
    scan_type: ScanTypeEnum
    result: ScanResultResponse
    metadata: Dict[str, Any] = {}

class ScanStatusResponse(BaseModel):
    """Scan module status response"""
    module: str
    status: str
    version: str
    scans_performed: int
    total_vulnerabilities: int
    vulnerability_breakdown: Dict[str, int]
    last_scan: Optional[str]

# Dependency to get scanning module
async def get_scan_module():
    """Get scanning module instance"""
    module = get_module("scanning")
    if not module:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scanning module not available"
        )
    return module

# Endpoints
@router.get("/status", response_model=ScanStatusResponse, responses=STANDARD_RESPONSES)
async def get_scanning_status(module=Depends(get_scan_module)):
    """Get scanning module status and statistics"""
    try:
        status_info = await module.get_status()
        return ScanStatusResponse(**status_info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )

@router.post("/scan", response_model=ScanOperationResponse, responses=STANDARD_RESPONSES)
async def execute_scan(
    scan_request: ScanTargetRequest,
    module=Depends(get_scan_module)
):
    """Execute a scan against a target"""
    try:
        # Convert request to ScanTarget
        scan_target = ScanTarget(
            target=scan_request.target,
            scan_type=ScanType(scan_request.scan_type.value),
            intensity=ScanIntensity(scan_request.intensity.value),
            ports=scan_request.ports,
            timeout=scan_request.timeout,
            options=scan_request.options
        )
        
        # Execute scan
        start_time = datetime.utcnow()
        scan_result = await module.execute_scan(scan_target)
        
        # Convert result to response format
        vulnerabilities = []
        for vuln in scan_result.vulnerabilities:
            vulnerabilities.append(VulnerabilityResponse(
                vuln_id=vuln.vuln_id,
                name=vuln.name,
                description=vuln.description,
                severity=VulnerabilitySeverityEnum(vuln.severity.value),
                cvss_score=vuln.cvss_score,
                cve_id=vuln.cve_id,
                affected_service=vuln.affected_service,
                evidence=vuln.evidence,
                remediation=vuln.remediation,
                references=vuln.references
            ))
        
        services = []
        for service in scan_result.services:
            services.append(ServiceResponse(
                host=service.get("host", ""),
                port=service.get("port", 0),
                protocol=service.get("protocol", ""),
                service=service.get("service", ""),
                product=service.get("product", ""),
                version=service.get("version", ""),
                extrainfo=service.get("extrainfo", ""),
                banner=service.get("banner")
            ))
        
        scan_response = ScanResultResponse(
            target=scan_result.target,
            scan_type=ScanTypeEnum(scan_result.scan_type.value),
            timestamp=scan_result.timestamp,
            duration=scan_result.duration.total_seconds(),
            status=scan_result.status,
            vulnerabilities=vulnerabilities,
            services=services,
            metadata=scan_result.metadata
        )
        
        return ScanOperationResponse(
            status=APIResponseStatus.COMPLETED,
            operation_id=f"scan_{int(start_time.timestamp())}",
            target=scan_request.target,
            scan_type=ScanTypeEnum(scan_request.scan_type.value),
            result=scan_response,
            metadata={
                "intensity": scan_request.intensity.value,
                "vulnerabilities_found": len(vulnerabilities),
                "services_found": len(services)
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scan failed: {str(e)}"
        )

@router.post("/bulk-scan", response_model=List[ScanOperationResponse], responses=STANDARD_RESPONSES)
async def execute_bulk_scan(
    bulk_request: BulkScanRequest,
    module=Depends(get_scan_module)
):
    """Execute scans against multiple targets"""
    try:
        # Convert requests to ScanTargets
        scan_targets = []
        for scan_request in bulk_request.targets:
            scan_target = ScanTarget(
                target=scan_request.target,
                scan_type=ScanType(scan_request.scan_type.value),
                intensity=ScanIntensity(scan_request.intensity.value),
                ports=scan_request.ports,
                timeout=scan_request.timeout,
                options=scan_request.options
            )
            scan_targets.append(scan_target)
        
        # Execute bulk scan
        start_time = datetime.utcnow()
        scan_results = await module.bulk_scan(scan_targets)
        
        # Convert results to response format
        operation_responses = []
        for scan_result in scan_results:
            # Convert vulnerabilities
            vulnerabilities = []
            for vuln in scan_result.vulnerabilities:
                vulnerabilities.append(VulnerabilityResponse(
                    vuln_id=vuln.vuln_id,
                    name=vuln.name,
                    description=vuln.description,
                    severity=VulnerabilitySeverityEnum(vuln.severity.value),
                    cvss_score=vuln.cvss_score,
                    cve_id=vuln.cve_id,
                    affected_service=vuln.affected_service,
                    evidence=vuln.evidence,
                    remediation=vuln.remediation,
                    references=vuln.references
                ))
            
            # Convert services
            services = []
            for service in scan_result.services:
                services.append(ServiceResponse(
                    host=service.get("host", ""),
                    port=service.get("port", 0),
                    protocol=service.get("protocol", ""),
                    service=service.get("service", ""),
                    product=service.get("product", ""),
                    version=service.get("version", ""),
                    extrainfo=service.get("extrainfo", ""),
                    banner=service.get("banner")
                ))
            
            scan_response = ScanResultResponse(
                target=scan_result.target,
                scan_type=ScanTypeEnum(scan_result.scan_type.value),
                timestamp=scan_result.timestamp,
                duration=scan_result.duration.total_seconds(),
                status=scan_result.status,
                vulnerabilities=vulnerabilities,
                services=services,
                metadata=scan_result.metadata
            )
            
            operation_responses.append(ScanOperationResponse(
                status=APIResponseStatus.COMPLETED,
                operation_id=f"bulk_scan_{int(start_time.timestamp())}_{scan_result.target}",
                target=scan_result.target,
                scan_type=ScanTypeEnum(scan_result.scan_type.value),
                result=scan_response,
                metadata={
                    "bulk_operation": True,
                    "vulnerabilities_found": len(vulnerabilities),
                    "services_found": len(services)
                }
            ))
        
        return operation_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk scan failed: {str(e)}"
        )

@router.get("/results", response_model=List[ScanResultResponse], responses=STANDARD_RESPONSES)
async def get_scan_results(
    target: Optional[str] = Query(None, description="Filter by target"),
    scan_type: Optional[ScanTypeEnum] = Query(None, description="Filter by scan type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
    module=Depends(get_scan_module)
):
    """Get scan results with optional filtering"""
    try:
        # Get results from module
        scan_type_filter = ScanType(scan_type.value) if scan_type else None
        results = module.get_scan_results(target=target, scan_type=scan_type_filter)
        
        # Limit results
        if limit:
            results = results[:limit]
        
        # Convert to response format
        scan_responses = []
        for scan_result in results:
            # Convert vulnerabilities
            vulnerabilities = []
            for vuln in scan_result.vulnerabilities:
                vulnerabilities.append(VulnerabilityResponse(
                    vuln_id=vuln.vuln_id,
                    name=vuln.name,
                    description=vuln.description,
                    severity=VulnerabilitySeverityEnum(vuln.severity.value),
                    cvss_score=vuln.cvss_score,
                    cve_id=vuln.cve_id,
                    affected_service=vuln.affected_service,
                    evidence=vuln.evidence,
                    remediation=vuln.remediation,
                    references=vuln.references
                ))
            
            # Convert services
            services = []
            for service in scan_result.services:
                services.append(ServiceResponse(
                    host=service.get("host", ""),
                    port=service.get("port", 0),
                    protocol=service.get("protocol", ""),
                    service=service.get("service", ""),
                    product=service.get("product", ""),
                    version=service.get("version", ""),
                    extrainfo=service.get("extrainfo", ""),
                    banner=service.get("banner")
                ))
            
            scan_responses.append(ScanResultResponse(
                target=scan_result.target,
                scan_type=ScanTypeEnum(scan_result.scan_type.value),
                timestamp=scan_result.timestamp,
                duration=scan_result.duration.total_seconds(),
                status=scan_result.status,
                vulnerabilities=vulnerabilities,
                services=services,
                metadata=scan_result.metadata
            ))
        
        return scan_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get results: {str(e)}"
        )

@router.get("/vulnerabilities", response_model=List[VulnerabilityResponse], responses=STANDARD_RESPONSES)
async def get_vulnerabilities(
    severity: Optional[VulnerabilitySeverityEnum] = Query(None, description="Filter by severity"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
    module=Depends(get_scan_module)
):
    """Get vulnerabilities with optional filtering"""
    try:
        # Get vulnerabilities from module
        severity_filter = VulnerabilitySeverity(severity.value) if severity else None
        vulnerabilities = module.get_vulnerabilities(severity=severity_filter)
        
        # Limit results
        if limit:
            vulnerabilities = vulnerabilities[:limit]
        
        # Convert to response format
        vuln_responses = []
        for vuln in vulnerabilities:
            vuln_responses.append(VulnerabilityResponse(
                vuln_id=vuln.vuln_id,
                name=vuln.name,
                description=vuln.description,
                severity=VulnerabilitySeverityEnum(vuln.severity.value),
                cvss_score=vuln.cvss_score,
                cve_id=vuln.cve_id,
                affected_service=vuln.affected_service,
                evidence=vuln.evidence,
                remediation=vuln.remediation,
                references=vuln.references
            ))
        
        return vuln_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vulnerabilities: {str(e)}"
        )

@router.get("/export", responses=STANDARD_RESPONSES)
async def export_scan_results(
    format: str = Query("json", regex="^(json|csv)$", description="Export format"),
    module=Depends(get_scan_module)
):
    """Export scan results"""
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

@router.get("/scan-types", responses=STANDARD_RESPONSES)
async def get_scan_types():
    """Get available scan types"""
    return {
        "scan_types": [
            {
                "name": "network_discovery",
                "description": "Discover active hosts in network ranges",
                "target_type": "network_range",
                "estimated_duration": "fast"
            },
            {
                "name": "port_scan",
                "description": "Scan for open ports on target hosts",
                "target_type": "host",
                "estimated_duration": "medium"
            },
            {
                "name": "service_detection",
                "description": "Detect services and versions on open ports",
                "target_type": "host",
                "estimated_duration": "medium"
            },
            {
                "name": "vulnerability_scan",
                "description": "Comprehensive vulnerability assessment",
                "target_type": "host",
                "estimated_duration": "slow"
            },
            {
                "name": "web_application_scan",
                "description": "Web application security testing",
                "target_type": "web_application",
                "estimated_duration": "slow"
            },
            {
                "name": "ssl_scan",
                "description": "SSL/TLS configuration assessment",
                "target_type": "host",
                "estimated_duration": "fast"
            }
        ]
    }

@router.get("/intensities", responses=STANDARD_RESPONSES)
async def get_scan_intensities():
    """Get available scan intensities"""
    return {
        "intensities": [
            {
                "name": "stealth",
                "description": "Slow, stealthy scanning to avoid detection",
                "speed": "very_slow",
                "detection_risk": "low",
                "accuracy": "medium"
            },
            {
                "name": "normal",
                "description": "Standard scanning speed and accuracy",
                "speed": "medium",
                "detection_risk": "medium",
                "accuracy": "high"
            },
            {
                "name": "aggressive",
                "description": "Fast scanning with higher detection risk",
                "speed": "fast",
                "detection_risk": "high",
                "accuracy": "high"
            },
            {
                "name": "comprehensive",
                "description": "Thorough scanning with all available techniques",
                "speed": "slow",
                "detection_risk": "high",
                "accuracy": "very_high"
            }
        ]
    }