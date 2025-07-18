"""
Reporting API Endpoints

REST API endpoints for report generation and analysis including
automated report creation, data visualization, and compliance reporting.

Security Level: DEFENSIVE_ONLY
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator

from ...modules import get_module
from ...modules.reporting import (
    ReportType, ReportFormat, SeverityLevel, ComplianceFramework,
    ReportMetadata, ReportData, Finding
)
from . import STANDARD_RESPONSES, APIResponseStatus

router = APIRouter()

# Pydantic models
class ReportTypeEnum(str, Enum):
    """Report type enumeration"""
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_DETAILED = "technical_detailed"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    PENETRATION_TEST = "penetration_test"
    COMPLIANCE_AUDIT = "compliance_audit"
    THREAT_INTELLIGENCE = "threat_intelligence"
    INCIDENT_RESPONSE = "incident_response"
    RISK_ASSESSMENT = "risk_assessment"

class ReportFormatEnum(str, Enum):
    """Report format enumeration"""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    MARKDOWN = "markdown"

class SeverityLevelEnum(str, Enum):
    """Severity level enumeration"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ComplianceFrameworkEnum(str, Enum):
    """Compliance framework enumeration"""
    NIST_CSF = "nist_csf"
    ISO_27001 = "iso_27001"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOX = "sox"
    GDPR = "gdpr"
    OWASP_TOP10 = "owasp_top10"

class ReportMetadataRequest(BaseModel):
    """Report metadata request model"""
    title: str = Field(..., description="Report title")
    description: str = Field(..., description="Report description")
    report_type: ReportTypeEnum = Field(..., description="Type of report")
    target_audience: str = Field(..., description="Target audience for the report")
    classification: str = Field(default="CONFIDENTIAL", description="Report classification")
    author: str = Field(..., description="Report author")
    organization: str = Field(..., description="Organization name")
    executive_summary: bool = Field(default=True, description="Include executive summary")
    include_charts: bool = Field(default=True, description="Include charts and visualizations")
    include_raw_data: bool = Field(default=False, description="Include raw data appendix")
    compliance_framework: Optional[ComplianceFrameworkEnum] = Field(None, description="Compliance framework")
    
    @validator('title', 'description', 'author', 'organization')
    def validate_required_fields(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Field cannot be empty')
        return v.strip()

class ReportGenerationRequest(BaseModel):
    """Report generation request model"""
    metadata: ReportMetadataRequest = Field(..., description="Report metadata")
    data_sources: List[str] = Field(..., description="Data sources to include in report")
    output_format: ReportFormatEnum = Field(default=ReportFormatEnum.PDF, description="Output format")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Data filters")
    
    @validator('data_sources')
    def validate_data_sources(cls, v):
        if len(v) == 0:
            raise ValueError('At least one data source is required')
        return v

class FindingResponse(BaseModel):
    """Finding response model"""
    finding_id: str
    title: str
    description: str
    severity: SeverityLevelEnum
    cvss_score: Optional[float]
    cve_id: Optional[str]
    affected_assets: List[str]
    evidence: Dict[str, Any]
    remediation: str
    references: List[str] = []
    compliance_mappings: Dict[str, List[str]] = {}
    timestamp: datetime
    verified: bool

class ReportGenerationResponse(BaseModel):
    """Report generation response model"""
    status: APIResponseStatus
    report_id: str
    file_path: str
    metadata: ReportMetadataRequest
    generation_time: float
    file_size: int
    findings_count: int
    download_url: str

class ReportListResponse(BaseModel):
    """Report list response model"""
    report_id: str
    title: str
    type: ReportTypeEnum
    timestamp: datetime
    findings_count: int
    author: str
    file_path: str
    file_size: int

class ReportingStatusResponse(BaseModel):
    """Reporting module status response"""
    module: str
    status: str
    version: str
    reports_generated: int
    supported_formats: List[str]
    supported_types: List[str]
    compliance_frameworks: List[str]

# Dependency to get reporting module
async def get_reporting_module():
    """Get reporting module instance"""
    module = get_module("reporting")
    if not module:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reporting module not available"
        )
    return module

# Endpoints
@router.get("/status", response_model=ReportingStatusResponse, responses=STANDARD_RESPONSES)
async def get_reporting_status(module=Depends(get_reporting_module)):
    """Get reporting module status and statistics"""
    try:
        status_info = await module.get_status()
        return ReportingStatusResponse(**status_info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )

@router.post("/generate", response_model=ReportGenerationResponse, responses=STANDARD_RESPONSES)
async def generate_report(
    report_request: ReportGenerationRequest,
    module=Depends(get_reporting_module)
):
    """Generate a security report"""
    try:
        # Create report metadata
        report_metadata = ReportMetadata(
            report_id=f"report_{int(datetime.utcnow().timestamp())}",
            title=report_request.metadata.title,
            description=report_request.metadata.description,
            report_type=ReportType(report_request.metadata.report_type.value),
            target_audience=report_request.metadata.target_audience,
            classification=report_request.metadata.classification,
            author=report_request.metadata.author,
            organization=report_request.metadata.organization,
            executive_summary=report_request.metadata.executive_summary,
            include_charts=report_request.metadata.include_charts,
            include_raw_data=report_request.metadata.include_raw_data,
            compliance_framework=ComplianceFramework(report_request.metadata.compliance_framework.value) if report_request.metadata.compliance_framework else None
        )
        
        # Collect raw data from specified sources
        raw_data = await collect_raw_data(report_request.data_sources, report_request.filters)
        
        # Generate report
        start_time = datetime.utcnow()
        file_path = await module.generate_report(
            report_metadata,
            raw_data,
            ReportFormat(report_request.output_format.value)
        )
        generation_time = (datetime.utcnow() - start_time).total_seconds()
        
        if not file_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Report generation failed"
            )
        
        # Get file size
        import os
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        # Count findings
        findings_count = len(raw_data.get("findings", []))
        
        # Create download URL
        download_url = f"/api/v1/reporting/download/{report_metadata.report_id}"
        
        return ReportGenerationResponse(
            status=APIResponseStatus.COMPLETED,
            report_id=report_metadata.report_id,
            file_path=file_path,
            metadata=report_request.metadata,
            generation_time=generation_time,
            file_size=file_size,
            findings_count=findings_count,
            download_url=download_url
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )

@router.get("/reports", response_model=List[ReportListResponse], responses=STANDARD_RESPONSES)
async def list_reports(
    report_type: Optional[ReportTypeEnum] = Query(None, description="Filter by report type"),
    author: Optional[str] = Query(None, description="Filter by author"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum reports to return"),
    module=Depends(get_reporting_module)
):
    """List generated reports"""
    try:
        reports = module.list_generated_reports()
        
        # Apply filters
        if report_type:
            reports = [r for r in reports if r["type"] == report_type.value]
        
        if author:
            reports = [r for r in reports if author.lower() in r["author"].lower()]
        
        # Limit results
        if limit:
            reports = reports[:limit]
        
        # Convert to response format
        report_responses = []
        for report in reports:
            report_responses.append(ReportListResponse(
                report_id=report["report_id"],
                title=report["title"],
                type=ReportTypeEnum(report["type"]),
                timestamp=datetime.fromisoformat(report["timestamp"]),
                findings_count=report["findings_count"],
                author=report["author"],
                file_path="",  # Don't expose file path in list
                file_size=0  # Would need to be calculated
            ))
        
        return report_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list reports: {str(e)}"
        )

@router.get("/download/{report_id}", responses=STANDARD_RESPONSES)
async def download_report(
    report_id: str = Path(..., description="Report ID to download"),
    module=Depends(get_reporting_module)
):
    """Download a generated report"""
    try:
        # Find report in generated reports
        reports = module.list_generated_reports()
        report = next((r for r in reports if r["report_id"] == report_id), None)
        
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report {report_id} not found"
            )
        
        # For now, return a placeholder since we don't have actual file paths
        # In a real implementation, this would return FileResponse(file_path)
        return {"message": f"Report {report_id} download would start here"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Download failed: {str(e)}"
        )

@router.get("/templates", responses=STANDARD_RESPONSES)
async def get_report_templates():
    """Get available report templates"""
    return {
        "templates": [
            {
                "name": "Executive Summary",
                "type": "executive_summary",
                "description": "High-level security assessment summary for executives",
                "target_audience": "Executive Leadership",
                "estimated_pages": "2-5"
            },
            {
                "name": "Technical Vulnerability Report",
                "type": "vulnerability_assessment",
                "description": "Detailed technical vulnerability assessment report",
                "target_audience": "Technical Teams",
                "estimated_pages": "10-50"
            },
            {
                "name": "Penetration Test Report",
                "type": "penetration_test",
                "description": "Comprehensive penetration testing report",
                "target_audience": "Security Teams",
                "estimated_pages": "20-100"
            },
            {
                "name": "Compliance Audit Report",
                "type": "compliance_audit",
                "description": "Security compliance assessment report",
                "target_audience": "Compliance Teams",
                "estimated_pages": "15-40"
            },
            {
                "name": "Threat Intelligence Report",
                "type": "threat_intelligence",
                "description": "Threat intelligence analysis and recommendations",
                "target_audience": "Security Operations",
                "estimated_pages": "5-20"
            }
        ]
    }

@router.get("/formats", responses=STANDARD_RESPONSES)
async def get_report_formats():
    """Get available report formats"""
    return {
        "formats": [
            {
                "name": "pdf",
                "description": "Portable Document Format",
                "file_extension": ".pdf",
                "supports_charts": True,
                "supports_formatting": True
            },
            {
                "name": "html",
                "description": "HyperText Markup Language",
                "file_extension": ".html",
                "supports_charts": True,
                "supports_formatting": True
            },
            {
                "name": "json",
                "description": "JavaScript Object Notation",
                "file_extension": ".json",
                "supports_charts": False,
                "supports_formatting": False
            },
            {
                "name": "csv",
                "description": "Comma-Separated Values",
                "file_extension": ".csv",
                "supports_charts": False,
                "supports_formatting": False
            },
            {
                "name": "markdown",
                "description": "Markdown Format",
                "file_extension": ".md",
                "supports_charts": False,
                "supports_formatting": True
            }
        ]
    }

@router.get("/compliance-frameworks", responses=STANDARD_RESPONSES)
async def get_compliance_frameworks():
    """Get available compliance frameworks"""
    return {
        "frameworks": [
            {
                "name": "nist_csf",
                "description": "NIST Cybersecurity Framework",
                "categories": ["Identify", "Protect", "Detect", "Respond", "Recover"],
                "version": "1.1"
            },
            {
                "name": "iso_27001",
                "description": "ISO/IEC 27001 Information Security Management",
                "categories": ["Information Security Policies", "Access Control", "Cryptography"],
                "version": "2013"
            },
            {
                "name": "pci_dss",
                "description": "Payment Card Industry Data Security Standard",
                "categories": ["Network Security", "Data Protection", "Vulnerability Management"],
                "version": "4.0"
            },
            {
                "name": "owasp_top10",
                "description": "OWASP Top 10 Web Application Security Risks",
                "categories": ["Injection", "Authentication", "Sensitive Data Exposure"],
                "version": "2021"
            }
        ]
    }

# Helper function to collect raw data
async def collect_raw_data(data_sources: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
    """Collect raw data from specified sources"""
    raw_data = {
        "findings": [],
        "statistics": {},
        "metadata": {
            "collection_timestamp": datetime.utcnow().isoformat(),
            "sources": data_sources,
            "filters": filters
        }
    }
    
    # In a real implementation, this would collect data from the specified modules
    # For now, return placeholder data
    
    if "scanning" in data_sources:
        raw_data["vulnerability_scan"] = {
            "vulnerabilities": [
                {
                    "name": "Example Vulnerability",
                    "description": "This is an example vulnerability for demonstration",
                    "severity": "medium",
                    "cvss_score": 5.0,
                    "affected_service": "Web Server",
                    "evidence": {"example": "evidence"},
                    "remediation": "Apply security patches"
                }
            ]
        }
    
    if "reconnaissance" in data_sources:
        raw_data["reconnaissance"] = {
            "results": [
                {
                    "target": "example.com",
                    "technique": "dns_lookup",
                    "data": {"ip": "192.168.1.1"},
                    "confidence": 0.9
                }
            ]
        }
    
    return raw_data