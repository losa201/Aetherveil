"""
Reporting Module for Aetherveil Sentinel

Comprehensive report generation and analysis capabilities for security operations.
Includes automated report creation, data visualization, executive summaries,
and compliance reporting for various security frameworks.

Security Level: DEFENSIVE_ONLY
"""

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
import hashlib

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ..config.config import AetherVeilConfig
from . import ModuleType, ModuleStatus, register_module

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of security reports"""
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_DETAILED = "technical_detailed"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    PENETRATION_TEST = "penetration_test"
    COMPLIANCE_AUDIT = "compliance_audit"
    THREAT_INTELLIGENCE = "threat_intelligence"
    INCIDENT_RESPONSE = "incident_response"
    RISK_ASSESSMENT = "risk_assessment"

class ReportFormat(Enum):
    """Report output formats"""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    MARKDOWN = "markdown"

class SeverityLevel(Enum):
    """Severity levels for findings"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    NIST_CSF = "nist_csf"
    ISO_27001 = "iso_27001"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOX = "sox"
    GDPR = "gdpr"
    OWASP_TOP10 = "owasp_top10"

@dataclass
class Finding:
    """Individual security finding"""
    finding_id: str
    title: str
    description: str
    severity: SeverityLevel
    cvss_score: Optional[float]
    cve_id: Optional[str]
    affected_assets: List[str]
    evidence: Dict[str, Any]
    remediation: str
    references: List[str] = field(default_factory=list)
    compliance_mappings: Dict[str, List[str]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    verified: bool = False

@dataclass
class ReportMetadata:
    """Report metadata and configuration"""
    report_id: str
    title: str
    description: str
    report_type: ReportType
    target_audience: str
    classification: str
    author: str
    organization: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    executive_summary: bool = True
    include_charts: bool = True
    include_raw_data: bool = False
    compliance_framework: Optional[ComplianceFramework] = None

@dataclass
class ReportData:
    """Complete report data structure"""
    metadata: ReportMetadata
    findings: List[Finding]
    statistics: Dict[str, Any]
    recommendations: List[str]
    appendices: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)

class DataAnalyzer:
    """Analyzes security data and generates insights"""
    
    def __init__(self):
        self.analysis_cache = {}
        
    def analyze_vulnerability_data(self, vulnerabilities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze vulnerability data for reporting insights"""
        analysis = {
            "total_vulnerabilities": len(vulnerabilities),
            "severity_breakdown": {},
            "top_vulnerability_types": {},
            "affected_services": {},
            "remediation_priority": [],
            "trend_analysis": {},
            "risk_metrics": {}
        }
        
        if not vulnerabilities:
            return analysis
        
        # Severity breakdown
        severity_counts = {}
        for vuln in vulnerabilities:
            severity = vuln.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        analysis["severity_breakdown"] = severity_counts
        
        # Top vulnerability types
        vuln_types = {}
        for vuln in vulnerabilities:
            vuln_type = vuln.get("name", "Unknown")
            vuln_types[vuln_type] = vuln_types.get(vuln_type, 0) + 1
        analysis["top_vulnerability_types"] = dict(sorted(vuln_types.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Affected services
        services = {}
        for vuln in vulnerabilities:
            service = vuln.get("affected_service", "Unknown")
            services[service] = services.get(service, 0) + 1
        analysis["affected_services"] = services
        
        # Risk calculation
        risk_score = self._calculate_risk_score(vulnerabilities)
        analysis["risk_metrics"] = {
            "overall_risk_score": risk_score,
            "critical_risk_items": [v for v in vulnerabilities if v.get("severity") == "critical"],
            "immediate_action_required": len([v for v in vulnerabilities if v.get("severity") in ["critical", "high"]])
        }
        
        return analysis
    
    def analyze_reconnaissance_data(self, recon_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze reconnaissance data"""
        analysis = {
            "total_discoveries": len(recon_data),
            "discovery_types": {},
            "information_exposure": {},
            "attack_surface": {},
            "intelligence_confidence": 0.0
        }
        
        if not recon_data:
            return analysis
        
        # Discovery types
        for item in recon_data:
            technique = item.get("technique", "unknown")
            analysis["discovery_types"][technique] = analysis["discovery_types"].get(technique, 0) + 1
        
        # Information exposure assessment
        sensitive_indicators = ["email", "phone", "credential", "internal", "admin"]
        exposure_count = 0
        for item in recon_data:
            data_value = str(item.get("data", {})).lower()
            if any(indicator in data_value for indicator in sensitive_indicators):
                exposure_count += 1
        
        analysis["information_exposure"] = {
            "potentially_sensitive_items": exposure_count,
            "exposure_percentage": (exposure_count / len(recon_data)) * 100
        }
        
        return analysis
    
    def analyze_stealth_effectiveness(self, stealth_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stealth technique effectiveness"""
        analysis = {
            "techniques_used": len(stealth_data),
            "average_detection_probability": 0.0,
            "stealth_rating": 0.0,
            "technique_effectiveness": {},
            "recommendations": []
        }
        
        if not stealth_data:
            return analysis
        
        detection_probs = []
        for item in stealth_data:
            detection_prob = item.get("detection_probability", 1.0)
            detection_probs.append(detection_prob)
            
            technique = item.get("technique", "unknown")
            if technique not in analysis["technique_effectiveness"]:
                analysis["technique_effectiveness"][technique] = {
                    "uses": 0,
                    "avg_detection_prob": 0.0,
                    "success_rate": 0.0
                }
            
            analysis["technique_effectiveness"][technique]["uses"] += 1
            analysis["technique_effectiveness"][technique]["avg_detection_prob"] += detection_prob
        
        # Calculate averages
        if detection_probs:
            analysis["average_detection_probability"] = sum(detection_probs) / len(detection_probs)
            analysis["stealth_rating"] = 1.0 - analysis["average_detection_probability"]
        
        # Finalize technique effectiveness
        for technique in analysis["technique_effectiveness"]:
            data = analysis["technique_effectiveness"][technique]
            data["avg_detection_prob"] /= data["uses"]
            data["success_rate"] = 1.0 - data["avg_detection_prob"]
        
        return analysis
    
    def _calculate_risk_score(self, vulnerabilities: List[Dict[str, Any]]) -> float:
        """Calculate overall risk score from vulnerabilities"""
        if not vulnerabilities:
            return 0.0
        
        severity_weights = {
            "critical": 10.0,
            "high": 7.5,
            "medium": 5.0,
            "low": 2.5,
            "info": 1.0
        }
        
        total_score = 0.0
        for vuln in vulnerabilities:
            severity = vuln.get("severity", "low")
            weight = severity_weights.get(severity, 1.0)
            
            # Factor in CVSS score if available
            cvss = vuln.get("cvss_score", 0.0)
            if cvss:
                weight = weight * (cvss / 10.0)
            
            total_score += weight
        
        # Normalize to 0-10 scale
        max_possible = len(vulnerabilities) * 10.0
        normalized_score = (total_score / max_possible) * 10.0 if max_possible > 0 else 0.0
        
        return min(10.0, normalized_score)

class ChartGenerator:
    """Generates charts and visualizations for reports"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        
    def create_severity_pie_chart(self, severity_data: Dict[str, int], output_path: str) -> str:
        """Create pie chart for vulnerability severity distribution"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            colors = {
                'critical': '#d32f2f',
                'high': '#f57c00',
                'medium': '#fbc02d',
                'low': '#388e3c',
                'info': '#1976d2'
            }
            
            labels = list(severity_data.keys())
            sizes = list(severity_data.values())
            chart_colors = [colors.get(label.lower(), '#757575') for label in labels]
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=chart_colors,
                                            autopct='%1.1f%%', startangle=90)
            
            ax.set_title('Vulnerability Severity Distribution', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create pie chart: {e}")
            return ""
    
    def create_vulnerability_trend_chart(self, trend_data: Dict[str, int], output_path: str) -> str:
        """Create trend chart for vulnerability discovery over time"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            dates = list(trend_data.keys())
            counts = list(trend_data.values())
            
            ax.plot(dates, counts, marker='o', linewidth=2, markersize=6)
            ax.set_title('Vulnerability Discovery Trend', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Number of Vulnerabilities')
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create trend chart: {e}")
            return ""
    
    def create_risk_heatmap(self, risk_data: Dict[str, Dict[str, float]], output_path: str) -> str:
        """Create risk heatmap visualization"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Convert to DataFrame for seaborn
            df = pd.DataFrame(risk_data)
            
            sns.heatmap(df, annot=True, cmap='RdYlBu_r', center=5.0, 
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            
            ax.set_title('Security Risk Heatmap', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create heatmap: {e}")
            return ""

class PDFReportGenerator:
    """Generates PDF reports using ReportLab"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
        
    def _create_custom_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles"""
        styles = {}
        
        styles['CustomTitle'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        
        styles['SectionHeader'] = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=12,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=5
        )
        
        styles['FindingTitle'] = ParagraphStyle(
            'FindingTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=6,
            textColor=colors.darkred
        )
        
        styles['ExecutivePara'] = ParagraphStyle(
            'ExecutivePara',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            leftIndent=20,
            rightIndent=20
        )
        
        return styles
    
    def generate_pdf_report(self, report_data: ReportData, output_path: str) -> str:
        """Generate complete PDF report"""
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            story = []
            
            # Title page
            story.extend(self._create_title_page(report_data.metadata))
            story.append(PageBreak())
            
            # Executive summary
            if report_data.metadata.executive_summary:
                story.extend(self._create_executive_summary(report_data))
                story.append(PageBreak())
            
            # Table of contents (placeholder)
            story.extend(self._create_table_of_contents(report_data))
            story.append(PageBreak())
            
            # Findings section
            story.extend(self._create_findings_section(report_data.findings))
            
            # Statistics and charts
            if report_data.metadata.include_charts and report_data.statistics:
                story.append(PageBreak())
                story.extend(self._create_statistics_section(report_data.statistics))
            
            # Recommendations
            if report_data.recommendations:
                story.append(PageBreak())
                story.extend(self._create_recommendations_section(report_data.recommendations))
            
            # Appendices
            if report_data.appendices:
                story.append(PageBreak())
                story.extend(self._create_appendices_section(report_data.appendices))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return ""
    
    def _create_title_page(self, metadata: ReportMetadata) -> List[Any]:
        """Create report title page"""
        story = []
        
        # Title
        story.append(Paragraph(metadata.title, self.custom_styles['CustomTitle']))
        story.append(Spacer(1, 30))
        
        # Metadata table
        metadata_data = [
            ['Report Type:', metadata.report_type.value.replace('_', ' ').title()],
            ['Author:', metadata.author],
            ['Organization:', metadata.organization],
            ['Date:', metadata.timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            ['Classification:', metadata.classification],
            ['Target Audience:', metadata.target_audience]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey)
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 50))
        
        # Description
        if metadata.description:
            story.append(Paragraph('<b>Description:</b>', self.styles['Heading2']))
            story.append(Paragraph(metadata.description, self.styles['Normal']))
        
        return story
    
    def _create_executive_summary(self, report_data: ReportData) -> List[Any]:
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph('Executive Summary', self.custom_styles['SectionHeader']))
        
        # Key statistics
        total_findings = len(report_data.findings)
        critical_findings = len([f for f in report_data.findings if f.severity == SeverityLevel.CRITICAL])
        high_findings = len([f for f in report_data.findings if f.severity == SeverityLevel.HIGH])
        
        summary_text = f"""
        This security assessment identified {total_findings} total findings, including 
        {critical_findings} critical and {high_findings} high severity issues that require 
        immediate attention. The assessment covered multiple security domains and provides 
        actionable recommendations for improving the overall security posture.
        """
        
        story.append(Paragraph(summary_text, self.custom_styles['ExecutivePara']))
        
        # Key findings summary
        if critical_findings > 0:
            story.append(Paragraph('<b>Critical Issues Requiring Immediate Action:</b>', self.styles['Heading3']))
            critical_items = [f for f in report_data.findings if f.severity == SeverityLevel.CRITICAL][:5]
            for finding in critical_items:
                story.append(Paragraph(f"• {finding.title}", self.styles['Normal']))
        
        return story
    
    def _create_table_of_contents(self, report_data: ReportData) -> List[Any]:
        """Create table of contents"""
        story = []
        
        story.append(Paragraph('Table of Contents', self.custom_styles['SectionHeader']))
        
        toc_data = [
            ['Section', 'Page'],
            ['Executive Summary', '3'],
            ['Findings Summary', '4'],
            ['Detailed Findings', '5'],
            ['Statistics and Analysis', str(5 + len(report_data.findings))],
            ['Recommendations', str(6 + len(report_data.findings))],
            ['Appendices', str(7 + len(report_data.findings))]
        ]
        
        toc_table = Table(toc_data, colWidths=[5*inch, 1*inch])
        toc_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)
        ]))
        
        story.append(toc_table)
        
        return story
    
    def _create_findings_section(self, findings: List[Finding]) -> List[Any]:
        """Create detailed findings section"""
        story = []
        
        story.append(Paragraph('Detailed Findings', self.custom_styles['SectionHeader']))
        
        # Group findings by severity
        severity_groups = {}
        for finding in findings:
            severity = finding.severity.value
            if severity not in severity_groups:
                severity_groups[severity] = []
            severity_groups[severity].append(finding)
        
        # Order by severity
        severity_order = ['critical', 'high', 'medium', 'low', 'info']
        
        for severity in severity_order:
            if severity in severity_groups:
                story.append(Paragraph(f'{severity.title()} Severity Findings', self.styles['Heading2']))
                
                for finding in severity_groups[severity]:
                    story.extend(self._create_finding_detail(finding))
                    story.append(Spacer(1, 12))
        
        return story
    
    def _create_finding_detail(self, finding: Finding) -> List[Any]:
        """Create detailed finding entry"""
        story = []
        
        # Finding title with severity color
        severity_colors = {
            'critical': colors.red,
            'high': colors.orange,
            'medium': colors.yellow,
            'low': colors.green,
            'info': colors.blue
        }
        
        color = severity_colors.get(finding.severity.value, colors.black)
        title_style = ParagraphStyle(
            'FindingTitleColored',
            parent=self.custom_styles['FindingTitle'],
            textColor=color
        )
        
        story.append(Paragraph(f"{finding.title} ({finding.severity.value.upper()})", title_style))
        
        # Finding details table
        finding_data = [
            ['Finding ID:', finding.finding_id],
            ['Severity:', finding.severity.value.upper()],
            ['CVSS Score:', str(finding.cvss_score) if finding.cvss_score else 'N/A'],
            ['CVE ID:', finding.cve_id if finding.cve_id else 'N/A'],
            ['Affected Assets:', ', '.join(finding.affected_assets)],
            ['Verified:', 'Yes' if finding.verified else 'No']
        ]
        
        details_table = Table(finding_data, colWidths=[1.5*inch, 4.5*inch])
        details_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey)
        ]))
        
        story.append(details_table)
        story.append(Spacer(1, 6))
        
        # Description
        story.append(Paragraph('<b>Description:</b>', self.styles['Heading4']))
        story.append(Paragraph(finding.description, self.styles['Normal']))
        story.append(Spacer(1, 6))
        
        # Remediation
        story.append(Paragraph('<b>Remediation:</b>', self.styles['Heading4']))
        story.append(Paragraph(finding.remediation, self.styles['Normal']))
        
        return story
    
    def _create_statistics_section(self, statistics: Dict[str, Any]) -> List[Any]:
        """Create statistics and analysis section"""
        story = []
        
        story.append(Paragraph('Statistics and Analysis', self.custom_styles['SectionHeader']))
        
        # Summary statistics table
        if 'vulnerability_analysis' in statistics:
            vuln_stats = statistics['vulnerability_analysis']
            
            stats_data = [
                ['Metric', 'Value'],
                ['Total Vulnerabilities', str(vuln_stats.get('total_vulnerabilities', 0))],
                ['Critical Findings', str(vuln_stats.get('severity_breakdown', {}).get('critical', 0))],
                ['High Findings', str(vuln_stats.get('severity_breakdown', {}).get('high', 0))],
                ['Overall Risk Score', f"{vuln_stats.get('risk_metrics', {}).get('overall_risk_score', 0):.1f}/10"]
            ]
            
            stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
            stats_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)
            ]))
            
            story.append(stats_table)
        
        return story
    
    def _create_recommendations_section(self, recommendations: List[str]) -> List[Any]:
        """Create recommendations section"""
        story = []
        
        story.append(Paragraph('Recommendations', self.custom_styles['SectionHeader']))
        
        for i, recommendation in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {recommendation}", self.styles['Normal']))
            story.append(Spacer(1, 6))
        
        return story
    
    def _create_appendices_section(self, appendices: Dict[str, Any]) -> List[Any]:
        """Create appendices section"""
        story = []
        
        story.append(Paragraph('Appendices', self.custom_styles['SectionHeader']))
        
        for title, content in appendices.items():
            story.append(Paragraph(f"Appendix: {title}", self.styles['Heading2']))
            
            if isinstance(content, str):
                story.append(Paragraph(content, self.styles['Normal']))
            elif isinstance(content, dict):
                story.append(Paragraph(json.dumps(content, indent=2), self.styles['Code']))
            
            story.append(Spacer(1, 12))
        
        return story

class ComplianceMapper:
    """Maps findings to compliance frameworks"""
    
    def __init__(self):
        self.framework_mappings = self._load_framework_mappings()
        
    def _load_framework_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """Load compliance framework mappings"""
        return {
            "nist_csf": {
                "authentication": ["PR.AC-1", "PR.AC-4", "PR.AC-7"],
                "encryption": ["PR.DS-1", "PR.DS-2"],
                "vulnerability_management": ["DE.CM-8", "RS.MI-3"],
                "incident_response": ["RS.RP-1", "RS.CO-2"],
                "logging": ["DE.AE-3", "DE.CM-1"]
            },
            "owasp_top10": {
                "injection": ["A03:2021"],
                "authentication": ["A07:2021"],
                "sensitive_data": ["A02:2021"],
                "xxe": ["A05:2021"],
                "broken_access": ["A01:2021"],
                "security_misconfig": ["A05:2021"],
                "xss": ["A03:2021"],
                "deserialization": ["A08:2021"],
                "components": ["A06:2021"],
                "logging": ["A09:2021"]
            },
            "iso_27001": {
                "access_control": ["A.9.1.1", "A.9.2.1"],
                "cryptography": ["A.10.1.1", "A.10.1.2"],
                "physical_security": ["A.11.1.1", "A.11.2.1"],
                "operations_security": ["A.12.1.1", "A.12.6.1"],
                "incident_management": ["A.16.1.1", "A.16.1.2"]
            }
        }
    
    def map_finding_to_frameworks(self, finding: Finding) -> Dict[str, List[str]]:
        """Map a finding to relevant compliance framework controls"""
        mappings = {}
        
        # Analyze finding title and description for keywords
        text = f"{finding.title} {finding.description}".lower()
        
        for framework, categories in self.framework_mappings.items():
            framework_controls = []
            
            for category, controls in categories.items():
                if category in text:
                    framework_controls.extend(controls)
            
            if framework_controls:
                mappings[framework] = list(set(framework_controls))  # Remove duplicates
        
        return mappings

class ReportingModule:
    """Main reporting module orchestrator"""
    
    def __init__(self, config: AetherVeilConfig):
        self.config = config
        self.module_type = ModuleType.REPORTING
        self.status = ModuleStatus.INITIALIZED
        self.version = "1.0.0"
        
        # Initialize components
        self.data_analyzer = DataAnalyzer()
        self.chart_generator = ChartGenerator()
        self.pdf_generator = PDFReportGenerator()
        self.compliance_mapper = ComplianceMapper()
        
        # Storage
        self.generated_reports: List[ReportData] = []
        self.report_cache: Dict[str, str] = {}
        
        logger.info("Reporting module initialized")
        
    async def start(self) -> bool:
        """Start the reporting module"""
        try:
            self.status = ModuleStatus.RUNNING
            logger.info("Reporting module started")
            return True
        except Exception as e:
            self.status = ModuleStatus.ERROR
            logger.error(f"Failed to start reporting module: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the reporting module"""
        try:
            self.status = ModuleStatus.STOPPED
            logger.info("Reporting module stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop reporting module: {e}")
            return False
    
    async def generate_report(self, report_metadata: ReportMetadata, 
                            raw_data: Dict[str, Any], 
                            output_format: ReportFormat = ReportFormat.PDF) -> str:
        """Generate comprehensive security report"""
        
        try:
            logger.info(f"Generating {report_metadata.report_type.value} report: {report_metadata.title}")
            
            # Process raw data into findings
            findings = await self._process_raw_data_to_findings(raw_data)
            
            # Add compliance mappings
            for finding in findings:
                finding.compliance_mappings = self.compliance_mapper.map_finding_to_frameworks(finding)
            
            # Generate statistics and analysis
            statistics = await self._generate_statistics(raw_data, findings)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(findings, statistics)
            
            # Create report data structure
            report_data = ReportData(
                metadata=report_metadata,
                findings=findings,
                statistics=statistics,
                recommendations=recommendations,
                raw_data=raw_data if report_metadata.include_raw_data else {}
            )
            
            # Generate charts if requested
            if report_metadata.include_charts:
                charts = await self._generate_charts(statistics)
                report_data.appendices["charts"] = charts
            
            # Store report
            self.generated_reports.append(report_data)
            
            # Generate output in requested format
            output_path = await self._generate_output(report_data, output_format)
            
            logger.info(f"Report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return ""
    
    async def _process_raw_data_to_findings(self, raw_data: Dict[str, Any]) -> List[Finding]:
        """Process raw security data into structured findings"""
        findings = []
        
        # Process vulnerability scan results
        if "vulnerability_scan" in raw_data:
            vuln_data = raw_data["vulnerability_scan"]
            for vuln in vuln_data.get("vulnerabilities", []):
                finding = Finding(
                    finding_id=f"VULN-{hashlib.md5(str(vuln).encode()).hexdigest()[:8]}",
                    title=vuln.get("name", "Unknown Vulnerability"),
                    description=vuln.get("description", "No description available"),
                    severity=SeverityLevel(vuln.get("severity", "medium")),
                    cvss_score=vuln.get("cvss_score"),
                    cve_id=vuln.get("cve_id"),
                    affected_assets=[vuln.get("affected_service", "Unknown")],
                    evidence=vuln.get("evidence", {}),
                    remediation=vuln.get("remediation", "No remediation steps provided"),
                    references=vuln.get("references", [])
                )
                findings.append(finding)
        
        # Process exploitation results
        if "exploitation" in raw_data:
            exploit_data = raw_data["exploitation"]
            for exploit in exploit_data.get("successful_exploits", []):
                finding = Finding(
                    finding_id=f"EXPLOIT-{hashlib.md5(str(exploit).encode()).hexdigest()[:8]}",
                    title=f"Successful Exploitation: {exploit.get('exploitation_type', 'Unknown')}",
                    description=exploit.get('impact_assessment', 'Exploitation was successful'),
                    severity=SeverityLevel.CRITICAL,  # Successful exploits are always critical
                    cvss_score=9.0,  # High CVSS for successful exploits
                    cve_id=None,
                    affected_assets=[exploit.get("target", "Unknown")],
                    evidence=exploit.get("evidence", {}),
                    remediation="; ".join(exploit.get("remediation_steps", [])),
                    verified=True
                )
                findings.append(finding)
        
        # Process reconnaissance findings
        if "reconnaissance" in raw_data:
            recon_data = raw_data["reconnaissance"]
            sensitive_items = []
            
            for item in recon_data.get("results", []):
                # Check for sensitive information exposure
                data_str = str(item.get("data", {})).lower()
                if any(keyword in data_str for keyword in ["password", "secret", "key", "token"]):
                    sensitive_items.append(item)
            
            if sensitive_items:
                finding = Finding(
                    finding_id=f"RECON-{hashlib.md5(str(sensitive_items).encode()).hexdigest()[:8]}",
                    title="Sensitive Information Exposure",
                    description=f"Reconnaissance identified {len(sensitive_items)} instances of potentially sensitive information exposure",
                    severity=SeverityLevel.MEDIUM,
                    cvss_score=5.0,
                    cve_id=None,
                    affected_assets=[item.get("target", "Unknown") for item in sensitive_items],
                    evidence={"sensitive_items": sensitive_items[:5]},  # Limit evidence size
                    remediation="Review and secure exposed sensitive information"
                )
                findings.append(finding)
        
        return findings
    
    async def _generate_statistics(self, raw_data: Dict[str, Any], findings: List[Finding]) -> Dict[str, Any]:
        """Generate comprehensive statistics from data"""
        statistics = {}
        
        # Vulnerability analysis
        if findings:
            vuln_data = [asdict(f) for f in findings]
            statistics["vulnerability_analysis"] = self.data_analyzer.analyze_vulnerability_data(vuln_data)
        
        # Reconnaissance analysis
        if "reconnaissance" in raw_data:
            recon_results = raw_data["reconnaissance"].get("results", [])
            statistics["reconnaissance_analysis"] = self.data_analyzer.analyze_reconnaissance_data(recon_results)
        
        # Stealth analysis
        if "stealth" in raw_data:
            stealth_results = raw_data["stealth"].get("results", [])
            statistics["stealth_analysis"] = self.data_analyzer.analyze_stealth_effectiveness(stealth_results)
        
        # Overall assessment
        statistics["overall_assessment"] = {
            "total_findings": len(findings),
            "critical_findings": len([f for f in findings if f.severity == SeverityLevel.CRITICAL]),
            "high_findings": len([f for f in findings if f.severity == SeverityLevel.HIGH]),
            "verified_findings": len([f for f in findings if f.verified]),
            "risk_score": statistics.get("vulnerability_analysis", {}).get("risk_metrics", {}).get("overall_risk_score", 0.0)
        }
        
        return statistics
    
    async def _generate_recommendations(self, findings: List[Finding], 
                                      statistics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Critical findings recommendations
        critical_findings = [f for f in findings if f.severity == SeverityLevel.CRITICAL]
        if critical_findings:
            recommendations.append(
                f"Immediately address {len(critical_findings)} critical security findings "
                "that pose significant risk to the organization."
            )
        
        # Vulnerability management recommendations
        vuln_stats = statistics.get("vulnerability_analysis", {})
        total_vulns = vuln_stats.get("total_vulnerabilities", 0)
        if total_vulns > 0:
            recommendations.append(
                f"Implement a systematic vulnerability management program to address "
                f"{total_vulns} identified vulnerabilities."
            )
        
        # Stealth and detection recommendations
        stealth_stats = statistics.get("stealth_analysis", {})
        if stealth_stats.get("average_detection_probability", 1.0) < 0.5:
            recommendations.append(
                "Enhance detection capabilities as current security controls may not "
                "effectively detect advanced persistent threats."
            )
        
        # Compliance recommendations
        compliance_gaps = []
        for finding in findings:
            for framework in finding.compliance_mappings.keys():
                if framework not in compliance_gaps:
                    compliance_gaps.append(framework)
        
        if compliance_gaps:
            recommendations.append(
                f"Address compliance gaps in {', '.join(compliance_gaps)} frameworks "
                "to ensure regulatory compliance."
            )
        
        # General security posture recommendations
        recommendations.extend([
            "Implement regular security assessments to maintain visibility into the security posture.",
            "Establish incident response procedures for handling security events.",
            "Provide security awareness training to all personnel.",
            "Implement defense-in-depth security architecture.",
            "Establish continuous monitoring and threat detection capabilities."
        ])
        
        return recommendations
    
    async def _generate_charts(self, statistics: Dict[str, Any]) -> Dict[str, str]:
        """Generate charts and visualizations"""
        charts = {}
        
        try:
            # Create temporary directory for charts
            temp_dir = tempfile.mkdtemp()
            
            # Severity distribution pie chart
            vuln_stats = statistics.get("vulnerability_analysis", {})
            severity_data = vuln_stats.get("severity_breakdown", {})
            if severity_data:
                chart_path = os.path.join(temp_dir, "severity_distribution.png")
                created_chart = self.chart_generator.create_severity_pie_chart(severity_data, chart_path)
                if created_chart:
                    charts["severity_distribution"] = created_chart
            
            # Risk heatmap
            risk_data = {"Assets": {"Critical": 8.5, "High": 6.2, "Medium": 4.1, "Low": 2.3}}
            heatmap_path = os.path.join(temp_dir, "risk_heatmap.png")
            created_heatmap = self.chart_generator.create_risk_heatmap(risk_data, heatmap_path)
            if created_heatmap:
                charts["risk_heatmap"] = created_heatmap
            
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
        
        return charts
    
    async def _generate_output(self, report_data: ReportData, 
                             output_format: ReportFormat) -> str:
        """Generate report output in specified format"""
        
        # Create output directory
        output_dir = os.path.join(tempfile.gettempdir(), "aetherveil_reports")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{report_data.metadata.report_type.value}_{timestamp}"
        
        if output_format == ReportFormat.PDF:
            output_path = os.path.join(output_dir, f"{base_filename}.pdf")
            return self.pdf_generator.generate_pdf_report(report_data, output_path)
            
        elif output_format == ReportFormat.JSON:
            output_path = os.path.join(output_dir, f"{base_filename}.json")
            with open(output_path, 'w') as f:
                # Convert to serializable format
                report_dict = asdict(report_data)
                json.dump(report_dict, f, indent=2, default=str)
            return output_path
            
        elif output_format == ReportFormat.HTML:
            output_path = os.path.join(output_dir, f"{base_filename}.html")
            html_content = await self._generate_html_report(report_data)
            with open(output_path, 'w') as f:
                f.write(html_content)
            return output_path
            
        else:
            logger.error(f"Unsupported output format: {output_format}")
            return ""
    
    async def _generate_html_report(self, report_data: ReportData) -> str:
        """Generate HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .finding {{ margin: 20px 0; padding: 15px; border-left: 5px solid #ccc; }}
                .critical {{ border-left-color: #d32f2f; }}
                .high {{ border-left-color: #f57c00; }}
                .medium {{ border-left-color: #fbc02d; }}
                .low {{ border-left-color: #388e3c; }}
                .info {{ border-left-color: #1976d2; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p><strong>Type:</strong> {report_type}</p>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Author:</strong> {author}</p>
            </div>
            
            <h2>Executive Summary</h2>
            <p>This assessment identified {total_findings} security findings requiring attention.</p>
            
            <h2>Findings</h2>
            {findings_html}
            
            <h2>Recommendations</h2>
            <ul>
                {recommendations_html}
            </ul>
        </body>
        </html>
        """
        
        # Generate findings HTML
        findings_html = ""
        for finding in report_data.findings:
            severity_class = finding.severity.value
            findings_html += f"""
            <div class="finding {severity_class}">
                <h3>{finding.title} ({finding.severity.value.upper()})</h3>
                <p><strong>Description:</strong> {finding.description}</p>
                <p><strong>Affected Assets:</strong> {', '.join(finding.affected_assets)}</p>
                <p><strong>Remediation:</strong> {finding.remediation}</p>
            </div>
            """
        
        # Generate recommendations HTML
        recommendations_html = ""
        for rec in report_data.recommendations:
            recommendations_html += f"<li>{rec}</li>"
        
        return html_template.format(
            title=report_data.metadata.title,
            report_type=report_data.metadata.report_type.value,
            timestamp=report_data.metadata.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            author=report_data.metadata.author,
            total_findings=len(report_data.findings),
            findings_html=findings_html,
            recommendations_html=recommendations_html
        )
    
    def list_generated_reports(self) -> List[Dict[str, Any]]:
        """List all generated reports"""
        reports = []
        for report in self.generated_reports:
            reports.append({
                "report_id": report.metadata.report_id,
                "title": report.metadata.title,
                "type": report.metadata.report_type.value,
                "timestamp": report.metadata.timestamp.isoformat(),
                "findings_count": len(report.findings),
                "author": report.metadata.author
            })
        return reports
    
    async def get_status(self) -> Dict[str, Any]:
        """Get reporting module status"""
        return {
            "module": "reporting",
            "status": self.status.value,
            "version": self.version,
            "reports_generated": len(self.generated_reports),
            "supported_formats": [fmt.value for fmt in ReportFormat],
            "supported_types": [rt.value for rt in ReportType],
            "compliance_frameworks": list(self.compliance_mapper.framework_mappings.keys())
        }

# Register module on import
def create_reporting_module(config: AetherVeilConfig) -> ReportingModule:
    """Factory function to create and register reporting module"""
    module = ReportingModule(config)
    register_module("reporting", module)
    return module

__all__ = [
    "ReportingModule",
    "ReportMetadata",
    "ReportData",
    "Finding",
    "ReportType",
    "ReportFormat",
    "SeverityLevel",
    "ComplianceFramework",
    "create_reporting_module"
]