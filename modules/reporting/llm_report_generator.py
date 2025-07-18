"""
LLM-based Report Generation System
Advanced report generation using prompt engineering and multiple LLM providers
"""

import asyncio
import aiohttp
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import re
from jinja2 import Template, Environment, FileSystemLoader
from pathlib import Path
import uuid

from ...config.config import config
from ...coordinator.security_manager import security_manager

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """LLM providers for report generation"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class ReportType(Enum):
    """Types of reports that can be generated"""
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_ANALYSIS = "technical_analysis"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    PENETRATION_TEST = "penetration_test"
    THREAT_INTELLIGENCE = "threat_intelligence"
    INCIDENT_RESPONSE = "incident_response"
    COMPLIANCE_AUDIT = "compliance_audit"
    RISK_ASSESSMENT = "risk_assessment"
    FORENSIC_ANALYSIS = "forensic_analysis"


class ReportSeverity(Enum):
    """Report severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ReportData:
    """Structured data for report generation"""
    target_info: Dict[str, Any]
    findings: List[Dict[str, Any]]
    vulnerabilities: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    threat_intelligence: Dict[str, Any]
    attack_timeline: List[Dict[str, Any]]
    risk_score: float
    severity: ReportSeverity
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "target_info": self.target_info,
            "findings": self.findings,
            "vulnerabilities": self.vulnerabilities,
            "recommendations": self.recommendations,
            "threat_intelligence": self.threat_intelligence,
            "attack_timeline": self.attack_timeline,
            "risk_score": self.risk_score,
            "severity": self.severity.value,
            "metadata": self.metadata
        }


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: LLMProvider
    model: str
    api_key: str
    base_url: str
    max_tokens: int
    temperature: float
    timeout: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout
        }


class PromptTemplate:
    """Advanced prompt template with context injection"""
    
    def __init__(self, template_name: str, template_content: str, 
                 context_fields: List[str], output_format: str = "markdown"):
        self.template_name = template_name
        self.template_content = template_content
        self.context_fields = context_fields
        self.output_format = output_format
        self.jinja_template = Template(template_content)
    
    def render(self, context: Dict[str, Any]) -> str:
        """Render template with context"""
        # Validate required fields
        missing_fields = [field for field in self.context_fields 
                         if field not in context]
        if missing_fields:
            raise ValueError(f"Missing required context fields: {missing_fields}")
        
        return self.jinja_template.render(**context)


class LLMReportGenerator:
    """Advanced LLM-based report generator with multiple providers and prompt engineering"""
    
    def __init__(self):
        self.session = None
        self.llm_configs = {}
        self.prompt_templates = {}
        self.report_cache = {}
        self.generation_metrics = {
            "total_reports": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_generation_time": 0.0
        }
        
        # Initialize LLM configurations
        self._initialize_llm_configs()
        
        # Initialize prompt templates
        self._initialize_prompt_templates()
    
    def _initialize_llm_configs(self):
        """Initialize LLM provider configurations"""
        self.llm_configs = {
            LLMProvider.OPENAI: LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                api_key=security_manager.get_secret("OPENAI_API_KEY"),
                base_url="https://api.openai.com/v1",
                max_tokens=4096,
                temperature=0.3,
                timeout=60
            ),
            LLMProvider.ANTHROPIC: LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-opus-20240229",
                api_key=security_manager.get_secret("ANTHROPIC_API_KEY"),
                base_url="https://api.anthropic.com/v1",
                max_tokens=4096,
                temperature=0.3,
                timeout=60
            ),
            LLMProvider.GOOGLE: LLMConfig(
                provider=LLMProvider.GOOGLE,
                model="gemini-pro",
                api_key=security_manager.get_secret("GOOGLE_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1",
                max_tokens=4096,
                temperature=0.3,
                timeout=60
            ),
            LLMProvider.GROQ: LLMConfig(
                provider=LLMProvider.GROQ,
                model="mixtral-8x7b-32768",
                api_key=security_manager.get_secret("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
                max_tokens=4096,
                temperature=0.3,
                timeout=60
            ),
            LLMProvider.OLLAMA: LLMConfig(
                provider=LLMProvider.OLLAMA,
                model="llama2",
                api_key="",  # Local inference
                base_url="http://localhost:11434/v1",
                max_tokens=4096,
                temperature=0.3,
                timeout=60
            )
        }
    
    def _initialize_prompt_templates(self):
        """Initialize prompt templates for different report types"""
        
        # Executive Summary Template
        executive_summary_template = """
# Executive Summary Report

## Assessment Overview
You are a cybersecurity expert generating an executive summary based on penetration testing results.

**Target Information:**
- Organization: {{ target_info.organization }}
- Scope: {{ target_info.scope }}
- Assessment Period: {{ target_info.start_date }} to {{ target_info.end_date }}
- Methodology: {{ target_info.methodology }}

## Key Findings Summary
{% for finding in findings %}
- **{{ finding.title }}** ({{ finding.severity }}): {{ finding.description }}
{% endfor %}

## Risk Assessment
- Overall Risk Score: {{ risk_score }}/10
- Risk Level: {{ severity.upper() }}
- Critical Vulnerabilities: {{ vulnerabilities|selectattr("severity", "equalto", "critical")|list|length }}
- High Vulnerabilities: {{ vulnerabilities|selectattr("severity", "equalto", "high")|list|length }}

## Business Impact
Generate a business impact assessment considering:
1. Potential financial losses
2. Operational disruption
3. Regulatory compliance issues
4. Reputation damage

## Recommendations Summary
{% for recommendation in recommendations %}
- **{{ recommendation.title }}** (Priority: {{ recommendation.priority }}): {{ recommendation.description }}
{% endfor %}

Please provide a comprehensive executive summary that:
1. Explains technical findings in business terms
2. Prioritizes risks based on business impact
3. Provides clear, actionable recommendations
4. Includes timeline for remediation
5. Addresses regulatory and compliance concerns

Format the output as a professional executive summary suitable for C-level executives.
"""
        
        # Technical Analysis Template
        technical_analysis_template = """
# Technical Security Analysis Report

## Methodology and Scope
You are a senior penetration tester creating a detailed technical analysis report.

**Assessment Details:**
- Target: {{ target_info.target }}
- IP Range: {{ target_info.ip_range }}
- Domains: {{ target_info.domains }}
- Assessment Type: {{ target_info.assessment_type }}

## Technical Findings

### Vulnerability Analysis
{% for vuln in vulnerabilities %}
**{{ vuln.title }}** ({{ vuln.severity.upper() }})
- CVE: {{ vuln.cve_id }}
- CVSS Score: {{ vuln.cvss_score }}
- Affected Systems: {{ vuln.affected_systems|join(", ") }}
- Description: {{ vuln.description }}
- Exploitation Method: {{ vuln.exploitation_method }}
- Evidence: {{ vuln.evidence }}

{% endfor %}

### Attack Path Analysis
{% for step in attack_timeline %}
{{ loop.index }}. **{{ step.technique }}** ({{ step.timestamp }})
   - MITRE ATT&CK: {{ step.mitre_technique }}
   - Description: {{ step.description }}
   - Impact: {{ step.impact }}
{% endfor %}

### Network Analysis
{{ threat_intelligence.network_analysis }}

### Service Analysis
{{ threat_intelligence.service_analysis }}

## Technical Recommendations
{% for rec in recommendations %}
### {{ rec.title }}
- **Priority:** {{ rec.priority }}
- **Difficulty:** {{ rec.difficulty }}
- **Timeline:** {{ rec.timeline }}
- **Technical Details:** {{ rec.technical_details }}
- **Implementation Steps:** {{ rec.implementation_steps }}
{% endfor %}

Generate a comprehensive technical analysis that includes:
1. Detailed vulnerability explanations with technical context
2. Step-by-step attack path reconstruction
3. Network and service security analysis
4. Specific technical remediation steps
5. References to industry standards and frameworks

Format as a detailed technical report for security engineers and system administrators.
"""
        
        # Vulnerability Assessment Template
        vulnerability_assessment_template = """
# Vulnerability Assessment Report

## Assessment Summary
You are a vulnerability researcher creating a comprehensive vulnerability assessment report.

**Scan Results:**
- Total Vulnerabilities Found: {{ vulnerabilities|length }}
- Critical: {{ vulnerabilities|selectattr("severity", "equalto", "critical")|list|length }}
- High: {{ vulnerabilities|selectattr("severity", "equalto", "high")|list|length }}
- Medium: {{ vulnerabilities|selectattr("severity", "equalto", "medium")|list|length }}
- Low: {{ vulnerabilities|selectattr("severity", "equalto", "low")|list|length }}

## Vulnerability Details
{% for vuln in vulnerabilities %}
### {{ vuln.title }}
**Severity:** {{ vuln.severity.upper() }}
**CVSS Score:** {{ vuln.cvss_score }}
**CWE:** {{ vuln.cwe_id }}
**OWASP Top 10:** {{ vuln.owasp_category }}

**Description:**
{{ vuln.description }}

**Affected Assets:**
{% for asset in vuln.affected_assets %}
- {{ asset.hostname }} ({{ asset.ip }})
{% endfor %}

**Proof of Concept:**
{{ vuln.proof_of_concept }}

**Remediation:**
{{ vuln.remediation }}

**Risk Assessment:**
- Exploitability: {{ vuln.exploitability }}
- Impact: {{ vuln.impact }}
- Likelihood: {{ vuln.likelihood }}

---
{% endfor %}

## Risk Matrix
Generate a risk matrix showing vulnerability distribution by severity and asset criticality.

## Remediation Prioritization
Create a prioritized remediation plan considering:
1. Risk score (CVSS × Asset criticality)
2. Ease of exploitation
3. Business impact
4. Remediation complexity

## Compliance Mapping
Map findings to relevant compliance frameworks:
- NIST Cybersecurity Framework
- ISO 27001
- PCI DSS
- GDPR
- SOC 2

Generate a comprehensive vulnerability assessment report with detailed analysis and prioritized remediation guidance.
"""
        
        # Threat Intelligence Template
        threat_intelligence_template = """
# Threat Intelligence Report

## Intelligence Summary
You are a threat intelligence analyst creating a comprehensive threat intelligence report.

**Target Context:**
- Organization: {{ target_info.organization }}
- Industry: {{ target_info.industry }}
- Geographic Location: {{ target_info.location }}
- Digital Footprint: {{ target_info.digital_footprint }}

## Threat Landscape Analysis

### External Threat Intelligence
{{ threat_intelligence.external_intelligence }}

### APT Campaign Analysis
{% for campaign in threat_intelligence.apt_campaigns %}
**{{ campaign.name }}** ({{ campaign.attribution }})
- Active Since: {{ campaign.first_seen }}
- Targets: {{ campaign.targets }}
- TTPs: {{ campaign.ttps }}
- IoCs: {{ campaign.iocs }}
- Relevance Score: {{ campaign.relevance_score }}
{% endfor %}

### Indicator Analysis
{% for indicator in threat_intelligence.indicators %}
**{{ indicator.value }}** ({{ indicator.type }})
- Confidence: {{ indicator.confidence }}
- Threat Level: {{ indicator.threat_level }}
- Sources: {{ indicator.sources }}
- Context: {{ indicator.context }}
{% endfor %}

## Threat Actor Profiling
{% for actor in threat_intelligence.threat_actors %}
### {{ actor.name }}
- Motivation: {{ actor.motivation }}
- Sophistication: {{ actor.sophistication }}
- Targeting: {{ actor.targeting }}
- Capabilities: {{ actor.capabilities }}
- Recent Activity: {{ actor.recent_activity }}
{% endfor %}

## Industry Threat Trends
Analyze current threat trends specific to {{ target_info.industry }}:
1. Common attack vectors
2. Emerging threats
3. Regulatory changes
4. Incident patterns

## Attribution Analysis
Provide analysis on potential attribution based on:
- Technical indicators
- Behavioral patterns
- Geopolitical context
- Historical precedents

## Recommendations
{% for rec in recommendations %}
### {{ rec.title }}
- **Threat:** {{ rec.threat }}
- **Mitigation:** {{ rec.mitigation }}
- **Detection:** {{ rec.detection }}
- **Response:** {{ rec.response }}
{% endfor %}

Generate a comprehensive threat intelligence report that provides actionable insights for threat hunting and defense improvement.
"""
        
        # Create prompt templates
        self.prompt_templates = {
            ReportType.EXECUTIVE_SUMMARY: PromptTemplate(
                "executive_summary",
                executive_summary_template,
                ["target_info", "findings", "vulnerabilities", "recommendations", "risk_score", "severity"]
            ),
            ReportType.TECHNICAL_ANALYSIS: PromptTemplate(
                "technical_analysis",
                technical_analysis_template,
                ["target_info", "vulnerabilities", "attack_timeline", "threat_intelligence", "recommendations"]
            ),
            ReportType.VULNERABILITY_ASSESSMENT: PromptTemplate(
                "vulnerability_assessment",
                vulnerability_assessment_template,
                ["vulnerabilities", "target_info"]
            ),
            ReportType.THREAT_INTELLIGENCE: PromptTemplate(
                "threat_intelligence",
                threat_intelligence_template,
                ["target_info", "threat_intelligence", "recommendations"]
            )
        }
    
    async def initialize(self):
        """Initialize HTTP session"""
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=120, connect=30)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "AetherVeil-Sentinel-ReportGenerator/1.0"
            }
        )
        
        logger.info("LLM report generator initialized")
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def _make_llm_request(self, llm_config: LLMConfig, prompt: str, 
                               system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Make request to LLM provider"""
        if not self.session:
            await self.initialize()
        
        try:
            if llm_config.provider == LLMProvider.OPENAI:
                return await self._openai_request(llm_config, prompt, system_prompt)
            elif llm_config.provider == LLMProvider.ANTHROPIC:
                return await self._anthropic_request(llm_config, prompt, system_prompt)
            elif llm_config.provider == LLMProvider.GOOGLE:
                return await self._google_request(llm_config, prompt, system_prompt)
            elif llm_config.provider == LLMProvider.GROQ:
                return await self._groq_request(llm_config, prompt, system_prompt)
            elif llm_config.provider == LLMProvider.OLLAMA:
                return await self._ollama_request(llm_config, prompt, system_prompt)
            else:
                return {"error": f"Unsupported LLM provider: {llm_config.provider}"}
        
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return {"error": str(e)}
    
    async def _openai_request(self, llm_config: LLMConfig, prompt: str, 
                             system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Make OpenAI API request"""
        headers = {
            "Authorization": f"Bearer {llm_config.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": llm_config.model,
            "messages": messages,
            "max_tokens": llm_config.max_tokens,
            "temperature": llm_config.temperature
        }
        
        async with self.session.post(
            f"{llm_config.base_url}/chat/completions",
            headers=headers,
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "content": result["choices"][0]["message"]["content"],
                    "usage": result.get("usage", {}),
                    "model": llm_config.model
                }
            else:
                error_text = await response.text()
                return {"error": f"OpenAI API error: {response.status} - {error_text}"}
    
    async def _anthropic_request(self, llm_config: LLMConfig, prompt: str, 
                                system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Make Anthropic API request"""
        headers = {
            "x-api-key": llm_config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        data = {
            "model": llm_config.model,
            "max_tokens": llm_config.max_tokens,
            "temperature": llm_config.temperature,
            "messages": [{"role": "user", "content": full_prompt}]
        }
        
        async with self.session.post(
            f"{llm_config.base_url}/messages",
            headers=headers,
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "content": result["content"][0]["text"],
                    "usage": result.get("usage", {}),
                    "model": llm_config.model
                }
            else:
                error_text = await response.text()
                return {"error": f"Anthropic API error: {response.status} - {error_text}"}
    
    async def _google_request(self, llm_config: LLMConfig, prompt: str, 
                             system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Make Google Gemini API request"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        data = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": llm_config.max_tokens,
                "temperature": llm_config.temperature
            }
        }
        
        async with self.session.post(
            f"{llm_config.base_url}/models/{llm_config.model}:generateContent?key={llm_config.api_key}",
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "content": result["candidates"][0]["content"]["parts"][0]["text"],
                    "usage": result.get("usageMetadata", {}),
                    "model": llm_config.model
                }
            else:
                error_text = await response.text()
                return {"error": f"Google API error: {response.status} - {error_text}"}
    
    async def _groq_request(self, llm_config: LLMConfig, prompt: str, 
                           system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Make Groq API request"""
        headers = {
            "Authorization": f"Bearer {llm_config.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": llm_config.model,
            "messages": messages,
            "max_tokens": llm_config.max_tokens,
            "temperature": llm_config.temperature
        }
        
        async with self.session.post(
            f"{llm_config.base_url}/chat/completions",
            headers=headers,
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "content": result["choices"][0]["message"]["content"],
                    "usage": result.get("usage", {}),
                    "model": llm_config.model
                }
            else:
                error_text = await response.text()
                return {"error": f"Groq API error: {response.status} - {error_text}"}
    
    async def _ollama_request(self, llm_config: LLMConfig, prompt: str, 
                             system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Make Ollama API request"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        data = {
            "model": llm_config.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": llm_config.temperature,
                "num_predict": llm_config.max_tokens
            }
        }
        
        async with self.session.post(
            f"{llm_config.base_url}/api/generate",
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "content": result["response"],
                    "usage": {"total_tokens": result.get("eval_count", 0)},
                    "model": llm_config.model
                }
            else:
                error_text = await response.text()
                return {"error": f"Ollama API error: {response.status} - {error_text}"}
    
    async def generate_report(self, report_type: ReportType, report_data: ReportData, 
                             llm_provider: LLMProvider = LLMProvider.OPENAI,
                             custom_instructions: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive report using LLM"""
        start_time = time.time()
        
        try:
            # Get LLM configuration
            llm_config = self.llm_configs.get(llm_provider)
            if not llm_config or not llm_config.api_key:
                return {"error": f"LLM provider {llm_provider.value} not configured"}
            
            # Get prompt template
            prompt_template = self.prompt_templates.get(report_type)
            if not prompt_template:
                return {"error": f"Report type {report_type.value} not supported"}
            
            # Prepare context
            context = report_data.to_dict()
            context.update({
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "report_id": str(uuid.uuid4()),
                "report_type": report_type.value
            })
            
            # Render prompt
            try:
                rendered_prompt = prompt_template.render(context)
            except Exception as e:
                return {"error": f"Failed to render prompt template: {e}"}
            
            # Add custom instructions if provided
            if custom_instructions:
                rendered_prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"
            
            # System prompt for cybersecurity context
            system_prompt = f"""You are an expert cybersecurity consultant and report writer with 20+ years of experience. 
You specialize in creating professional, accurate, and actionable cybersecurity reports.

Guidelines:
1. Use professional language appropriate for your audience
2. Be specific and factual, avoiding generic statements
3. Include relevant industry standards and frameworks
4. Provide clear, actionable recommendations
5. Use proper cybersecurity terminology
6. Structure content logically with clear sections
7. Include risk ratings and prioritization
8. Reference MITRE ATT&CK techniques where applicable
9. Consider business impact in your analysis
10. Ensure accuracy and avoid false positives

Report Type: {report_type.value.replace('_', ' ').title()}
Output Format: Professional markdown suitable for executive and technical audiences
"""
            
            # Check cache
            cache_key = hashlib.md5(f"{report_type.value}:{rendered_prompt}".encode()).hexdigest()
            if cache_key in self.report_cache:
                cached_report, cache_time = self.report_cache[cache_key]
                if time.time() - cache_time < 3600:  # 1 hour cache
                    return cached_report
            
            # Make LLM request
            llm_response = await self._make_llm_request(llm_config, rendered_prompt, system_prompt)
            
            if "error" in llm_response:
                self.generation_metrics["failed_generations"] += 1
                return llm_response
            
            # Process and enhance the response
            enhanced_report = await self._enhance_report(llm_response["content"], report_type, context)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            self.generation_metrics["total_reports"] += 1
            self.generation_metrics["successful_generations"] += 1
            
            # Update average generation time
            total_reports = self.generation_metrics["total_reports"]
            current_avg = self.generation_metrics["avg_generation_time"]
            self.generation_metrics["avg_generation_time"] = (
                (current_avg * (total_reports - 1) + generation_time) / total_reports
            )
            
            # Prepare final report
            final_report = {
                "report_id": context["report_id"],
                "report_type": report_type.value,
                "generated_at": datetime.now().isoformat(),
                "generation_time": generation_time,
                "llm_provider": llm_provider.value,
                "llm_model": llm_config.model,
                "content": enhanced_report,
                "metadata": {
                    "prompt_template": prompt_template.template_name,
                    "context_fields": prompt_template.context_fields,
                    "usage": llm_response.get("usage", {}),
                    "word_count": len(enhanced_report.split()),
                    "character_count": len(enhanced_report)
                }
            }
            
            # Cache the report
            self.report_cache[cache_key] = (final_report, time.time())
            
            return final_report
            
        except Exception as e:
            self.generation_metrics["failed_generations"] += 1
            logger.error(f"Report generation failed: {e}")
            return {"error": f"Report generation failed: {e}"}
    
    async def _enhance_report(self, content: str, report_type: ReportType, 
                            context: Dict[str, Any]) -> str:
        """Enhance report with additional formatting and validation"""
        enhanced_content = content
        
        # Add report header
        header = f"""# {report_type.value.replace('_', ' ').title()}
**Report ID:** {context['report_id']}
**Generated:** {context['current_date']}
**Target:** {context.get('target_info', {}).get('organization', 'N/A')}

---

"""
        enhanced_content = header + enhanced_content
        
        # Add footer
        footer = f"""

---

**Report Generation Details:**
- Generated by: Aetherveil Sentinel LLM Report Generator
- Report Type: {report_type.value}
- Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
- Version: 1.0

*This report was generated using advanced AI-powered analysis. Please review all findings and recommendations with qualified cybersecurity professionals.*
"""
        enhanced_content += footer
        
        # Validate and sanitize content
        enhanced_content = self._sanitize_content(enhanced_content)
        
        return enhanced_content
    
    def _sanitize_content(self, content: str) -> str:
        """Sanitize report content"""
        # Remove potentially harmful content
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        sanitized = re.sub(r'<iframe[^>]*>.*?</iframe>', '', sanitized, flags=re.DOTALL | re.IGNORECASE)
        
        # Fix markdown formatting issues
        sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)  # Max 2 consecutive newlines
        sanitized = re.sub(r'#{7,}', '######', sanitized)  # Max 6 heading levels
        
        return sanitized.strip()
    
    async def generate_multi_perspective_report(self, report_type: ReportType, 
                                              report_data: ReportData,
                                              perspectives: List[str] = None) -> Dict[str, Any]:
        """Generate report from multiple perspectives for comprehensive analysis"""
        if perspectives is None:
            perspectives = ["technical", "business", "compliance", "risk"]
        
        reports = {}
        
        for perspective in perspectives:
            # Customize instructions for each perspective
            if perspective == "technical":
                custom_instructions = """Focus on technical details, implementation specifics, and technical remediation steps. 
                Include code snippets, configuration examples, and technical references."""
            elif perspective == "business":
                custom_instructions = """Focus on business impact, cost implications, and strategic recommendations. 
                Translate technical findings into business language."""
            elif perspective == "compliance":
                custom_instructions = """Focus on regulatory compliance, industry standards, and audit requirements. 
                Map findings to relevant compliance frameworks."""
            elif perspective == "risk":
                custom_instructions = """Focus on risk assessment, threat modeling, and risk mitigation strategies. 
                Provide quantitative risk analysis where possible."""
            else:
                custom_instructions = f"Focus on the {perspective} perspective of the security assessment."
            
            # Generate report for this perspective
            report = await self.generate_report(
                report_type=report_type,
                report_data=report_data,
                llm_provider=LLMProvider.OPENAI,
                custom_instructions=custom_instructions
            )
            
            reports[perspective] = report
        
        # Generate a consolidated report
        consolidated_report = await self._consolidate_reports(reports, report_type)
        
        return {
            "consolidated_report": consolidated_report,
            "perspective_reports": reports,
            "generation_summary": {
                "perspectives": perspectives,
                "total_reports": len(reports) + 1,
                "generated_at": datetime.now().isoformat()
            }
        }
    
    async def _consolidate_reports(self, reports: Dict[str, Any], 
                                  report_type: ReportType) -> Dict[str, Any]:
        """Consolidate multiple perspective reports into a single comprehensive report"""
        # Extract content from each perspective
        perspective_contents = {}
        for perspective, report in reports.items():
            if "error" not in report:
                perspective_contents[perspective] = report.get("content", "")
        
        if not perspective_contents:
            return {"error": "No valid perspective reports to consolidate"}
        
        # Create consolidation prompt
        consolidation_prompt = f"""
You are tasked with consolidating multiple cybersecurity report perspectives into a single comprehensive report.

Report Type: {report_type.value.replace('_', ' ').title()}

Available Perspectives:
{', '.join(perspective_contents.keys())}

Your task is to:
1. Merge complementary information from all perspectives
2. Resolve any conflicts or inconsistencies
3. Create a balanced view that addresses all stakeholders
4. Maintain accuracy and avoid redundancy
5. Structure the content logically
6. Ensure all critical findings are preserved

Please create a comprehensive consolidated report that incorporates insights from all perspectives while maintaining clarity and professional presentation.

Perspective Reports:
{json.dumps(perspective_contents, indent=2)}
"""
        
        # Generate consolidated report
        llm_config = self.llm_configs.get(LLMProvider.OPENAI)
        if not llm_config:
            return {"error": "OpenAI configuration not available for consolidation"}
        
        system_prompt = "You are an expert cybersecurity consultant specializing in comprehensive report consolidation and analysis."
        
        llm_response = await self._make_llm_request(llm_config, consolidation_prompt, system_prompt)
        
        if "error" in llm_response:
            return llm_response
        
        return {
            "report_id": str(uuid.uuid4()),
            "report_type": f"{report_type.value}_consolidated",
            "generated_at": datetime.now().isoformat(),
            "content": llm_response["content"],
            "metadata": {
                "consolidated_from": list(perspective_contents.keys()),
                "llm_provider": LLMProvider.OPENAI.value,
                "usage": llm_response.get("usage", {})
            }
        }
    
    async def get_generation_metrics(self) -> Dict[str, Any]:
        """Get report generation metrics"""
        return {
            "metrics": self.generation_metrics,
            "cache_size": len(self.report_cache),
            "configured_providers": [
                provider.value for provider, config in self.llm_configs.items()
                if config.api_key
            ],
            "available_report_types": [rt.value for rt in ReportType],
            "template_count": len(self.prompt_templates)
        }
    
    async def validate_report_data(self, report_data: ReportData) -> Dict[str, Any]:
        """Validate report data structure and completeness"""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "completeness_score": 0.0
        }
        
        # Check required fields
        required_fields = ["target_info", "findings", "vulnerabilities"]
        for field in required_fields:
            if not getattr(report_data, field, None):
                validation_results["errors"].append(f"Missing required field: {field}")
                validation_results["is_valid"] = False
        
        # Check data quality
        if report_data.findings:
            for i, finding in enumerate(report_data.findings):
                if not finding.get("title") or not finding.get("description"):
                    validation_results["warnings"].append(f"Finding {i+1} missing title or description")
        
        if report_data.vulnerabilities:
            for i, vuln in enumerate(report_data.vulnerabilities):
                if not vuln.get("title") or not vuln.get("severity"):
                    validation_results["warnings"].append(f"Vulnerability {i+1} missing title or severity")
        
        # Calculate completeness score
        total_fields = 9  # Total fields in ReportData
        complete_fields = sum(1 for field in ["target_info", "findings", "vulnerabilities", 
                                            "recommendations", "threat_intelligence", 
                                            "attack_timeline", "risk_score", "severity", "metadata"]
                             if getattr(report_data, field, None))
        
        validation_results["completeness_score"] = complete_fields / total_fields
        
        return validation_results
    
    def add_custom_template(self, report_type: str, template_content: str, 
                          context_fields: List[str]) -> bool:
        """Add custom report template"""
        try:
            # Convert string to enum if needed
            if isinstance(report_type, str):
                report_type_enum = ReportType(report_type)
            else:
                report_type_enum = report_type
            
            template = PromptTemplate(
                template_name=f"custom_{report_type}",
                template_content=template_content,
                context_fields=context_fields
            )
            
            self.prompt_templates[report_type_enum] = template
            logger.info(f"Added custom template for {report_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add custom template: {e}")
            return False
    
    async def export_report(self, report: Dict[str, Any], 
                          export_format: str = "markdown") -> str:
        """Export report to different formats"""
        if export_format == "markdown":
            return report.get("content", "")
        elif export_format == "json":
            return json.dumps(report, indent=2)
        elif export_format == "html":
            # Convert markdown to HTML (simplified)
            import markdown
            content = report.get("content", "")
            return markdown.markdown(content)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")