#!/usr/bin/env python3
"""
Security Gate Script for CI/CD Pipeline

This script analyzes security scan results and determines if the build should pass or fail
based on predefined security thresholds and policies.
"""

import json
import sys
import os
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    """Security finding severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityFinding:
    """Represents a security finding from various tools"""
    tool: str
    severity: Severity
    rule_id: str
    message: str
    filename: str
    line_number: Optional[int] = None
    confidence: Optional[str] = None
    cve_id: Optional[str] = None


class SecurityGate:
    """
    Security gate that analyzes scan results and enforces security policies
    """
    
    def __init__(self):
        self.findings: List[SecurityFinding] = []
        self.thresholds = {
            Severity.CRITICAL: 0,  # No critical findings allowed
            Severity.HIGH: 5,      # Max 5 high severity findings
            Severity.MEDIUM: 20,   # Max 20 medium severity findings
            Severity.LOW: 100      # Max 100 low severity findings
        }
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def analyze_bandit_report(self, report_path: str) -> bool:
        """Analyze Bandit security scan results"""
        try:
            if not os.path.exists(report_path):
                self.logger.warning(f"Bandit report not found: {report_path}")
                return True
            
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            results = report.get('results', [])
            for result in results:
                severity_map = {
                    'LOW': Severity.LOW,
                    'MEDIUM': Severity.MEDIUM,
                    'HIGH': Severity.HIGH
                }
                
                severity = severity_map.get(result.get('issue_severity'), Severity.LOW)
                
                # Treat certain rules as critical
                if result.get('test_id') in ['B602', 'B605', 'B607']:  # Shell injection
                    severity = Severity.CRITICAL
                
                finding = SecurityFinding(
                    tool="bandit",
                    severity=severity,
                    rule_id=result.get('test_id', 'unknown'),
                    message=result.get('issue_text', 'No description'),
                    filename=result.get('filename', 'unknown'),
                    line_number=result.get('line_number'),
                    confidence=result.get('issue_confidence')
                )
                self.findings.append(finding)
            
            self.logger.info(f"Analyzed {len(results)} Bandit findings")
            return True
            
        except Exception as e:
            self.logger.error(f"Error analyzing Bandit report: {e}")
            return False
    
    def analyze_safety_report(self, report_path: str) -> bool:
        """Analyze Safety dependency vulnerability scan results"""
        try:
            if not os.path.exists(report_path):
                self.logger.warning(f"Safety report not found: {report_path}")
                return True
            
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            vulnerabilities = report.get('vulnerabilities', [])
            for vuln in vulnerabilities:
                # Determine severity based on CVE score if available
                severity = Severity.MEDIUM
                if vuln.get('advisory'):
                    advisory = vuln['advisory'].lower()
                    if 'critical' in advisory or 'remote code execution' in advisory:
                        severity = Severity.CRITICAL
                    elif 'high' in advisory:
                        severity = Severity.HIGH
                
                finding = SecurityFinding(
                    tool="safety",
                    severity=severity,
                    rule_id=vuln.get('id', 'unknown'),
                    message=vuln.get('advisory', 'Vulnerability found'),
                    filename=vuln.get('package_name', 'unknown'),
                    cve_id=vuln.get('cve')
                )
                self.findings.append(finding)
            
            self.logger.info(f"Analyzed {len(vulnerabilities)} Safety findings")
            return True
            
        except Exception as e:
            self.logger.error(f"Error analyzing Safety report: {e}")
            return False
    
    def analyze_semgrep_report(self, report_path: str) -> bool:
        """Analyze Semgrep static analysis results"""
        try:
            if not os.path.exists(report_path):
                self.logger.warning(f"Semgrep report not found: {report_path}")
                return True
            
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            results = report.get('results', [])
            for result in results:
                # Map Semgrep severity to our severity levels
                severity_map = {
                    'INFO': Severity.LOW,
                    'WARNING': Severity.MEDIUM,
                    'ERROR': Severity.HIGH
                }
                
                severity = severity_map.get(
                    result.get('extra', {}).get('severity', 'INFO'),
                    Severity.LOW
                )
                
                # Treat certain security rules as critical
                rule_id = result.get('check_id', '')
                if any(pattern in rule_id for pattern in [
                    'security.audit.dangerous',
                    'security.audit.sqli',
                    'security.audit.xss',
                    'security.audit.crypto.weak'
                ]):
                    severity = Severity.CRITICAL
                
                finding = SecurityFinding(
                    tool="semgrep",
                    severity=severity,
                    rule_id=rule_id,
                    message=result.get('message', 'Security issue found'),
                    filename=result.get('path', 'unknown'),
                    line_number=result.get('start', {}).get('line')
                )
                self.findings.append(finding)
            
            self.logger.info(f"Analyzed {len(results)} Semgrep findings")
            return True
            
        except Exception as e:
            self.logger.error(f"Error analyzing Semgrep report: {e}")
            return False
    
    def get_severity_counts(self) -> Dict[Severity, int]:
        """Get count of findings by severity level"""
        counts = {severity: 0 for severity in Severity}
        
        for finding in self.findings:
            counts[finding.severity] += 1
        
        return counts
    
    def check_thresholds(self) -> bool:
        """Check if findings exceed defined thresholds"""
        counts = self.get_severity_counts()
        
        for severity, count in counts.items():
            threshold = self.thresholds[severity]
            if count > threshold:
                self.logger.error(
                    f"Threshold exceeded for {severity.value}: {count} > {threshold}"
                )
                return False
        
        return True
    
    def generate_report(self) -> str:
        """Generate a security report summary"""
        counts = self.get_severity_counts()
        
        report = [
            "Security Gate Analysis Report",
            "=" * 40,
            f"Total findings: {len(self.findings)}",
            ""
        ]
        
        for severity in Severity:
            count = counts[severity]
            threshold = self.thresholds[severity]
            status = "PASS" if count <= threshold else "FAIL"
            report.append(f"{severity.value.upper()}: {count}/{threshold} [{status}]")
        
        report.extend(["", "Findings by tool:"])
        tool_counts = {}
        for finding in self.findings:
            tool_counts[finding.tool] = tool_counts.get(finding.tool, 0) + 1
        
        for tool, count in tool_counts.items():
            report.append(f"  {tool}: {count}")
        
        if self.findings:
            report.extend(["", "Critical and High severity findings:"])
            critical_high = [
                f for f in self.findings 
                if f.severity in [Severity.CRITICAL, Severity.HIGH]
            ]
            
            for finding in critical_high[:10]:  # Show first 10
                report.append(
                    f"  [{finding.severity.value.upper()}] {finding.tool}: "
                    f"{finding.message} ({finding.filename}:{finding.line_number or 'N/A'})"
                )
            
            if len(critical_high) > 10:
                report.append(f"  ... and {len(critical_high) - 10} more")
        
        return "\n".join(report)
    
    def apply_exceptions(self):
        """Apply security exceptions for known false positives"""
        exceptions = [
            # Example: Ignore test files for certain rules
            {
                'tool': 'bandit',
                'rule_id': 'B101',  # assert_used
                'filename_pattern': 'test_'
            },
            {
                'tool': 'semgrep',
                'rule_id': 'python.flask.security.xss',
                'filename_pattern': 'templates/'
            }
        ]
        
        original_count = len(self.findings)
        filtered_findings = []
        
        for finding in self.findings:
            should_exclude = False
            
            for exception in exceptions:
                if (finding.tool == exception['tool'] and
                    finding.rule_id == exception['rule_id'] and
                    exception['filename_pattern'] in finding.filename):
                    should_exclude = True
                    break
            
            if not should_exclude:
                filtered_findings.append(finding)
        
        self.findings = filtered_findings
        excluded_count = original_count - len(self.findings)
        
        if excluded_count > 0:
            self.logger.info(f"Applied exceptions: excluded {excluded_count} findings")
    
    def run(self) -> bool:
        """Run the security gate analysis"""
        success = True
        
        # Analyze all available reports
        bandit_report = os.getenv('BANDIT_REPORT', 'bandit-report.json')
        safety_report = os.getenv('SAFETY_REPORT', 'safety-report.json')
        semgrep_report = os.getenv('SEMGREP_REPORT', 'semgrep-report.json')
        
        if not self.analyze_bandit_report(bandit_report):
            success = False
        
        if not self.analyze_safety_report(safety_report):
            success = False
        
        if not self.analyze_semgrep_report(semgrep_report):
            success = False
        
        # Apply exceptions
        self.apply_exceptions()
        
        # Generate and print report
        report = self.generate_report()
        print(report)
        
        # Check thresholds
        if not self.check_thresholds():
            success = False
        
        # Write detailed report to file
        with open('security-gate-report.txt', 'w') as f:
            f.write(report)
        
        return success


def main():
    """Main entry point"""
    gate = SecurityGate()
    
    if gate.run():
        print("\n✅ Security gate PASSED")
        sys.exit(0)
    else:
        print("\n❌ Security gate FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()