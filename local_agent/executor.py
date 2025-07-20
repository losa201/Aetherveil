#!/usr/bin/env python3
"""
Tool Executor Module - Executes pentesting tools and collects results
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import xml.etree.ElementTree as ET

from config import Config

class ToolExecutor:
    """Executes pentesting tools and manages results"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tool_configs = {}
        self.available_tools = set()
        
        # Load tool configurations
        self._load_tool_configs()
    
    async def initialize(self) -> bool:
        """Initialize tool executor"""
        try:
            self.logger.info("⚡ Initializing tool executor...")
            
            # Check tool availability
            await self._check_tool_availability()
            
            # Setup working directories
            await self._setup_directories()
            
            # Install missing tools
            await self._install_missing_tools()
            
            self.logger.info(f"✅ Tool executor initialized with {len(self.available_tools)} tools")
            return True
            
        except Exception as e:
            self.logger.error(f"Tool executor initialization failed: {e}")
            return False
    
    def _load_tool_configs(self):
        """Load tool-specific configurations"""
        self.tool_configs = {
            "nmap": {
                "binary": "nmap",
                "output_formats": ["xml", "normal"],
                "common_flags": {
                    "-oX": "output.xml",
                    "-oN": "output.txt",
                    "--stats-every": "30s"
                },
                "safety_flags": ["--max-rate", "100", "--max-retries", "2"],
                "timeout": 600
            },
            
            "masscan": {
                "binary": "masscan",
                "output_formats": ["xml", "list"],
                "common_flags": {
                    "-oX": "output.xml",
                    "-oL": "output.txt",
                    "--rate": "1000"
                },
                "safety_flags": ["--rate", "1000"],
                "timeout": 300
            },
            
            "rustscan": {
                "binary": "rustscan",
                "output_formats": ["json"],
                "common_flags": {
                    "--greppable": True,
                    "--timeout": "2000"
                },
                "safety_flags": ["--batch-size", "1000"],
                "timeout": 180
            },
            
            "wfuzz": {
                "binary": "wfuzz",
                "output_formats": ["json"],
                "common_flags": {
                    "--hc": "404",
                    "--threads": "10",
                    "-f": "output.json,json"
                },
                "safety_flags": ["--req-delay", "0.1"],
                "timeout": 300
            },
            
            "ffuf": {
                "binary": "ffuf",
                "output_formats": ["json"],
                "common_flags": {
                    "-mc": "200,301,302,403",
                    "-t": "10",
                    "-o": "output.json",
                    "-of": "json"
                },
                "safety_flags": ["-p", "0.1-1.0"],
                "timeout": 300
            },
            
            "sqlmap": {
                "binary": "sqlmap",
                "output_formats": ["json"],
                "common_flags": {
                    "--batch": True,
                    "--output-dir": "sqlmap_output",
                    "--flush-session": True
                },
                "safety_flags": ["--delay", "1", "--timeout", "10"],
                "timeout": 600
            },
            
            "nikto": {
                "binary": "nikto",
                "output_formats": ["json", "xml"],
                "common_flags": {
                    "-Format": "json",
                    "-output": "output.json",
                    "-maxtime": "300"
                },
                "safety_flags": ["-Pause", "1"],
                "timeout": 400
            },
            
            "nuclei": {
                "binary": "nuclei",
                "output_formats": ["json"],
                "common_flags": {
                    "-json": True,
                    "-output": "output.json",
                    "-rate-limit": "10"
                },
                "safety_flags": ["-rate-limit", "10", "-timeout", "5"],
                "timeout": 600
            },
            
            "gcloud": {
                "binary": "gcloud",
                "output_formats": ["json"],
                "common_flags": {
                    "--format": "json",
                    "--quiet": True
                },
                "timeout": 120
            },
            
            "curl": {
                "binary": "curl",
                "output_formats": ["text", "json"],
                "common_flags": {
                    "-s": True,
                    "-L": True,
                    "--max-time": "30"
                },
                "safety_flags": ["--max-time", "30", "--connect-timeout", "10"],
                "timeout": 60
            }
        }
    
    async def _check_tool_availability(self):
        """Check which tools are available on the system"""
        self.logger.info("🔍 Checking tool availability...")
        
        for tool_name, config in self.tool_configs.items():
            try:
                result = await asyncio.create_subprocess_exec(
                    "which", config["binary"],
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await result.communicate()
                
                if result.returncode == 0:
                    self.available_tools.add(tool_name)
                    self.logger.debug(f"✅ {tool_name} available")
                else:
                    self.logger.warning(f"❌ {tool_name} not found")
                    
            except Exception as e:
                self.logger.warning(f"Error checking {tool_name}: {e}")
        
        self.logger.info(f"Available tools: {', '.join(sorted(self.available_tools))}")
    
    async def _setup_directories(self):
        """Setup working directories"""
        directories = [
            self.config.storage.results_dir,
            self.config.storage.temp_dir,
            self.config.storage.tools_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def _install_missing_tools(self):
        """Install commonly used tools that are missing"""
        missing_tools = set(self.tool_configs.keys()) - self.available_tools
        
        if not missing_tools:
            return
        
        self.logger.info(f"📦 Installing missing tools: {', '.join(missing_tools)}")
        
        install_commands = {
            "nmap": ["apt", "update", "&&", "apt", "install", "-y", "nmap"],
            "masscan": ["apt", "install", "-y", "masscan"],
            "rustscan": ["cargo", "install", "rustscan"],
            "wfuzz": ["pip3", "install", "wfuzz"],
            "ffuf": ["go", "install", "github.com/ffuf/ffuf@latest"],
            "sqlmap": ["pip3", "install", "sqlmap"],
            "nikto": ["apt", "install", "-y", "nikto"],
            "nuclei": ["go", "install", "-v", "github.com/projectdiscovery/nuclei/v2/cmd/nuclei@latest"],
            "curl": ["apt", "install", "-y", "curl"]
        }
        
        for tool in missing_tools:
            if tool in install_commands:
                try:
                    self.logger.info(f"Installing {tool}...")
                    cmd = " ".join(install_commands[tool])
                    
                    process = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
                    
                    if process.returncode == 0:
                        self.available_tools.add(tool)
                        self.logger.info(f"✅ {tool} installed successfully")
                    else:
                        self.logger.warning(f"Failed to install {tool}: {stderr.decode()}")
                        
                except asyncio.TimeoutError:
                    self.logger.warning(f"Installation of {tool} timed out")
                except Exception as e:
                    self.logger.warning(f"Error installing {tool}: {e}")
    
    async def execute_task(self, task: Dict, timeout: Optional[int] = None) -> Optional[Dict]:
        """Execute a single pentesting task"""
        try:
            tool = task.get("tool", "").lower()
            task_name = task.get("name", "Unknown Task")
            
            self.logger.info(f"🔧 Executing task: {task_name} using {tool}")
            
            # Check if tool is available
            if tool not in self.available_tools:
                self.logger.warning(f"Tool {tool} not available, skipping task")
                return {
                    "task": task,
                    "status": "skipped",
                    "reason": f"Tool {tool} not available",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Create task-specific working directory
            task_id = f"{tool}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            work_dir = Path(self.config.storage.temp_dir) / task_id
            work_dir.mkdir(parents=True, exist_ok=True)
            
            # Build command
            command = await self._build_command(task, work_dir)
            if not command:
                return None
            
            # Execute with timeout
            task_timeout = timeout or task.get("timeout", self.tool_configs[tool].get("timeout", 300))
            
            result = await self._execute_command(command, work_dir, task_timeout)
            
            # Parse results
            parsed_result = await self._parse_result(result, task, work_dir)
            
            # Cleanup temporary files if configured
            if self.config.execution.cleanup_temp_files:
                shutil.rmtree(work_dir, ignore_errors=True)
            
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return {
                "task": task,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _build_command(self, task: Dict, work_dir: Path) -> Optional[List[str]]:
        """Build command line for task execution"""
        try:
            tool = task.get("tool", "").lower()
            tool_config = self.tool_configs.get(tool, {})
            
            # Start with binary
            command = [tool_config.get("binary", tool)]
            
            # Add common flags
            common_flags = tool_config.get("common_flags", {})
            for flag, value in common_flags.items():
                if isinstance(value, bool) and value:
                    command.append(flag)
                elif not isinstance(value, bool):
                    command.extend([flag, str(value)])
            
            # Add safety flags
            safety_flags = tool_config.get("safety_flags", [])
            command.extend(safety_flags)
            
            # Add task-specific parameters
            parameters = task.get("parameters", {})
            command.extend(await self._build_parameters(parameters, tool))
            
            # Add target
            target = await self._resolve_target(task.get("target", ""))
            if target:
                command.append(target)
            
            # Update paths to work directory
            command = [str(work_dir / arg) if arg.startswith("output") else arg for arg in command]
            
            self.logger.debug(f"Built command: {' '.join(command)}")
            return command
            
        except Exception as e:
            self.logger.error(f"Command building failed: {e}")
            return None
    
    async def _build_parameters(self, parameters: Dict, tool: str) -> List[str]:
        """Build parameter list for specific tool"""
        params = []
        
        for key, value in parameters.items():
            if isinstance(value, bool) and value:
                params.append(key)
            elif not isinstance(value, bool):
                params.extend([key, str(value)])
        
        return params
    
    async def _resolve_target(self, target: str) -> Optional[str]:
        """Resolve target specification"""
        if not target or target == "auto":
            # Use default target from config
            return self.config.execution.default_target
        
        # Validate target format
        if self._is_valid_target(target):
            return target
        
        self.logger.warning(f"Invalid target format: {target}")
        return None
    
    def _is_valid_target(self, target: str) -> bool:
        """Validate target format for security"""
        # Basic validation - extend as needed
        if not target:
            return False
        
        # Block obviously malicious patterns
        dangerous_patterns = [";", "|", "&", "`", "$", "rm ", "curl ", "wget "]
        if any(pattern in target.lower() for pattern in dangerous_patterns):
            return False
        
        return True
    
    async def _execute_command(self, command: List[str], work_dir: Path, timeout: int) -> Dict:
        """Execute command with proper error handling"""
        try:
            start_time = datetime.now()
            
            # Change to working directory for execution
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy()
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                return {
                    "returncode": process.returncode,
                    "stdout": stdout.decode('utf-8', errors='ignore'),
                    "stderr": stderr.decode('utf-8', errors='ignore'),
                    "duration": duration,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "command": " ".join(command),
                    "work_dir": str(work_dir)
                }
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                
                return {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout} seconds",
                    "duration": timeout,
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "command": " ".join(command),
                    "work_dir": str(work_dir),
                    "timeout": True
                }
                
        except Exception as e:
            return {
                "returncode": -2,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "duration": 0,
                "start_time": start_time.isoformat() if 'start_time' in locals() else datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "command": " ".join(command),
                "work_dir": str(work_dir),
                "error": str(e)
            }
    
    async def _parse_result(self, result: Dict, task: Dict, work_dir: Path) -> Dict:
        """Parse execution result and extract findings"""
        try:
            tool = task.get("tool", "").lower()
            
            # Base result structure
            parsed = {
                "task": task,
                "execution": result,
                "status": "completed" if result["returncode"] == 0 else "failed",
                "timestamp": datetime.now().isoformat(),
                "findings": [],
                "vulnerabilities": [],
                "interesting_findings": False
            }
            
            # Tool-specific parsing
            if tool == "nmap":
                parsed.update(await self._parse_nmap_result(result, work_dir))
            elif tool in ["wfuzz", "ffuf"]:
                parsed.update(await self._parse_fuzzer_result(result, work_dir))
            elif tool == "sqlmap":
                parsed.update(await self._parse_sqlmap_result(result, work_dir))
            elif tool == "nikto":
                parsed.update(await self._parse_nikto_result(result, work_dir))
            elif tool == "nuclei":
                parsed.update(await self._parse_nuclei_result(result, work_dir))
            elif tool == "gcloud":
                parsed.update(await self._parse_gcloud_result(result, work_dir))
            else:
                # Generic parsing
                parsed.update(await self._parse_generic_result(result, work_dir))
            
            # Determine if findings are interesting
            parsed["interesting_findings"] = (
                len(parsed["vulnerabilities"]) > 0 or
                len(parsed["findings"]) > 5 or
                any("interesting" in str(f).lower() for f in parsed["findings"])
            )
            
            return parsed
            
        except Exception as e:
            self.logger.error(f"Result parsing failed: {e}")
            return {
                "task": task,
                "execution": result,
                "status": "parse_error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "findings": [],
                "vulnerabilities": []
            }
    
    async def _parse_nmap_result(self, result: Dict, work_dir: Path) -> Dict:
        """Parse nmap XML output"""
        findings = []
        vulnerabilities = []
        
        try:
            xml_file = work_dir / "output.xml"
            if xml_file.exists():
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                for host in root.findall('host'):
                    host_info = {"type": "host_discovery"}
                    
                    # Get host address
                    address = host.find('address')
                    if address is not None:
                        host_info["ip"] = address.get('addr')
                        host_info["addr_type"] = address.get('addrtype')
                    
                    # Get hostname
                    hostnames = host.find('hostnames')
                    if hostnames is not None:
                        hostname = hostnames.find('hostname')
                        if hostname is not None:
                            host_info["hostname"] = hostname.get('name')
                    
                    # Get port information
                    ports = host.find('ports')
                    if ports is not None:
                        host_info["ports"] = []
                        for port in ports.findall('port'):
                            port_info = {
                                "port": port.get('portid'),
                                "protocol": port.get('protocol'),
                                "state": port.find('state').get('state') if port.find('state') is not None else "unknown"
                            }
                            
                            # Service information
                            service = port.find('service')
                            if service is not None:
                                port_info["service"] = {
                                    "name": service.get('name'),
                                    "version": service.get('version'),
                                    "product": service.get('product')
                                }
                                
                                # Check for potentially vulnerable services
                                if self._is_potentially_vulnerable_service(port_info):
                                    vulnerabilities.append({
                                        "type": "potentially_vulnerable_service",
                                        "port": port_info["port"],
                                        "service": port_info["service"]["name"],
                                        "version": port_info["service"].get("version", "unknown"),
                                        "severity": "medium"
                                    })
                            
                            host_info["ports"].append(port_info)
                    
                    findings.append(host_info)
            
        except ET.ParseError as e:
            self.logger.warning(f"Failed to parse nmap XML: {e}")
        except Exception as e:
            self.logger.error(f"Nmap parsing error: {e}")
        
        return {"findings": findings, "vulnerabilities": vulnerabilities}
    
    async def _parse_fuzzer_result(self, result: Dict, work_dir: Path) -> Dict:
        """Parse fuzzer (wfuzz/ffuf) JSON output"""
        findings = []
        vulnerabilities = []
        
        try:
            json_file = work_dir / "output.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        finding = {
                            "type": "directory_fuzzing",
                            "url": item.get("url", ""),
                            "status_code": item.get("code", 0),
                            "size": item.get("chars", 0),
                            "words": item.get("words", 0)
                        }
                        
                        # Check for interesting status codes
                        status = item.get("code", 0)
                        if status in [200, 301, 302, 403]:
                            finding["interesting"] = True
                            
                            if status == 200:
                                vulnerabilities.append({
                                    "type": "accessible_endpoint",
                                    "url": finding["url"],
                                    "status_code": status,
                                    "severity": "low"
                                })
                        
                        findings.append(finding)
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.warning(f"Failed to parse fuzzer output: {e}")
        except Exception as e:
            self.logger.error(f"Fuzzer parsing error: {e}")
        
        return {"findings": findings, "vulnerabilities": vulnerabilities}
    
    async def _parse_sqlmap_result(self, result: Dict, work_dir: Path) -> Dict:
        """Parse sqlmap output"""
        findings = []
        vulnerabilities = []
        
        try:
            # Look for sqlmap output directory
            sqlmap_dir = work_dir / "sqlmap_output"
            if sqlmap_dir.exists():
                # Parse log files for injection findings
                for log_file in sqlmap_dir.glob("*.log"):
                    with open(log_file, 'r') as f:
                        content = f.read()
                        
                        if "injectable" in content.lower():
                            vulnerabilities.append({
                                "type": "sql_injection",
                                "parameter": "detected",
                                "severity": "high",
                                "details": "SQL injection vulnerability detected"
                            })
                        
                        findings.append({
                            "type": "sqlmap_scan",
                            "log_file": str(log_file),
                            "injectable": "injectable" in content.lower()
                        })
            
        except Exception as e:
            self.logger.error(f"SQLMap parsing error: {e}")
        
        return {"findings": findings, "vulnerabilities": vulnerabilities}
    
    async def _parse_nikto_result(self, result: Dict, work_dir: Path) -> Dict:
        """Parse nikto JSON output"""
        findings = []
        vulnerabilities = []
        
        try:
            json_file = work_dir / "output.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for scan in data.get("vulnerabilities", []):
                    finding = {
                        "type": "web_vulnerability",
                        "id": scan.get("id"),
                        "osvdb": scan.get("osvdb"),
                        "method": scan.get("method"),
                        "url": scan.get("url"),
                        "message": scan.get("msg")
                    }
                    
                    # Classify severity based on OSVDB or message content
                    severity = "low"
                    if scan.get("osvdb") or "vulnerable" in scan.get("msg", "").lower():
                        severity = "medium"
                        vulnerabilities.append({
                            "type": "web_vulnerability",
                            "source": "nikto",
                            "details": scan.get("msg"),
                            "severity": severity
                        })
                    
                    findings.append(finding)
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.warning(f"Failed to parse nikto output: {e}")
        except Exception as e:
            self.logger.error(f"Nikto parsing error: {e}")
        
        return {"findings": findings, "vulnerabilities": vulnerabilities}
    
    async def _parse_nuclei_result(self, result: Dict, work_dir: Path) -> Dict:
        """Parse nuclei JSON output"""
        findings = []
        vulnerabilities = []
        
        try:
            json_file = work_dir / "output.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            
                            finding = {
                                "type": "nuclei_detection",
                                "template_id": data.get("template-id"),
                                "template_name": data.get("info", {}).get("name"),
                                "severity": data.get("info", {}).get("severity", "info"),
                                "matched_at": data.get("matched-at"),
                                "host": data.get("host")
                            }
                            
                            # Add to vulnerabilities if severity is medium or higher
                            severity = data.get("info", {}).get("severity", "info").lower()
                            if severity in ["medium", "high", "critical"]:
                                vulnerabilities.append({
                                    "type": "nuclei_vulnerability",
                                    "template_id": finding["template_id"],
                                    "template_name": finding["template_name"],
                                    "severity": severity,
                                    "host": finding["host"]
                                })
                            
                            findings.append(finding)
                            
                        except json.JSONDecodeError:
                            continue
            
        except FileNotFoundError:
            self.logger.warning("Nuclei output file not found")
        except Exception as e:
            self.logger.error(f"Nuclei parsing error: {e}")
        
        return {"findings": findings, "vulnerabilities": vulnerabilities}
    
    async def _parse_gcloud_result(self, result: Dict, work_dir: Path) -> Dict:
        """Parse gcloud command output"""
        findings = []
        vulnerabilities = []
        
        try:
            if result["returncode"] == 0 and result["stdout"]:
                data = json.loads(result["stdout"])
                
                finding = {
                    "type": "gcloud_query",
                    "data": data,
                    "command": result["command"]
                }
                
                # Basic analysis for common misconfigurations
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            # Check for public buckets
                            if "bucket" in result["command"] and item.get("storageClass") == "STANDARD":
                                if any(binding.get("role") == "roles/storage.objectViewer" 
                                      for binding in item.get("iamConfiguration", {}).get("bindings", [])):
                                    vulnerabilities.append({
                                        "type": "public_storage_bucket",
                                        "bucket": item.get("name"),
                                        "severity": "medium"
                                    })
                
                findings.append(finding)
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Failed to parse gcloud output: {e}")
        except Exception as e:
            self.logger.error(f"Gcloud parsing error: {e}")
        
        return {"findings": findings, "vulnerabilities": vulnerabilities}
    
    async def _parse_generic_result(self, result: Dict, work_dir: Path) -> Dict:
        """Generic parsing for unknown tools"""
        findings = []
        
        # Extract basic information from stdout/stderr
        if result["stdout"]:
            findings.append({
                "type": "tool_output",
                "stdout": result["stdout"][:1000],  # Limit size
                "lines": len(result["stdout"].split('\n'))
            })
        
        if result["stderr"]:
            findings.append({
                "type": "tool_errors",
                "stderr": result["stderr"][:1000],  # Limit size
                "lines": len(result["stderr"].split('\n'))
            })
        
        return {"findings": findings, "vulnerabilities": []}
    
    def _is_potentially_vulnerable_service(self, port_info: Dict) -> bool:
        """Check if service version indicates potential vulnerability"""
        service = port_info.get("service", {})
        service_name = service.get("name", "").lower()
        version = service.get("version", "").lower()
        
        # Common vulnerable services/versions (simplified)
        vulnerable_patterns = [
            ("ssh", ["openssh 7.4"]),
            ("ftp", ["vsftpd 2.3.4"]),
            ("http", ["apache 2.2"]),
            ("mysql", ["mysql 5.5"]),
            ("smb", ["samba 3."])
        ]
        
        for vuln_service, vuln_versions in vulnerable_patterns:
            if vuln_service in service_name:
                for vuln_version in vuln_versions:
                    if vuln_version in version:
                        return True
        
        return False
    
    async def shutdown(self) -> None:
        """Shutdown the executor"""
        try:
            # Kill any running processes
            # Cleanup temporary files if needed
            self.logger.info("✅ Tool executor shutdown complete")
        except Exception as e:
            self.logger.error(f"Executor shutdown error: {e}")