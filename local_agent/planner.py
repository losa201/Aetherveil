#!/usr/bin/env python3
"""
AI Planner Module - Local LLM-based task planning and strategy generation
"""

import asyncio
import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import subprocess
import tempfile

from config import Config

class AIPlanner:
    """AI-powered planning using local LLM (llama.cpp)"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm_process = None
        self.model_path = None
        self.prompt_templates = {}
        
        # Load prompt templates
        self._load_prompt_templates()
    
    async def initialize(self) -> bool:
        """Initialize local LLM for planning"""
        try:
            self.logger.info("🧠 Initializing AI planner with local LLM...")
            
            # Check for llama.cpp installation
            if not await self._check_llama_cpp():
                return False
            
            # Load model
            if not await self._load_model():
                return False
            
            # Test inference
            if not await self._test_inference():
                return False
            
            self.logger.info("✅ AI planner initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"AI planner initialization failed: {e}")
            return False
    
    async def _check_llama_cpp(self) -> bool:
        """Check if llama.cpp is available"""
        try:
            result = subprocess.run(["llama-cpp-python", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.logger.info("✅ llama.cpp found")
                return True
            else:
                # Try alternative installation
                self.logger.warning("llama.cpp not found, attempting to install...")
                return await self._install_llama_cpp()
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return await self._install_llama_cpp()
    
    async def _install_llama_cpp(self) -> bool:
        """Install llama.cpp for ARM64/x64"""
        try:
            self.logger.info("📦 Installing llama.cpp...")
            
            # Install via pip (supports ARM64)
            install_cmd = [
                "pip", "install", "llama-cpp-python",
                "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu"
            ]
            
            result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.logger.info("✅ llama.cpp installed successfully")
                return True
            else:
                self.logger.error(f"Failed to install llama.cpp: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Installation error: {e}")
            return False
    
    async def _load_model(self) -> bool:
        """Load the local LLM model"""
        try:
            # Check for available models
            model_dir = Path(self.config.llm.model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Look for existing models
            model_files = list(model_dir.glob("*.gguf")) + list(model_dir.glob("*.bin"))
            
            if not model_files:
                self.logger.info("📥 No local model found, downloading default model...")
                if not await self._download_default_model():
                    return False
                model_files = list(model_dir.glob("*.gguf"))
            
            if not model_files:
                self.logger.error("No model files available")
                return False
            
            # Use the first available model
            self.model_path = str(model_files[0])
            self.logger.info(f"🤖 Using model: {self.model_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            return False
    
    async def _download_default_model(self) -> bool:
        """Download a small, efficient model for pentesting"""
        try:
            # Use a small but capable model (CodeLlama 7B quantized)
            model_url = "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf"
            model_path = Path(self.config.llm.model_dir) / "codellama-7b-instruct.Q4_K_M.gguf"
            
            self.logger.info(f"📥 Downloading model to {model_path}...")
            
            # Use wget or curl for download
            download_cmd = ["wget", "-O", str(model_path), model_url]
            
            process = subprocess.Popen(download_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0 and model_path.exists():
                self.logger.info("✅ Model downloaded successfully")
                return True
            else:
                self.logger.error(f"Download failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Model download error: {e}")
            return False
    
    async def _test_inference(self) -> bool:
        """Test basic LLM inference"""
        try:
            test_prompt = "What is penetration testing?"
            response = await self._generate_response(test_prompt, max_tokens=50)
            
            if response and len(response.strip()) > 10:
                self.logger.info("✅ LLM inference test passed")
                return True
            else:
                self.logger.error("LLM inference test failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Inference test error: {e}")
            return False
    
    def _load_prompt_templates(self):
        """Load prompt templates for different categories"""
        self.prompt_templates = {
            "web": {
                "system": "You are an expert web application security tester. Generate specific, actionable penetration testing tasks.",
                "planning": """Given the following context, generate a focused web application penetration testing plan:

Category: Web Application Security
Target: {target}
Previous Knowledge: {knowledge}
Recent Results: {recent_results}
Cycle: {cycle}

Generate 3-5 specific tasks focusing on:
- Information gathering and reconnaissance
- Authentication and session management testing
- Input validation vulnerabilities (XSS, SQLi, etc.)
- Business logic flaws
- Configuration issues

Format as JSON with this structure:
{{
  "category": "web",
  "priority": "high|medium|low",
  "estimated_duration": "minutes",
  "tasks": [
    {{
      "name": "Task name",
      "type": "reconnaissance|scanning|exploitation|verification",
      "tool": "nmap|sqlmap|nikto|wfuzz|custom",
      "target": "specific target",
      "parameters": {{"key": "value"}},
      "expected_outcome": "description",
      "risk_level": "low|medium|high"
    }}
  ],
  "success_criteria": ["criteria1", "criteria2"],
  "learning_objectives": ["objective1", "objective2"]
}}"""
            },
            
            "api": {
                "system": "You are an expert API security specialist. Generate comprehensive API testing strategies.",
                "planning": """Generate an API security testing plan:

Category: API Security
Target: {target}
Previous Knowledge: {knowledge}
Recent Results: {recent_results}
Cycle: {cycle}

Focus on:
- API discovery and enumeration
- Authentication bypass techniques
- Authorization flaws (BOLA, BFLA)
- Input validation and injection
- Rate limiting and DoS
- Data exposure issues

Use tools like: wfuzz, ffuf, curl, custom scripts
Format as JSON following the task structure."""
            },
            
            "cloud": {
                "system": "You are a cloud security expert specializing in GCP, AWS, and Azure. Generate cloud security assessment tasks.",
                "planning": """Generate a cloud infrastructure security assessment plan:

Category: Cloud Security
Target: {target}
Previous Knowledge: {knowledge}
Recent Results: {recent_results}
Cycle: {cycle}

Focus on:
- Cloud service enumeration
- IAM misconfigurations
- Storage bucket security
- Network security groups
- Compute instance vulnerabilities
- Serverless security

Use tools like: gcloud, aws cli, cloud_enum, ScoutSuite
Format as JSON following the task structure."""
            },
            
            "infrastructure": {
                "system": "You are a network and infrastructure security expert. Generate comprehensive infrastructure testing plans.",
                "planning": """Generate an infrastructure penetration testing plan:

Category: Infrastructure Security
Target: {target}
Previous Knowledge: {knowledge}
Recent Results: {recent_results}
Cycle: {cycle}

Focus on:
- Network discovery and port scanning
- Service enumeration and fingerprinting
- Vulnerability scanning
- Privilege escalation opportunities
- Lateral movement possibilities
- Persistence mechanisms

Use tools like: nmap, masscan, rustscan, nuclei, metasploit
Format as JSON following the task structure."""
            },
            
            "identity": {
                "system": "You are an identity and access management security specialist. Generate IAM-focused testing strategies.",
                "planning": """Generate an identity and access management security plan:

Category: Identity & Access Management
Target: {target}
Previous Knowledge: {knowledge}
Recent Results: {recent_results}
Cycle: {cycle}

Focus on:
- User enumeration techniques
- Password policy analysis
- Multi-factor authentication bypass
- Single sign-on vulnerabilities
- Privilege escalation through IAM
- Token manipulation and replay

Use tools like: kerbrute, bloodhound, impacket, custom scripts
Format as JSON following the task structure."""
            },
            
            "supply_chain": {
                "system": "You are a supply chain security expert. Generate comprehensive supply chain attack strategies.",
                "planning": """Generate a supply chain security assessment plan:

Category: Supply Chain Security
Target: {target}
Previous Knowledge: {knowledge}
Recent Results: {recent_results}
Cycle: {cycle}

Focus on:
- Dependency analysis and vulnerability scanning
- CI/CD pipeline security
- Container and image security
- Package repository security
- Third-party integrations
- Software composition analysis

Use tools like: syft, grype, trivy, semgrep, custom scripts
Format as JSON following the task structure."""
            }
        }
    
    async def generate_plan(self, category: str, target: Optional[str], 
                          knowledge: Dict, recent_results: List[Dict], 
                          cycle_number: int) -> Optional[Dict]:
        """Generate a comprehensive testing plan using local LLM"""
        try:
            # Select appropriate template
            if category not in self.prompt_templates:
                category = "web"  # Default fallback
            
            template = self.prompt_templates[category]
            
            # Prepare context
            context = {
                "target": target or "general reconnaissance",
                "knowledge": json.dumps(knowledge, indent=2) if knowledge else "None available",
                "recent_results": json.dumps(recent_results[-5:], indent=2) if recent_results else "None available",
                "cycle": cycle_number
            }
            
            # Generate prompt
            system_prompt = template["system"]
            user_prompt = template["planning"].format(**context)
            
            # Get LLM response
            response = await self._generate_response(
                user_prompt, 
                system_prompt=system_prompt,
                max_tokens=self.config.llm.max_tokens
            )
            
            if not response:
                return None
            
            # Parse JSON response
            plan = self._parse_plan_response(response, category)
            
            if plan:
                # Add metadata
                plan["generated_at"] = datetime.now().isoformat()
                plan["target"] = target
                plan["cycle"] = cycle_number
                plan["model_used"] = self.model_path
                
                # Validate and enhance plan
                plan = await self._validate_and_enhance_plan(plan)
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Plan generation failed: {e}")
            return None
    
    async def generate_focused_plan(self, category: str, target: Optional[str]) -> Optional[Dict]:
        """Generate a focused plan for single category execution"""
        try:
            # Use simplified context for focused planning
            knowledge = {"type": "focused_assessment", "category": category}
            recent_results = []
            
            return await self.generate_plan(category, target, knowledge, recent_results, 1)
            
        except Exception as e:
            self.logger.error(f"Focused plan generation failed: {e}")
            return None
    
    async def _generate_response(self, prompt: str, system_prompt: str = None, 
                               max_tokens: int = 1024) -> Optional[str]:
        """Generate response using local LLM"""
        try:
            # Import llama_cpp here to avoid import errors during initialization
            from llama_cpp import Llama
            
            # Initialize LLM if not already done
            if not hasattr(self, '_llm_instance'):
                self.logger.info("🤖 Loading LLM model...")
                self._llm_instance = Llama(
                    model_path=self.model_path,
                    n_ctx=self.config.llm.context_length,
                    n_threads=self.config.llm.threads,
                    verbose=False
                )
            
            # Prepare full prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\n\nAssistant:"
            
            # Generate response
            response = self._llm_instance(
                full_prompt,
                max_tokens=max_tokens,
                temperature=self.config.llm.temperature,
                top_p=self.config.llm.top_p,
                echo=False,
                stop=["User:", "\n\n"]
            )
            
            if response and "choices" in response and response["choices"]:
                return response["choices"][0]["text"].strip()
            
            return None
            
        except Exception as e:
            self.logger.error(f"LLM response generation failed: {e}")
            return None
    
    def _parse_plan_response(self, response: str, category: str) -> Optional[Dict]:
        """Parse and validate LLM response as JSON plan"""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end <= json_start:
                # No JSON found, create fallback plan
                return self._create_fallback_plan(category)
            
            json_text = response[json_start:json_end]
            plan = json.loads(json_text)
            
            # Validate required fields
            required_fields = ["category", "tasks"]
            if not all(field in plan for field in required_fields):
                return self._create_fallback_plan(category)
            
            # Validate tasks structure
            for task in plan.get("tasks", []):
                if not all(field in task for field in ["name", "type", "tool"]):
                    self.logger.warning(f"Invalid task structure: {task}")
                    continue
            
            return plan
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            return self._create_fallback_plan(category)
        except Exception as e:
            self.logger.error(f"Plan parsing error: {e}")
            return None
    
    def _create_fallback_plan(self, category: str) -> Dict:
        """Create a fallback plan when LLM fails"""
        fallback_plans = {
            "web": {
                "category": "web",
                "priority": "medium",
                "estimated_duration": "30",
                "tasks": [
                    {
                        "name": "Basic reconnaissance",
                        "type": "reconnaissance",
                        "tool": "nmap",
                        "target": "auto",
                        "parameters": {"-sV": True, "-sC": True, "--top-ports": "1000"},
                        "expected_outcome": "Service discovery",
                        "risk_level": "low"
                    },
                    {
                        "name": "Web application discovery",
                        "type": "scanning",
                        "tool": "wfuzz",
                        "target": "auto",
                        "parameters": {"wordlist": "common.txt", "extensions": "php,asp,aspx,html"},
                        "expected_outcome": "Hidden directories and files",
                        "risk_level": "low"
                    }
                ],
                "success_criteria": ["Services identified", "Web paths discovered"],
                "learning_objectives": ["Service fingerprinting", "Directory enumeration"]
            },
            
            "api": {
                "category": "api",
                "priority": "medium",
                "estimated_duration": "20",
                "tasks": [
                    {
                        "name": "API endpoint discovery",
                        "type": "reconnaissance",
                        "tool": "wfuzz",
                        "target": "auto",
                        "parameters": {"wordlist": "api-endpoints.txt"},
                        "expected_outcome": "API endpoints identified",
                        "risk_level": "low"
                    }
                ],
                "success_criteria": ["API endpoints found"],
                "learning_objectives": ["API enumeration"]
            },
            
            "infrastructure": {
                "category": "infrastructure",
                "priority": "high",
                "estimated_duration": "45",
                "tasks": [
                    {
                        "name": "Network discovery",
                        "type": "reconnaissance",
                        "tool": "nmap",
                        "target": "auto",
                        "parameters": {"-sn": True, "--top-ports": "100"},
                        "expected_outcome": "Live hosts discovered",
                        "risk_level": "low"
                    },
                    {
                        "name": "Service enumeration",
                        "type": "scanning",
                        "tool": "nmap",
                        "target": "auto",
                        "parameters": {"-sV": True, "-sC": True},
                        "expected_outcome": "Services identified",
                        "risk_level": "medium"
                    }
                ],
                "success_criteria": ["Network mapped", "Services enumerated"],
                "learning_objectives": ["Network topology", "Service identification"]
            }
        }
        
        return fallback_plans.get(category, fallback_plans["web"])
    
    async def _validate_and_enhance_plan(self, plan: Dict) -> Dict:
        """Validate and enhance the generated plan"""
        try:
            # Ensure required fields
            plan.setdefault("priority", "medium")
            plan.setdefault("estimated_duration", "30")
            plan.setdefault("success_criteria", [])
            plan.setdefault("learning_objectives", [])
            
            # Validate and enhance tasks
            enhanced_tasks = []
            for task in plan.get("tasks", []):
                enhanced_task = await self._enhance_task(task)
                if enhanced_task:
                    enhanced_tasks.append(enhanced_task)
            
            plan["tasks"] = enhanced_tasks
            
            # Add safety limits
            if len(plan["tasks"]) > self.config.planning.max_tasks_per_cycle:
                plan["tasks"] = plan["tasks"][:self.config.planning.max_tasks_per_cycle]
                self.logger.warning(f"Limited tasks to {self.config.planning.max_tasks_per_cycle}")
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Plan validation failed: {e}")
            return plan
    
    async def _enhance_task(self, task: Dict) -> Optional[Dict]:
        """Enhance individual task with additional metadata"""
        try:
            # Set defaults
            task.setdefault("risk_level", "medium")
            task.setdefault("timeout", 300)  # 5 minutes
            task.setdefault("retry_count", 2)
            
            # Validate tool availability
            if not await self._validate_tool_availability(task.get("tool")):
                self.logger.warning(f"Tool {task.get('tool')} not available, skipping task")
                return None
            
            # Add task-specific enhancements
            tool = task.get("tool", "")
            if tool == "nmap":
                task["parameters"].setdefault("--max-rate", "100")
                task["parameters"].setdefault("--max-retries", "2")
            elif tool in ["wfuzz", "ffuf"]:
                task["parameters"].setdefault("--hc", "404")
                task["parameters"].setdefault("--threads", "10")
            
            return task
            
        except Exception as e:
            self.logger.error(f"Task enhancement failed: {e}")
            return task
    
    async def _validate_tool_availability(self, tool: str) -> bool:
        """Check if required tool is available"""
        try:
            result = subprocess.run(["which", tool], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the planner"""
        try:
            if hasattr(self, '_llm_instance'):
                del self._llm_instance
            self.logger.info("✅ AI planner shutdown complete")
        except Exception as e:
            self.logger.error(f"Planner shutdown error: {e}")