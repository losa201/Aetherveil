"""
Integrated Security System for Aetherveil Sentinel
Combines all security components into a unified, production-ready security framework
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import yaml

# Import all security components
from coordinator.security_manager import SecurityManager, security_manager
from coordinator.zmq_encryption import ZMQEncryptionManager, zmq_encryption_manager
from coordinator.jwt_manager import JWTManager, jwt_manager
from coordinator.rbac_manager import RBACManager, rbac_manager
from coordinator.blockchain_logger import BlockchainLogger, blockchain_logger
from coordinator.rate_limiter import RateLimiter, rate_limiter
from coordinator.gcp_secret_manager import GCPSecretManager, initialize_gcp_secret_manager
from coordinator.security_monitor import SecurityMonitor, security_monitor

logger = logging.getLogger(__name__)

class IntegratedSecuritySystem:
    """
    Integrated Security System that orchestrates all security components
    """
    
    def __init__(self, config_path: str = "/app/config/security_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Security components
        self.security_manager = security_manager
        self.zmq_encryption = zmq_encryption_manager
        self.jwt_manager = jwt_manager
        self.rbac_manager = rbac_manager
        self.blockchain_logger = blockchain_logger
        self.rate_limiter = rate_limiter
        self.gcp_secret_manager: Optional[GCPSecretManager] = None
        self.security_monitor = security_monitor
        
        # System state
        self.initialized = False
        self.running = False
        self.initialization_tasks = []
        
        # Component status
        self.component_status = {
            'security_manager': False,
            'zmq_encryption': False,
            'jwt_manager': False,
            'rbac_manager': False,
            'blockchain_logger': False,
            'rate_limiter': False,
            'gcp_secret_manager': False,
            'security_monitor': False
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load security configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load security config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default security configuration"""
        return {
            'security': {
                'encryption': {
                    'enabled': True,
                    'algorithms': ['AES-256-GCM', 'ChaCha20-Poly1305'],
                    'key_rotation_interval': 86400,  # 24 hours
                    'master_key_source': 'gcp_secret_manager'
                },
                'authentication': {
                    'jwt': {
                        'enabled': True,
                        'access_token_expire': 900,  # 15 minutes
                        'refresh_token_expire': 86400,  # 24 hours
                        'algorithm': 'RS256'
                    },
                    'certificate': {
                        'enabled': True,
                        'mutual_tls': True,
                        'ca_cert_path': '/app/certs/ca.crt',
                        'cert_rotation_interval': 2592000  # 30 days
                    }
                },
                'authorization': {
                    'rbac': {
                        'enabled': True,
                        'default_roles': ['admin', 'coordinator', 'agent', 'monitor'],
                        'permission_cache_ttl': 300
                    }
                },
                'logging': {
                    'blockchain': {
                        'enabled': True,
                        'block_size': 100,
                        'difficulty': 4,
                        'storage_path': '/app/blockchain_logs'
                    }
                },
                'rate_limiting': {
                    'enabled': True,
                    'default_limit': 1000,
                    'window_seconds': 60,
                    'ddos_protection': True
                },
                'monitoring': {
                    'enabled': True,
                    'anomaly_detection': True,
                    'threat_intelligence': True,
                    'alerting': {
                        'email': {
                            'enabled': False,
                            'smtp_server': 'localhost',
                            'recipients': []
                        },
                        'webhook': {
                            'enabled': False,
                            'url': ''
                        }
                    }
                }
            },
            'gcp': {
                'enabled': False,
                'project_id': '',
                'credentials_path': '',
                'secret_manager': {
                    'enabled': False
                }
            },
            'network': {
                'zmq': {
                    'enabled': True,
                    'encryption': True,
                    'curve_auth': True
                },
                'tls': {
                    'min_version': 'TLSv1.2',
                    'cipher_suites': [
                        'ECDHE+AESGCM',
                        'ECDHE+CHACHA20',
                        'DHE+AESGCM',
                        'DHE+CHACHA20'
                    ]
                }
            }
        }
    
    async def initialize(self):
        """Initialize the integrated security system"""
        try:
            logger.info("Initializing Integrated Security System...")
            
            # Initialize components in proper order
            await self._initialize_security_manager()
            await self._initialize_gcp_secret_manager()
            await self._initialize_jwt_manager()
            await self._initialize_rbac_manager()
            await self._initialize_blockchain_logger()
            await self._initialize_rate_limiter()
            await self._initialize_zmq_encryption()
            await self._initialize_security_monitor()
            
            # Verify all components are initialized
            await self._verify_initialization()
            
            # Setup component integrations
            await self._setup_integrations()
            
            self.initialized = True
            logger.info("Integrated Security System initialized successfully")
            
            # Generate initialization report
            await self._generate_initialization_report()
            
        except Exception as e:
            logger.error(f"Failed to initialize Integrated Security System: {e}")
            raise
    
    async def _initialize_security_manager(self):
        """Initialize security manager"""
        try:
            logger.info("Initializing Security Manager...")
            await self.security_manager.initialize()
            self.component_status['security_manager'] = True
            logger.info("Security Manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Security Manager: {e}")
            raise
    
    async def _initialize_gcp_secret_manager(self):
        """Initialize GCP Secret Manager"""
        try:
            gcp_config = self.config.get('gcp', {})
            
            if gcp_config.get('enabled', False):
                logger.info("Initializing GCP Secret Manager...")
                
                project_id = gcp_config.get('project_id')
                credentials_path = gcp_config.get('credentials_path')
                
                if project_id:
                    self.gcp_secret_manager = initialize_gcp_secret_manager(
                        project_id=project_id,
                        credentials_path=credentials_path
                    )
                    
                    await self.gcp_secret_manager.initialize()
                    self.component_status['gcp_secret_manager'] = True
                    logger.info("GCP Secret Manager initialized")
                else:
                    logger.warning("GCP Secret Manager enabled but no project_id configured")
            else:
                logger.info("GCP Secret Manager disabled")
                self.component_status['gcp_secret_manager'] = True  # Mark as "initialized" (disabled)
                
        except Exception as e:
            logger.error(f"Failed to initialize GCP Secret Manager: {e}")
            # Don't raise - GCP Secret Manager is optional
            self.component_status['gcp_secret_manager'] = True
    
    async def _initialize_jwt_manager(self):
        """Initialize JWT manager"""
        try:
            logger.info("Initializing JWT Manager...")
            
            # Load or generate JWT keys
            private_key_path = "/app/keys/jwt_private.pem"
            public_key_path = "/app/keys/jwt_public.pem"
            
            await self.jwt_manager.initialize(private_key_path, public_key_path)
            self.component_status['jwt_manager'] = True
            logger.info("JWT Manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize JWT Manager: {e}")
            raise
    
    async def _initialize_rbac_manager(self):
        """Initialize RBAC manager"""
        try:
            logger.info("Initializing RBAC Manager...")
            
            # Initialize with default roles and permissions
            self.rbac_manager.initialize_default_roles()
            self.component_status['rbac_manager'] = True
            logger.info("RBAC Manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize RBAC Manager: {e}")
            raise
    
    async def _initialize_blockchain_logger(self):
        """Initialize blockchain logger"""
        try:
            logger.info("Initializing Blockchain Logger...")
            
            # Configure blockchain logger
            blockchain_config = self.config.get('security', {}).get('logging', {}).get('blockchain', {})
            
            if blockchain_config.get('enabled', True):
                storage_path = blockchain_config.get('storage_path', '/app/blockchain_logs')
                self.blockchain_logger.storage_path = Path(storage_path)
                self.blockchain_logger.block_size = blockchain_config.get('block_size', 100)
                self.blockchain_logger.difficulty = blockchain_config.get('difficulty', 4)
                
                await self.blockchain_logger.initialize()
                self.component_status['blockchain_logger'] = True
                logger.info("Blockchain Logger initialized")
            else:
                logger.info("Blockchain Logger disabled")
                self.component_status['blockchain_logger'] = True
                
        except Exception as e:
            logger.error(f"Failed to initialize Blockchain Logger: {e}")
            raise
    
    async def _initialize_rate_limiter(self):
        """Initialize rate limiter"""
        try:
            logger.info("Initializing Rate Limiter...")
            
            await self.rate_limiter.initialize()
            self.component_status['rate_limiter'] = True
            logger.info("Rate Limiter initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Rate Limiter: {e}")
            raise
    
    async def _initialize_zmq_encryption(self):
        """Initialize ZMQ encryption"""
        try:
            logger.info("Initializing ZMQ Encryption...")
            
            network_config = self.config.get('network', {}).get('zmq', {})
            
            if network_config.get('enabled', True):
                enable_curve = network_config.get('curve_auth', True)
                self.zmq_encryption.initialize(enable_curve=enable_curve)
                self.component_status['zmq_encryption'] = True
                logger.info("ZMQ Encryption initialized")
            else:
                logger.info("ZMQ Encryption disabled")
                self.component_status['zmq_encryption'] = True
                
        except Exception as e:
            logger.error(f"Failed to initialize ZMQ Encryption: {e}")
            raise
    
    async def _initialize_security_monitor(self):
        """Initialize security monitor"""
        try:
            logger.info("Initializing Security Monitor...")
            
            await self.security_monitor.initialize(
                blockchain_logger=self.blockchain_logger,
                rate_limiter=self.rate_limiter,
                rbac_manager=self.rbac_manager
            )
            self.component_status['security_monitor'] = True
            logger.info("Security Monitor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Security Monitor: {e}")
            raise
    
    async def _verify_initialization(self):
        """Verify all components are properly initialized"""
        try:
            failed_components = [
                name for name, status in self.component_status.items()
                if not status
            ]
            
            if failed_components:
                raise RuntimeError(f"Failed to initialize components: {failed_components}")
            
            logger.info("All security components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization verification failed: {e}")
            raise
    
    async def _setup_integrations(self):
        """Setup integrations between components"""
        try:
            logger.info("Setting up component integrations...")
            
            # Setup security event forwarding
            await self._setup_event_forwarding()
            
            # Setup key management integration
            await self._setup_key_management()
            
            # Setup monitoring integration
            await self._setup_monitoring_integration()
            
            logger.info("Component integrations configured")
            
        except Exception as e:
            logger.error(f"Failed to setup integrations: {e}")
            raise
    
    async def _setup_event_forwarding(self):
        """Setup event forwarding between components"""
        try:
            # Forward rate limiter events to security monitor
            if hasattr(self.rate_limiter, 'register_event_handler'):
                self.rate_limiter.register_event_handler(
                    self.security_monitor.process_event
                )
            
            # Forward authentication events
            if hasattr(self.jwt_manager, 'register_event_handler'):
                self.jwt_manager.register_event_handler(
                    self.security_monitor.process_event
                )
            
            # Forward authorization events
            if hasattr(self.rbac_manager, 'register_event_handler'):
                self.rbac_manager.register_event_handler(
                    self.security_monitor.process_event
                )
            
        except Exception as e:
            logger.error(f"Failed to setup event forwarding: {e}")
    
    async def _setup_key_management(self):
        """Setup key management integration"""
        try:
            # If GCP Secret Manager is available, use it for key storage
            if self.gcp_secret_manager:
                # Store master encryption key
                master_key = self.security_manager.get_encryption_key()
                if master_key:
                    await self.gcp_secret_manager.create_secret(
                        name="master_encryption_key",
                        value=master_key.decode() if isinstance(master_key, bytes) else master_key,
                        secret_type="encryption_key",
                        description="Master encryption key for Aetherveil Sentinel"
                    )
                
                # Store JWT signing key
                # Note: This would need to be implemented in JWT manager
                # to export the private key securely
            
        except Exception as e:
            logger.error(f"Failed to setup key management: {e}")
    
    async def _setup_monitoring_integration(self):
        """Setup monitoring integration"""
        try:
            # Register alert handlers
            from coordinator.security_monitor import AlertType
            
            # Handle authentication failures
            self.security_monitor.register_alert_handler(
                AlertType.AUTHENTICATION_FAILURE,
                self._handle_auth_failure_alert
            )
            
            # Handle rate limit exceeded
            self.security_monitor.register_alert_handler(
                AlertType.RATE_LIMIT_EXCEEDED,
                self._handle_rate_limit_alert
            )
            
            # Handle DDoS attacks
            self.security_monitor.register_alert_handler(
                AlertType.DDOS_ATTACK,
                self._handle_ddos_alert
            )
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring integration: {e}")
    
    async def _handle_auth_failure_alert(self, alert):
        """Handle authentication failure alert"""
        try:
            # Log to blockchain
            self.blockchain_logger.log_security_event(
                event_type="authentication_failure",
                source="integrated_security_system",
                message=f"Authentication failure alert: {alert.title}",
                data=alert.metadata
            )
            
            # Additional response actions could be implemented here
            
        except Exception as e:
            logger.error(f"Failed to handle auth failure alert: {e}")
    
    async def _handle_rate_limit_alert(self, alert):
        """Handle rate limit alert"""
        try:
            # Log to blockchain
            self.blockchain_logger.log_security_event(
                event_type="rate_limit_exceeded",
                source="integrated_security_system",
                message=f"Rate limit alert: {alert.title}",
                data=alert.metadata
            )
            
            # Could implement automatic IP blocking here
            
        except Exception as e:
            logger.error(f"Failed to handle rate limit alert: {e}")
    
    async def _handle_ddos_alert(self, alert):
        """Handle DDoS attack alert"""
        try:
            # Log to blockchain
            self.blockchain_logger.log_security_event(
                event_type="ddos_attack",
                source="integrated_security_system",
                message=f"DDoS attack alert: {alert.title}",
                data=alert.metadata
            )
            
            # Could implement automatic mitigation measures here
            
        except Exception as e:
            logger.error(f"Failed to handle DDoS alert: {e}")
    
    async def _generate_initialization_report(self):
        """Generate initialization report"""
        try:
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'system_version': '1.0.0',
                'components': {
                    name: {
                        'status': 'initialized' if status else 'failed',
                        'version': '1.0.0'
                    }
                    for name, status in self.component_status.items()
                },
                'configuration': {
                    'security_level': 'high',
                    'encryption_enabled': self.config.get('security', {}).get('encryption', {}).get('enabled', True),
                    'authentication_methods': ['jwt', 'certificate'],
                    'authorization_enabled': self.config.get('security', {}).get('authorization', {}).get('rbac', {}).get('enabled', True),
                    'logging_enabled': self.config.get('security', {}).get('logging', {}).get('blockchain', {}).get('enabled', True),
                    'monitoring_enabled': self.config.get('security', {}).get('monitoring', {}).get('enabled', True)
                },
                'statistics': {
                    'total_components': len(self.component_status),
                    'initialized_components': sum(1 for status in self.component_status.values() if status),
                    'failed_components': sum(1 for status in self.component_status.values() if not status)
                }
            }
            
            # Save report
            report_path = Path("/app/logs/security_initialization_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Initialization report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate initialization report: {e}")
    
    async def start(self):
        """Start the integrated security system"""
        try:
            if not self.initialized:
                await self.initialize()
            
            logger.info("Starting Integrated Security System...")
            
            # Start all components
            # Note: Most components start automatically during initialization
            
            self.running = True
            logger.info("Integrated Security System started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Integrated Security System: {e}")
            raise
    
    async def stop(self):
        """Stop the integrated security system"""
        try:
            logger.info("Stopping Integrated Security System...")
            
            self.running = False
            
            # Stop all components
            await self._stop_components()
            
            logger.info("Integrated Security System stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop Integrated Security System: {e}")
    
    async def _stop_components(self):
        """Stop all security components"""
        try:
            # Stop security monitor
            if self.security_monitor:
                await self.security_monitor.shutdown()
            
            # Stop blockchain logger
            if self.blockchain_logger:
                await self.blockchain_logger.shutdown()
            
            # Stop ZMQ encryption
            if self.zmq_encryption:
                self.zmq_encryption.cleanup()
            
            # Stop GCP Secret Manager
            if self.gcp_secret_manager:
                await self.gcp_secret_manager.shutdown()
            
            logger.info("All security components stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop components: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'system': {
                    'initialized': self.initialized,
                    'running': self.running,
                    'timestamp': datetime.utcnow().isoformat()
                },
                'components': self.component_status.copy(),
                'statistics': {
                    'security_manager': self.security_manager.get_security_status() if self.security_manager else {},
                    'jwt_manager': self.jwt_manager.get_statistics() if self.jwt_manager else {},
                    'rbac_manager': self.rbac_manager.get_statistics() if self.rbac_manager else {},
                    'rate_limiter': self.rate_limiter.get_statistics() if self.rate_limiter else {},
                    'blockchain_logger': self.blockchain_logger.get_statistics() if self.blockchain_logger else {},
                    'zmq_encryption': self.zmq_encryption.get_encryption_stats() if self.zmq_encryption else {},
                    'gcp_secret_manager': self.gcp_secret_manager.get_statistics() if self.gcp_secret_manager else {},
                    'security_monitor': self.security_monitor.get_statistics() if self.security_monitor else {}
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'system': {
                    'initialized': False,
                    'running': False,
                    'error': str(e)
                }
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health_status = {
                'overall_health': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'components': {}
            }
            
            # Check each component
            for component_name, component in [
                ('security_manager', self.security_manager),
                ('jwt_manager', self.jwt_manager),
                ('rbac_manager', self.rbac_manager),
                ('rate_limiter', self.rate_limiter),
                ('blockchain_logger', self.blockchain_logger),
                ('zmq_encryption', self.zmq_encryption),
                ('gcp_secret_manager', self.gcp_secret_manager),
                ('security_monitor', self.security_monitor)
            ]:
                try:
                    if component and hasattr(component, 'health_check'):
                        component_health = await component.health_check()
                        health_status['components'][component_name] = component_health
                    else:
                        health_status['components'][component_name] = {
                            'status': 'healthy' if self.component_status.get(component_name, False) else 'unhealthy',
                            'message': 'Component initialized' if self.component_status.get(component_name, False) else 'Component not initialized'
                        }
                except Exception as e:
                    health_status['components'][component_name] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                    health_status['overall_health'] = 'degraded'
            
            # Check overall system health
            unhealthy_components = [
                name for name, status in health_status['components'].items()
                if status.get('status') == 'unhealthy'
            ]
            
            if unhealthy_components:
                health_status['overall_health'] = 'unhealthy'
                health_status['issues'] = f"Unhealthy components: {', '.join(unhealthy_components)}"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Failed to perform health check: {e}")
            return {
                'overall_health': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def create_secure_context(self, entity_id: str, roles: List[str] = None, 
                            permissions: List[str] = None) -> Dict[str, Any]:
        """Create secure context for entity"""
        try:
            # Generate JWT token
            token = self.jwt_manager.create_access_token(
                entity_id=entity_id,
                permissions=permissions or [],
                roles=roles or []
            )
            
            # Get certificate
            certificate = self.security_manager.certificate_manager.get_certificate(entity_id)
            
            # Create secure context
            context = {
                'entity_id': entity_id,
                'access_token': token,
                'certificate': certificate.to_dict() if certificate else None,
                'roles': roles or [],
                'permissions': permissions or [],
                'created_at': datetime.utcnow().isoformat(),
                'expires_at': (datetime.utcnow() + timedelta(seconds=900)).isoformat()
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to create secure context: {e}")
            raise

# Global integrated security system instance
integrated_security_system = IntegratedSecuritySystem()

# Convenience functions for easy access
async def initialize_security():
    """Initialize the integrated security system"""
    return await integrated_security_system.initialize()

async def start_security():
    """Start the integrated security system"""
    return await integrated_security_system.start()

async def stop_security():
    """Stop the integrated security system"""
    return await integrated_security_system.stop()

def get_security_status():
    """Get security system status"""
    return integrated_security_system.get_system_status()

async def perform_security_health_check():
    """Perform security health check"""
    return await integrated_security_system.health_check()