#!/usr/bin/env python3
"""
Staging Environment Smoke Tests

Basic smoke tests to verify that the staging deployment is working correctly.
"""

import os
import sys
import time
import requests
import json
from typing import Dict, Any, List
import logging


class StagingSmokeTest:
    """Smoke test suite for staging environment"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Test results
        self.results = []
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results"""
        self.logger.info(f"Running test: {test_name}")
        
        try:
            start_time = time.time()
            test_func()
            duration = time.time() - start_time
            
            self.results.append({
                'test_name': test_name,
                'status': 'PASS',
                'duration': duration,
                'error': None
            })
            self.logger.info(f"✅ {test_name} - PASSED ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append({
                'test_name': test_name,
                'status': 'FAIL',
                'duration': duration,
                'error': str(e)
            })
            self.logger.error(f"❌ {test_name} - FAILED ({duration:.2f}s): {e}")
    
    def test_health_endpoint(self):
        """Test basic health endpoint"""
        response = self.session.get(f"{self.base_url}/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data['status'] == 'healthy'
        assert 'timestamp' in health_data
        assert 'version' in health_data
    
    def test_authentication(self):
        """Test authentication is working"""
        # Test with valid API key
        response = self.session.get(f"{self.base_url}/api/status")
        assert response.status_code == 200
        
        # Test without API key
        session_no_auth = requests.Session()
        response = session_no_auth.get(f"{self.base_url}/api/status")
        assert response.status_code == 401
    
    def test_coordinator_status(self):
        """Test coordinator status endpoint"""
        response = self.session.get(f"{self.base_url}/api/coordinator/status")
        assert response.status_code == 200
        
        status_data = response.json()
        assert 'coordinator_id' in status_data
        assert 'agents' in status_data
        assert 'uptime' in status_data
        assert status_data['status'] == 'active'
    
    def test_agent_registration(self):
        """Test agent registration endpoint"""
        response = self.session.get(f"{self.base_url}/api/agents")
        assert response.status_code == 200
        
        agents_data = response.json()
        assert isinstance(agents_data, list)
        
        # Should have at least one agent type
        agent_types = [agent['type'] for agent in agents_data]
        expected_types = ['reconnaissance', 'scanner', 'osint']
        assert any(agent_type in expected_types for agent_type in agent_types)
    
    def test_knowledge_graph_connection(self):
        """Test knowledge graph connectivity"""
        response = self.session.get(f"{self.base_url}/api/graph/status")
        assert response.status_code == 200
        
        graph_data = response.json()
        assert 'connected' in graph_data
        assert graph_data['connected'] is True
        assert 'node_count' in graph_data
        assert 'edge_count' in graph_data
    
    def test_threat_intelligence_api(self):
        """Test threat intelligence API endpoints"""
        # Test search endpoint
        response = self.session.get(f"{self.base_url}/api/threat-intel/search?q=test")
        assert response.status_code == 200
        
        search_data = response.json()
        assert 'results' in search_data
        assert isinstance(search_data['results'], list)
    
    def test_report_generation(self):
        """Test report generation endpoint"""
        # Request a simple report
        report_request = {
            'type': 'summary',
            'format': 'json',
            'filters': {}
        }
        
        response = self.session.post(
            f"{self.base_url}/api/reports/generate",
            json=report_request
        )
        assert response.status_code == 202  # Accepted for async processing
        
        report_data = response.json()
        assert 'report_id' in report_data
        assert 'status' in report_data
    
    def test_websocket_connection(self):
        """Test WebSocket connection (basic connectivity)"""
        # This is a basic test - in practice you'd use a WebSocket client
        response = self.session.get(f"{self.base_url}/api/ws/status")
        assert response.status_code == 200
        
        ws_data = response.json()
        assert 'websocket_enabled' in ws_data
        assert ws_data['websocket_enabled'] is True
    
    def test_database_connections(self):
        """Test database connectivity"""
        response = self.session.get(f"{self.base_url}/api/system/databases")
        assert response.status_code == 200
        
        db_data = response.json()
        assert 'redis' in db_data
        assert 'neo4j' in db_data
        assert db_data['redis']['connected'] is True
        assert db_data['neo4j']['connected'] is True
    
    def test_performance_metrics(self):
        """Test performance metrics endpoint"""
        response = self.session.get(f"{self.base_url}/api/metrics")
        assert response.status_code == 200
        
        metrics_data = response.json()
        assert 'cpu_usage' in metrics_data
        assert 'memory_usage' in metrics_data
        assert 'request_count' in metrics_data
        assert 'response_time' in metrics_data
    
    def test_security_headers(self):
        """Test security headers are present"""
        response = self.session.get(f"{self.base_url}/health")
        
        # Check for security headers
        security_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security'
        ]
        
        for header in security_headers:
            assert header in response.headers, f"Missing security header: {header}"
    
    def test_rate_limiting(self):
        """Test rate limiting is working"""
        # Make multiple rapid requests
        rapid_requests = []
        for _ in range(10):
            response = self.session.get(f"{self.base_url}/health")
            rapid_requests.append(response.status_code)
        
        # All should succeed (health endpoint typically has higher limits)
        assert all(status == 200 for status in rapid_requests)
        
        # Check rate limit headers
        response = self.session.get(f"{self.base_url}/health")
        assert 'X-RateLimit-Limit' in response.headers
        assert 'X-RateLimit-Remaining' in response.headers
    
    def run_all_tests(self):
        """Run all smoke tests"""
        self.logger.info("Starting staging smoke tests...")
        
        # Define all tests
        tests = [
            ('Health Endpoint', self.test_health_endpoint),
            ('Authentication', self.test_authentication),
            ('Coordinator Status', self.test_coordinator_status),
            ('Agent Registration', self.test_agent_registration),
            ('Knowledge Graph Connection', self.test_knowledge_graph_connection),
            ('Threat Intelligence API', self.test_threat_intelligence_api),
            ('Report Generation', self.test_report_generation),
            ('WebSocket Connection', self.test_websocket_connection),
            ('Database Connections', self.test_database_connections),
            ('Performance Metrics', self.test_performance_metrics),
            ('Security Headers', self.test_security_headers),
            ('Rate Limiting', self.test_rate_limiting)
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Generate summary
        self.generate_summary()
        
        return self.all_tests_passed()
    
    def generate_summary(self):
        """Generate test summary"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        total_duration = sum(r['duration'] for r in self.results)
        
        self.logger.info("=" * 50)
        self.logger.info("SMOKE TEST SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Passed: {passed_tests}")
        self.logger.info(f"Failed: {failed_tests}")
        self.logger.info(f"Total Duration: {total_duration:.2f}s")
        self.logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            self.logger.info("\nFAILED TESTS:")
            for result in self.results:
                if result['status'] == 'FAIL':
                    self.logger.info(f"  - {result['test_name']}: {result['error']}")
    
    def all_tests_passed(self) -> bool:
        """Check if all tests passed"""
        return all(r['status'] == 'PASS' for r in self.results)
    
    def save_results(self, filepath: str):
        """Save test results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': len(self.results),
                    'passed': sum(1 for r in self.results if r['status'] == 'PASS'),
                    'failed': sum(1 for r in self.results if r['status'] == 'FAIL'),
                    'total_duration': sum(r['duration'] for r in self.results)
                },
                'results': self.results
            }, f, indent=2)


def main():
    """Main entry point"""
    # Get configuration from environment
    staging_url = os.getenv('STAGING_URL')
    api_key = os.getenv('API_KEY')
    
    if not staging_url or not api_key:
        print("Error: STAGING_URL and API_KEY environment variables are required")
        sys.exit(1)
    
    # Run smoke tests
    smoke_test = StagingSmokeTest(staging_url, api_key)
    success = smoke_test.run_all_tests()
    
    # Save results
    smoke_test.save_results('staging-smoke-test-results.json')
    
    if success:
        print("\n✅ All staging smoke tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some staging smoke tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()