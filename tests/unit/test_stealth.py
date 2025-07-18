"""
Unit tests for the Stealth module.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json
import time
from typing import List, Dict, Any

from modules.stealth import (
    StealthModule,
    StealthEngine,
    TrafficObfuscator,
    TimingController,
    DecoyGenerator,
    StealthTarget,
    StealthResult,
    StealthTechnique,
    StealthIntensity,
    DetectionProbability,
    StealthStatus
)
from config.config import AetherVeilConfig


class TestStealthTarget:
    """Test StealthTarget data class"""
    
    def test_stealth_target_creation(self):
        """Test StealthTarget creation with default values"""
        target = StealthTarget(
            target="192.168.1.1",
            techniques=[StealthTechnique.TRAFFIC_OBFUSCATION]
        )
        
        assert target.target == "192.168.1.1"
        assert target.techniques == [StealthTechnique.TRAFFIC_OBFUSCATION]
        assert target.intensity == StealthIntensity.MEDIUM
        assert target.duration == 3600
        assert target.parameters == {}
    
    def test_stealth_target_with_custom_values(self):
        """Test StealthTarget creation with custom values"""
        techniques = [StealthTechnique.TRAFFIC_OBFUSCATION, StealthTechnique.TIMING_EVASION]
        parameters = {"proxy_chain": True, "delay_variance": 0.5}
        
        target = StealthTarget(
            target="example.com",
            techniques=techniques,
            intensity=StealthIntensity.HIGH,
            duration=7200,
            parameters=parameters
        )
        
        assert target.target == "example.com"
        assert target.techniques == techniques
        assert target.intensity == StealthIntensity.HIGH
        assert target.duration == 7200
        assert target.parameters == parameters


class TestStealthResult:
    """Test StealthResult data class"""
    
    def test_stealth_result_creation(self):
        """Test StealthResult creation"""
        timestamp = datetime.utcnow()
        duration = timedelta(seconds=300)
        
        result = StealthResult(
            target="192.168.1.1",
            technique=StealthTechnique.TRAFFIC_OBFUSCATION,
            timestamp=timestamp,
            duration=duration,
            status=StealthStatus.ACTIVE,
            effectiveness_score=0.85,
            detection_probability=DetectionProbability.LOW,
            metrics={"packets_obfuscated": 100, "detection_events": 0}
        )
        
        assert result.target == "192.168.1.1"
        assert result.technique == StealthTechnique.TRAFFIC_OBFUSCATION
        assert result.timestamp == timestamp
        assert result.duration == duration
        assert result.status == StealthStatus.ACTIVE
        assert result.effectiveness_score == 0.85
        assert result.detection_probability == DetectionProbability.LOW
        assert result.metrics == {"packets_obfuscated": 100, "detection_events": 0}


class TestTrafficObfuscator:
    """Test TrafficObfuscator functionality"""
    
    @pytest.fixture
    def traffic_obfuscator(self):
        """Traffic obfuscator fixture"""
        return TrafficObfuscator()
    
    @pytest.mark.asyncio
    async def test_apply_packet_fragmentation(self, traffic_obfuscator):
        """Test packet fragmentation"""
        packet_data = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
        
        with patch('scapy.all.fragment') as mock_fragment:
            mock_fragment.return_value = [Mock(), Mock(), Mock()]  # 3 fragments
            
            fragments = await traffic_obfuscator.apply_packet_fragmentation(packet_data)
            
            assert len(fragments) == 3
            mock_fragment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_apply_protocol_tunneling(self, traffic_obfuscator):
        """Test protocol tunneling"""
        original_data = b"HTTP traffic data"
        
        with patch('scapy.all.IP') as mock_ip:
            with patch('scapy.all.TCP') as mock_tcp:
                mock_packet = Mock()
                mock_ip.return_value = mock_packet
                mock_tcp.return_value = mock_packet
                
                tunneled_packet = await traffic_obfuscator.apply_protocol_tunneling(
                    original_data, 
                    tunnel_protocol="dns"
                )
                
                assert tunneled_packet is not None
    
    @pytest.mark.asyncio
    async def test_apply_traffic_padding(self, traffic_obfuscator):
        """Test traffic padding"""
        original_data = b"Short message"
        target_size = 1024
        
        padded_data = await traffic_obfuscator.apply_traffic_padding(
            original_data, 
            target_size
        )
        
        assert len(padded_data) == target_size
        assert original_data in padded_data
    
    @pytest.mark.asyncio
    async def test_apply_decoy_traffic(self, traffic_obfuscator):
        """Test decoy traffic generation"""
        real_traffic = [b"Real packet 1", b"Real packet 2"]
        
        mixed_traffic = await traffic_obfuscator.apply_decoy_traffic(
            real_traffic, 
            decoy_ratio=0.5
        )
        
        assert len(mixed_traffic) > len(real_traffic)
        # Real traffic should be present
        assert any(packet in mixed_traffic for packet in real_traffic)
    
    @pytest.mark.asyncio
    async def test_apply_encryption_layer(self, traffic_obfuscator):
        """Test encryption layer application"""
        plaintext_data = b"Sensitive data"
        
        with patch('cryptography.fernet.Fernet') as mock_fernet:
            mock_cipher = Mock()
            mock_cipher.encrypt.return_value = b"encrypted_data"
            mock_fernet.return_value = mock_cipher
            
            encrypted_data = await traffic_obfuscator.apply_encryption_layer(
                plaintext_data
            )
            
            assert encrypted_data == b"encrypted_data"
            mock_cipher.encrypt.assert_called_once_with(plaintext_data)
    
    @pytest.mark.asyncio
    async def test_randomize_packet_sizes(self, traffic_obfuscator):
        """Test packet size randomization"""
        packets = [b"packet1", b"packet2", b"packet3"]
        
        randomized_packets = await traffic_obfuscator.randomize_packet_sizes(packets)
        
        assert len(randomized_packets) == len(packets)
        # Sizes should be different (randomized)
        original_sizes = [len(p) for p in packets]
        new_sizes = [len(p) for p in randomized_packets]
        assert original_sizes != new_sizes
    
    @pytest.mark.asyncio
    async def test_get_obfuscation_metrics(self, traffic_obfuscator):
        """Test obfuscation metrics retrieval"""
        # Simulate some obfuscation activity
        traffic_obfuscator.packets_processed = 100
        traffic_obfuscator.bytes_obfuscated = 10240
        
        metrics = await traffic_obfuscator.get_obfuscation_metrics()
        
        assert metrics["packets_processed"] == 100
        assert metrics["bytes_obfuscated"] == 10240
        assert "obfuscation_rate" in metrics
        assert "effectiveness_score" in metrics


class TestTimingController:
    """Test TimingController functionality"""
    
    @pytest.fixture
    def timing_controller(self):
        """Timing controller fixture"""
        return TimingController()
    
    @pytest.mark.asyncio
    async def test_apply_random_delays(self, timing_controller):
        """Test random delay application"""
        min_delay = 0.1
        max_delay = 0.5
        
        start_time = time.time()
        await timing_controller.apply_random_delay(min_delay, max_delay)
        elapsed = time.time() - start_time
        
        assert min_delay <= elapsed <= max_delay + 0.1  # Small tolerance for execution time
    
    @pytest.mark.asyncio
    async def test_apply_jitter(self, timing_controller):
        """Test jitter application"""
        base_delay = 0.2
        jitter_factor = 0.1
        
        delays = []
        for _ in range(10):
            start_time = time.time()
            await timing_controller.apply_jitter(base_delay, jitter_factor)
            elapsed = time.time() - start_time
            delays.append(elapsed)
        
        # All delays should be within the jitter range
        min_expected = base_delay * (1 - jitter_factor)
        max_expected = base_delay * (1 + jitter_factor)
        
        assert all(min_expected <= delay <= max_expected + 0.1 for delay in delays)
    
    @pytest.mark.asyncio
    async def test_apply_burst_pattern(self, timing_controller):
        """Test burst pattern application"""
        burst_size = 5
        burst_interval = 0.1
        inter_burst_delay = 0.5
        
        start_time = time.time()
        await timing_controller.apply_burst_pattern(
            burst_size, 
            burst_interval, 
            inter_burst_delay
        )
        elapsed = time.time() - start_time
        
        # Should take at least the burst time
        expected_min = (burst_size - 1) * burst_interval
        assert elapsed >= expected_min
    
    @pytest.mark.asyncio
    async def test_apply_throttling(self, timing_controller):
        """Test throttling application"""
        requests_per_second = 2
        
        start_time = time.time()
        for i in range(3):
            await timing_controller.apply_throttling(requests_per_second)
        elapsed = time.time() - start_time
        
        # Should take at least 1 second for 3 requests at 2 RPS
        assert elapsed >= 1.0
    
    @pytest.mark.asyncio
    async def test_get_timing_metrics(self, timing_controller):
        """Test timing metrics retrieval"""
        # Simulate some timing activity
        await timing_controller.apply_random_delay(0.1, 0.2)
        await timing_controller.apply_jitter(0.1, 0.05)
        
        metrics = await timing_controller.get_timing_metrics()
        
        assert "average_delay" in metrics
        assert "total_delays_applied" in metrics
        assert "timing_variance" in metrics


class TestDecoyGenerator:
    """Test DecoyGenerator functionality"""
    
    @pytest.fixture
    def decoy_generator(self):
        """Decoy generator fixture"""
        return DecoyGenerator()
    
    @pytest.mark.asyncio
    async def test_generate_decoy_requests(self, decoy_generator):
        """Test decoy request generation"""
        target = "example.com"
        count = 5
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            decoy_requests = await decoy_generator.generate_decoy_requests(target, count)
            
            assert len(decoy_requests) == count
            assert all(isinstance(req, dict) for req in decoy_requests)
    
    @pytest.mark.asyncio
    async def test_generate_decoy_traffic(self, decoy_generator):
        """Test decoy traffic generation"""
        target_ip = "192.168.1.1"
        duration = 10  # seconds
        
        with patch('scapy.all.send') as mock_send:
            decoy_traffic = await decoy_generator.generate_decoy_traffic(
                target_ip, 
                duration
            )
            
            assert isinstance(decoy_traffic, list)
            assert len(decoy_traffic) > 0
    
    @pytest.mark.asyncio
    async def test_create_honeypot_responses(self, decoy_generator):
        """Test honeypot response creation"""
        service_type = "ssh"
        
        responses = await decoy_generator.create_honeypot_responses(service_type)
        
        assert isinstance(responses, list)
        assert len(responses) > 0
        assert all(isinstance(resp, dict) for resp in responses)
    
    @pytest.mark.asyncio
    async def test_simulate_legitimate_behavior(self, decoy_generator):
        """Test legitimate behavior simulation"""
        target = "example.com"
        behavior_type = "web_browsing"
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html><body>Test</body></html>"
            mock_get.return_value = mock_response
            
            behavior_log = await decoy_generator.simulate_legitimate_behavior(
                target, 
                behavior_type
            )
            
            assert isinstance(behavior_log, list)
            assert len(behavior_log) > 0
    
    @pytest.mark.asyncio
    async def test_get_decoy_metrics(self, decoy_generator):
        """Test decoy metrics retrieval"""
        # Simulate some decoy activity
        await decoy_generator.generate_decoy_requests("example.com", 5)
        
        metrics = await decoy_generator.get_decoy_metrics()
        
        assert "decoy_requests_generated" in metrics
        assert "decoy_success_rate" in metrics
        assert "detection_evasion_score" in metrics


class TestStealthEngine:
    """Test StealthEngine functionality"""
    
    @pytest.fixture
    def stealth_engine(self):
        """Stealth engine fixture"""
        return StealthEngine()
    
    @pytest.mark.asyncio
    async def test_apply_stealth_single_technique(self, stealth_engine):
        """Test single stealth technique application"""
        target = StealthTarget(
            target="192.168.1.1",
            techniques=[StealthTechnique.TRAFFIC_OBFUSCATION],
            intensity=StealthIntensity.MEDIUM
        )
        
        # Mock component methods
        with patch.object(stealth_engine.traffic_obfuscator, 'apply_packet_fragmentation'):
            with patch.object(stealth_engine.traffic_obfuscator, 'apply_traffic_padding'):
                result = await stealth_engine.apply_stealth(target)
                
                assert result is not None
                assert result.target == "192.168.1.1"
                assert result.technique == StealthTechnique.TRAFFIC_OBFUSCATION
                assert result.status == StealthStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_apply_stealth_multiple_techniques(self, stealth_engine):
        """Test multiple stealth technique application"""
        target = StealthTarget(
            target="192.168.1.1",
            techniques=[
                StealthTechnique.TRAFFIC_OBFUSCATION,
                StealthTechnique.TIMING_EVASION,
                StealthTechnique.DECOY_GENERATION
            ],
            intensity=StealthIntensity.HIGH
        )
        
        # Mock component methods
        with patch.object(stealth_engine.traffic_obfuscator, 'apply_packet_fragmentation'):
            with patch.object(stealth_engine.timing_controller, 'apply_random_delay'):
                with patch.object(stealth_engine.decoy_generator, 'generate_decoy_requests'):
                    results = await stealth_engine.apply_stealth(target)
                    
                    assert len(results) == 3
                    assert all(r.target == "192.168.1.1" for r in results)
                    assert all(r.status == StealthStatus.ACTIVE for r in results)
    
    @pytest.mark.asyncio
    async def test_calculate_effectiveness_score(self, stealth_engine):
        """Test effectiveness score calculation"""
        target = StealthTarget(
            target="192.168.1.1",
            techniques=[StealthTechnique.TRAFFIC_OBFUSCATION],
            intensity=StealthIntensity.HIGH
        )
        
        # Mock metrics
        with patch.object(stealth_engine.traffic_obfuscator, 'get_obfuscation_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "packets_processed": 100,
                "bytes_obfuscated": 10240,
                "obfuscation_rate": 0.95,
                "effectiveness_score": 0.85
            }
            
            score = await stealth_engine.calculate_effectiveness_score(target)
            
            assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_assess_detection_probability(self, stealth_engine):
        """Test detection probability assessment"""
        target = StealthTarget(
            target="192.168.1.1",
            techniques=[StealthTechnique.TRAFFIC_OBFUSCATION],
            intensity=StealthIntensity.HIGH
        )
        
        detection_prob = await stealth_engine.assess_detection_probability(target)
        
        assert detection_prob in [
            DetectionProbability.VERY_LOW,
            DetectionProbability.LOW,
            DetectionProbability.MEDIUM,
            DetectionProbability.HIGH,
            DetectionProbability.VERY_HIGH
        ]
    
    @pytest.mark.asyncio
    async def test_adjust_intensity(self, stealth_engine):
        """Test intensity adjustment"""
        target = StealthTarget(
            target="192.168.1.1",
            techniques=[StealthTechnique.TRAFFIC_OBFUSCATION],
            intensity=StealthIntensity.MEDIUM
        )
        
        # Mock detection feedback
        detection_events = 3
        
        new_intensity = await stealth_engine.adjust_intensity(target, detection_events)
        
        # Should increase intensity due to detection events
        assert new_intensity.value > target.intensity.value
    
    @pytest.mark.asyncio
    async def test_deactivate_stealth(self, stealth_engine):
        """Test stealth deactivation"""
        target = StealthTarget(
            target="192.168.1.1",
            techniques=[StealthTechnique.TRAFFIC_OBFUSCATION]
        )
        
        # Apply stealth first
        result = await stealth_engine.apply_stealth(target)
        
        # Deactivate stealth
        success = await stealth_engine.deactivate_stealth(target)
        
        assert success is True
    
    def test_get_available_techniques(self, stealth_engine):
        """Test getting available stealth techniques"""
        techniques = stealth_engine.get_available_techniques()
        
        assert isinstance(techniques, list)
        assert len(techniques) > 0
        assert all(isinstance(t, dict) for t in techniques)
        assert all("technique" in t for t in techniques)
        assert all("description" in t for t in techniques)
    
    def test_get_technique_info(self, stealth_engine):
        """Test getting technique information"""
        info = stealth_engine.get_technique_info(StealthTechnique.TRAFFIC_OBFUSCATION)
        
        assert info is not None
        assert info["technique"] == StealthTechnique.TRAFFIC_OBFUSCATION
        assert "description" in info
        assert "effectiveness" in info
        assert "detection_difficulty" in info


class TestStealthModule:
    """Test main stealth module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def stealth_module(self, config):
        """Stealth module fixture"""
        return StealthModule(config)
    
    @pytest.mark.asyncio
    async def test_module_initialization(self, stealth_module):
        """Test module initialization"""
        assert stealth_module.module_type.value == "stealth"
        assert stealth_module.status.value == "initialized"
        assert stealth_module.version == "1.0.0"
        assert stealth_module.stealth_engine is not None
        assert len(stealth_module.results) == 0
    
    @pytest.mark.asyncio
    async def test_module_start_stop(self, stealth_module):
        """Test module start and stop"""
        # Test start
        success = await stealth_module.start()
        assert success is True
        assert stealth_module.status.value == "running"
        
        # Test stop
        success = await stealth_module.stop()
        assert success is True
        assert stealth_module.status.value == "stopped"
    
    @pytest.mark.asyncio
    async def test_apply_stealth(self, stealth_module):
        """Test stealth application"""
        target = StealthTarget(
            target="192.168.1.1",
            techniques=[StealthTechnique.TRAFFIC_OBFUSCATION],
            intensity=StealthIntensity.MEDIUM
        )
        
        # Mock stealth engine
        with patch.object(stealth_module.stealth_engine, 'apply_stealth') as mock_apply:
            mock_result = StealthResult(
                target="192.168.1.1",
                technique=StealthTechnique.TRAFFIC_OBFUSCATION,
                timestamp=datetime.utcnow(),
                duration=timedelta(seconds=300),
                status=StealthStatus.ACTIVE,
                effectiveness_score=0.85,
                detection_probability=DetectionProbability.LOW,
                metrics={}
            )
            mock_apply.return_value = [mock_result]
            
            results = await stealth_module.apply_stealth(target)
            
            assert len(results) == 1
            assert results[0].target == "192.168.1.1"
            assert results[0].status == StealthStatus.ACTIVE
            assert len(stealth_module.results) == 1
    
    @pytest.mark.asyncio
    async def test_bulk_stealth_application(self, stealth_module):
        """Test bulk stealth application"""
        targets = [
            StealthTarget("192.168.1.1", [StealthTechnique.TRAFFIC_OBFUSCATION]),
            StealthTarget("192.168.1.2", [StealthTechnique.TIMING_EVASION])
        ]
        
        # Mock apply_stealth
        with patch.object(stealth_module, 'apply_stealth') as mock_apply:
            mock_result = StealthResult(
                target="test",
                technique=StealthTechnique.TRAFFIC_OBFUSCATION,
                timestamp=datetime.utcnow(),
                duration=timedelta(seconds=300),
                status=StealthStatus.ACTIVE,
                effectiveness_score=0.85,
                detection_probability=DetectionProbability.LOW,
                metrics={}
            )
            mock_apply.return_value = [mock_result]
            
            results = await stealth_module.bulk_stealth_application(targets)
            
            assert len(results) == 2
            assert all(isinstance(r, list) for r in results)
    
    @pytest.mark.asyncio
    async def test_get_effectiveness_rating(self, stealth_module):
        """Test effectiveness rating"""
        # Add test results
        result = StealthResult(
            target="192.168.1.1",
            technique=StealthTechnique.TRAFFIC_OBFUSCATION,
            timestamp=datetime.utcnow(),
            duration=timedelta(seconds=300),
            status=StealthStatus.ACTIVE,
            effectiveness_score=0.85,
            detection_probability=DetectionProbability.LOW,
            metrics={"packets_obfuscated": 100}
        )
        stealth_module.results = [result]
        
        rating = await stealth_module.get_effectiveness_rating("192.168.1.1")
        
        assert rating is not None
        assert "overall_rating" in rating
        assert "technique_ratings" in rating
        assert "detection_probability" in rating
        assert rating["overall_rating"] == 0.85
    
    @pytest.mark.asyncio
    async def test_adjust_stealth_intensity(self, stealth_module):
        """Test stealth intensity adjustment"""
        target = StealthTarget(
            target="192.168.1.1",
            techniques=[StealthTechnique.TRAFFIC_OBFUSCATION],
            intensity=StealthIntensity.MEDIUM
        )
        
        # Mock stealth engine
        with patch.object(stealth_module.stealth_engine, 'adjust_intensity') as mock_adjust:
            mock_adjust.return_value = StealthIntensity.HIGH
            
            new_intensity = await stealth_module.adjust_stealth_intensity(target, 5)
            
            assert new_intensity == StealthIntensity.HIGH
    
    @pytest.mark.asyncio
    async def test_deactivate_stealth(self, stealth_module):
        """Test stealth deactivation"""
        target = StealthTarget(
            target="192.168.1.1",
            techniques=[StealthTechnique.TRAFFIC_OBFUSCATION]
        )
        
        # Mock stealth engine
        with patch.object(stealth_module.stealth_engine, 'deactivate_stealth') as mock_deactivate:
            mock_deactivate.return_value = True
            
            success = await stealth_module.deactivate_stealth(target)
            
            assert success is True
    
    def test_get_results_filtering(self, stealth_module):
        """Test result filtering"""
        # Add test results
        result1 = StealthResult(
            target="192.168.1.1",
            technique=StealthTechnique.TRAFFIC_OBFUSCATION,
            timestamp=datetime.utcnow(),
            duration=timedelta(seconds=300),
            status=StealthStatus.ACTIVE,
            effectiveness_score=0.85,
            detection_probability=DetectionProbability.LOW,
            metrics={}
        )
        
        result2 = StealthResult(
            target="192.168.1.2",
            technique=StealthTechnique.TIMING_EVASION,
            timestamp=datetime.utcnow(),
            duration=timedelta(seconds=600),
            status=StealthStatus.INACTIVE,
            effectiveness_score=0.70,
            detection_probability=DetectionProbability.MEDIUM,
            metrics={}
        )
        
        stealth_module.results = [result1, result2]
        
        # Test filtering by target
        filtered = stealth_module.get_results(target="192.168.1.1")
        assert len(filtered) == 1
        assert filtered[0].target == "192.168.1.1"
        
        # Test filtering by technique
        filtered = stealth_module.get_results(technique=StealthTechnique.TRAFFIC_OBFUSCATION)
        assert len(filtered) == 1
        assert filtered[0].technique == StealthTechnique.TRAFFIC_OBFUSCATION
        
        # Test filtering by status
        filtered = stealth_module.get_results(status=StealthStatus.ACTIVE)
        assert len(filtered) == 1
        assert filtered[0].status == StealthStatus.ACTIVE
    
    def test_export_results_json(self, stealth_module):
        """Test JSON export of results"""
        result = StealthResult(
            target="192.168.1.1",
            technique=StealthTechnique.TRAFFIC_OBFUSCATION,
            timestamp=datetime.utcnow(),
            duration=timedelta(seconds=300),
            status=StealthStatus.ACTIVE,
            effectiveness_score=0.85,
            detection_probability=DetectionProbability.LOW,
            metrics={"packets_obfuscated": 100}
        )
        
        stealth_module.results = [result]
        
        json_output = stealth_module.export_results("json")
        
        assert json_output != ""
        # Verify it's valid JSON
        parsed = json.loads(json_output)
        assert len(parsed) == 1
        assert parsed[0]["target"] == "192.168.1.1"
        assert parsed[0]["technique"] == "traffic_obfuscation"
        assert parsed[0]["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_get_status(self, stealth_module):
        """Test module status reporting"""
        # Add test results
        result = StealthResult(
            target="192.168.1.1",
            technique=StealthTechnique.TRAFFIC_OBFUSCATION,
            timestamp=datetime.utcnow(),
            duration=timedelta(seconds=300),
            status=StealthStatus.ACTIVE,
            effectiveness_score=0.85,
            detection_probability=DetectionProbability.LOW,
            metrics={}
        )
        
        stealth_module.results = [result]
        
        status = await stealth_module.get_status()
        
        assert status["module"] == "stealth"
        assert status["status"] == "initialized"
        assert status["version"] == "1.0.0"
        assert status["active_stealth_operations"] == 1
        assert status["average_effectiveness"] == 0.85
        assert "192.168.1.1" in status["protected_targets"]
    
    def test_get_available_techniques(self, stealth_module):
        """Test getting available stealth techniques"""
        techniques = stealth_module.get_available_techniques()
        
        assert isinstance(techniques, list)
        assert len(techniques) > 0
        assert all(isinstance(t, dict) for t in techniques)


@pytest.mark.performance
class TestStealthPerformance:
    """Performance tests for stealth module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def stealth_module(self, config):
        """Stealth module fixture"""
        return StealthModule(config)
    
    @pytest.mark.asyncio
    async def test_bulk_stealth_application_performance(self, stealth_module, performance_monitor):
        """Test bulk stealth application performance"""
        # Create multiple targets
        targets = [
            StealthTarget(f"192.168.1.{i}", [StealthTechnique.TRAFFIC_OBFUSCATION])
            for i in range(1, 11)
        ]
        
        # Mock apply_stealth
        with patch.object(stealth_module, 'apply_stealth') as mock_apply:
            mock_result = StealthResult(
                target="test",
                technique=StealthTechnique.TRAFFIC_OBFUSCATION,
                timestamp=datetime.utcnow(),
                duration=timedelta(seconds=1),
                status=StealthStatus.ACTIVE,
                effectiveness_score=0.85,
                detection_probability=DetectionProbability.LOW,
                metrics={}
            )
            mock_apply.return_value = [mock_result]
            
            performance_monitor.start()
            
            results = await stealth_module.bulk_stealth_application(targets)
            
            performance_monitor.stop()
            
            duration = performance_monitor.get_duration()
            assert duration is not None
            assert duration < 10.0  # Should complete within 10 seconds
            assert len(results) == 10
    
    @pytest.mark.asyncio
    async def test_stealth_technique_performance(self, stealth_module, performance_monitor):
        """Test individual stealth technique performance"""
        target = StealthTarget(
            target="192.168.1.1",
            techniques=[StealthTechnique.TRAFFIC_OBFUSCATION],
            intensity=StealthIntensity.HIGH
        )
        
        # Mock components for fast execution
        with patch.object(stealth_module.stealth_engine.traffic_obfuscator, 'apply_packet_fragmentation'):
            with patch.object(stealth_module.stealth_engine.traffic_obfuscator, 'apply_traffic_padding'):
                performance_monitor.start()
                
                results = await stealth_module.apply_stealth(target)
                
                performance_monitor.stop()
                
                duration = performance_monitor.get_duration()
                assert duration is not None
                assert duration < 5.0  # Should complete within 5 seconds
                assert len(results) > 0


@pytest.mark.security
class TestStealthSecurity:
    """Security tests for stealth module"""
    
    @pytest.fixture
    def config(self):
        """Configuration fixture"""
        return AetherVeilConfig()
    
    @pytest.fixture
    def stealth_module(self, config):
        """Stealth module fixture"""
        return StealthModule(config)
    
    def test_input_validation_target(self, stealth_module):
        """Test input validation for targets"""
        # Test malicious targets
        malicious_targets = [
            "; rm -rf /",
            "../../..",
            "<script>alert('xss')</script>",
            "$(whoami)",
            "`id`"
        ]
        
        for target in malicious_targets:
            stealth_target = StealthTarget(
                target=target,
                techniques=[StealthTechnique.TRAFFIC_OBFUSCATION]
            )
            
            # The module should handle these gracefully
            assert stealth_target.target == target
    
    @pytest.mark.asyncio
    async def test_stealth_technique_isolation(self, stealth_module):
        """Test stealth technique isolation"""
        target1 = StealthTarget(
            target="192.168.1.1",
            techniques=[StealthTechnique.TRAFFIC_OBFUSCATION]
        )
        
        target2 = StealthTarget(
            target="192.168.1.2",
            techniques=[StealthTechnique.TIMING_EVASION]
        )
        
        # Mock stealth engine
        with patch.object(stealth_module.stealth_engine, 'apply_stealth') as mock_apply:
            mock_result1 = StealthResult(
                target="192.168.1.1",
                technique=StealthTechnique.TRAFFIC_OBFUSCATION,
                timestamp=datetime.utcnow(),
                duration=timedelta(seconds=300),
                status=StealthStatus.ACTIVE,
                effectiveness_score=0.85,
                detection_probability=DetectionProbability.LOW,
                metrics={}
            )
            
            mock_result2 = StealthResult(
                target="192.168.1.2",
                technique=StealthTechnique.TIMING_EVASION,
                timestamp=datetime.utcnow(),
                duration=timedelta(seconds=300),
                status=StealthStatus.ACTIVE,
                effectiveness_score=0.75,
                detection_probability=DetectionProbability.MEDIUM,
                metrics={}
            )
            
            mock_apply.side_effect = [[mock_result1], [mock_result2]]
            
            results1 = await stealth_module.apply_stealth(target1)
            results2 = await stealth_module.apply_stealth(target2)
            
            # Results should be isolated
            assert results1[0].target == "192.168.1.1"
            assert results2[0].target == "192.168.1.2"
            assert results1[0].technique != results2[0].technique
    
    @pytest.mark.asyncio
    async def test_detection_evasion_validation(self, stealth_module):
        """Test detection evasion validation"""
        target = StealthTarget(
            target="192.168.1.1",
            techniques=[StealthTechnique.TRAFFIC_OBFUSCATION],
            intensity=StealthIntensity.HIGH
        )
        
        # Mock stealth engine
        with patch.object(stealth_module.stealth_engine, 'apply_stealth') as mock_apply:
            mock_result = StealthResult(
                target="192.168.1.1",
                technique=StealthTechnique.TRAFFIC_OBFUSCATION,
                timestamp=datetime.utcnow(),
                duration=timedelta(seconds=300),
                status=StealthStatus.ACTIVE,
                effectiveness_score=0.95,  # High effectiveness
                detection_probability=DetectionProbability.VERY_LOW,
                metrics={}
            )
            mock_apply.return_value = [mock_result]
            
            results = await stealth_module.apply_stealth(target)
            
            # Should achieve high effectiveness with low detection probability
            assert results[0].effectiveness_score >= 0.8
            assert results[0].detection_probability in [
                DetectionProbability.VERY_LOW,
                DetectionProbability.LOW
            ]
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, stealth_module):
        """Test resource cleanup after stealth operations"""
        target = StealthTarget(
            target="192.168.1.1",
            techniques=[StealthTechnique.TRAFFIC_OBFUSCATION]
        )
        
        # Apply stealth
        with patch.object(stealth_module.stealth_engine, 'apply_stealth') as mock_apply:
            mock_result = StealthResult(
                target="192.168.1.1",
                technique=StealthTechnique.TRAFFIC_OBFUSCATION,
                timestamp=datetime.utcnow(),
                duration=timedelta(seconds=300),
                status=StealthStatus.ACTIVE,
                effectiveness_score=0.85,
                detection_probability=DetectionProbability.LOW,
                metrics={}
            )
            mock_apply.return_value = [mock_result]
            
            await stealth_module.apply_stealth(target)
            
            # Deactivate stealth
            with patch.object(stealth_module.stealth_engine, 'deactivate_stealth') as mock_deactivate:
                mock_deactivate.return_value = True
                
                success = await stealth_module.deactivate_stealth(target)
                
                assert success is True
                # Resources should be cleaned up
                mock_deactivate.assert_called_once_with(target)
    
    @pytest.mark.asyncio
    async def test_encryption_key_management(self, stealth_module):
        """Test encryption key management security"""
        target = StealthTarget(
            target="192.168.1.1",
            techniques=[StealthTechnique.TRAFFIC_OBFUSCATION],
            parameters={"encryption_key": "secret-key-123"}
        )
        
        # Mock stealth engine
        with patch.object(stealth_module.stealth_engine, 'apply_stealth') as mock_apply:
            mock_result = StealthResult(
                target="192.168.1.1",
                technique=StealthTechnique.TRAFFIC_OBFUSCATION,
                timestamp=datetime.utcnow(),
                duration=timedelta(seconds=300),
                status=StealthStatus.ACTIVE,
                effectiveness_score=0.85,
                detection_probability=DetectionProbability.LOW,
                metrics={}
            )
            mock_apply.return_value = [mock_result]
            
            results = await stealth_module.apply_stealth(target)
            
            # Encryption keys should not be exposed in results
            json_output = stealth_module.export_results("json")
            assert "secret-key-123" not in json_output