"""
Basic tests for Chimera components
"""

import pytest
import asyncio
from chimera.utils.config import ConfigManager
from chimera.core.events import EventSystem

class TestBasicFunctionality:
    """Test basic Chimera functionality"""
    
    def test_config_manager(self):
        """Test configuration manager"""
        config = ConfigManager()
        
        # Test default config
        assert config.get("core.max_concurrent_tasks", 1) == 1
        
        # Test setting values
        config.set("test.value", 42)
        assert config.get("test.value") == 42
        
    @pytest.mark.asyncio
    async def test_event_system(self):
        """Test event system"""
        event_system = EventSystem()
        await event_system.start()
        
        # Test basic event emission
        from chimera.core.events import EventType, ChimeraEvent
        from datetime import datetime
        
        event = ChimeraEvent(
            event_type=EventType.SYSTEM_STARTUP,
            source="test",
            data={"test": "data"},
            timestamp=datetime.utcnow()
        )
        
        await event_system.emit(event)
        
        # Wait a moment for processing
        await asyncio.sleep(0.1)
        
        stats = event_system.get_event_stats()
        assert stats["total_events"] >= 1
        
        await event_system.stop()
        
    def test_persona_config_loading(self):
        """Test persona configuration structure"""
        import yaml
        from pathlib import Path
        
        # Test that persona files exist and are valid
        persona_dir = Path("configs/personas")
        
        for persona_file in persona_dir.glob("*.yaml"):
            with open(persona_file, 'r') as f:
                persona_data = yaml.safe_load(f)
                
            # Check required fields
            assert "name" in persona_data
            assert "risk_tolerance" in persona_data
            assert "stealth_priority" in persona_data
            assert isinstance(persona_data["risk_tolerance"], (int, float))
            assert 0.0 <= persona_data["risk_tolerance"] <= 1.0

if __name__ == "__main__":
    pytest.main([__file__])