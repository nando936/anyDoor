"""
Door Configuration System
This module loads and manages the modular door configuration system
including construction methods, materials, hardware, and profiles.

This is a PARALLEL SYSTEM that doesn't affect existing code.
It can be tested independently before integration.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Configuration base directory
CONFIG_DIR = Path(__file__).parent / "door_configuration"

@dataclass
class DoorConfiguration:
    """Complete configuration for a door style"""
    style_number: str
    style_data: Dict[str, Any]
    construction_method: Dict[str, Any]
    material_schedule: Dict[str, Any]
    hardware_schedule: Dict[str, Any]
    profiles: Dict[str, Dict[str, Any]]
    
    def get_stile_width(self) -> float:
        """Get the stile width for this door style"""
        return self.style_data['dimensions']['stile_width']
    
    def get_rail_width(self) -> float:
        """Get the rail width for this door style"""
        return self.style_data['dimensions']['rail_width']
    
    def get_oversize(self, component: str, dimension: str) -> float:
        """Get oversize amount for a component"""
        overrides = self.style_data.get('style_specific_overrides', {})
        if component == 'stile' and dimension == 'length':
            return overrides.get('stile_length_oversize', 0)
        elif component == 'stile' and dimension == 'width':
            return overrides.get('stile_width_oversize', 0)
        return 0
    
    def calculate_stile_length(self, door_height: float) -> float:
        """Calculate stile length based on door height"""
        oversize = self.get_oversize('stile', 'length')
        return door_height + oversize
    
    def calculate_rail_length(self, door_width: float) -> float:
        """Calculate rail length based on door width and construction method"""
        method = self.construction_method['method_id']
        if method == 'cope_and_stick':
            stile_width = self.get_stile_width()
            stick_depth = self.style_data['style_specific_overrides'].get('stick_depth', 0.5)
            return door_width - ((2 * stile_width) + (2 * stick_depth))
        elif method == 'mitre_cut':
            return door_width
        return door_width
    
    def calculate_panel_dimensions(self, door_width: float, door_height: float) -> tuple:
        """Calculate panel dimensions based on construction method"""
        method = self.construction_method['method_id']
        stick_depth = self.style_data['style_specific_overrides'].get('stick_depth', 0.5)
        
        if method == 'cope_and_stick':
            stile_width = self.get_stile_width()
            panel_width = door_width - (stile_width + ((stick_depth * 2) - 0.125))
            panel_height = door_height - (stile_width + ((stick_depth * 2) - 0.125))
        elif method == 'mitre_cut':
            stile_width = self.get_stile_width()
            rail_width = self.get_rail_width()
            panel_width = door_width - (2 * stile_width) + (stick_depth - 0.125)
            panel_height = door_height - (2 * rail_width) + (stick_depth - 0.125)
        else:
            panel_width = door_width - 3
            panel_height = door_height - 3
            
        return panel_width, panel_height


class DoorConfigurationLoader:
    """Loads door configurations from JSON files"""
    
    def __init__(self, config_dir: Path = CONFIG_DIR):
        self.config_dir = config_dir
        self._cache = {}
    
    def load_json_file(self, filepath: Path) -> Dict[str, Any]:
        """Load a JSON configuration file"""
        if filepath in self._cache:
            return self._cache[filepath]
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._cache[filepath] = data
                return data
        except FileNotFoundError:
            print(f"[WARNING] Configuration file not found: {filepath}")
            return {}
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in {filepath}: {e}")
            return {}
    
    def load_construction_method(self, method_id: str) -> Dict[str, Any]:
        """Load a construction method configuration"""
        filepath = self.config_dir / "construction_methods" / f"{method_id}.json"
        return self.load_json_file(filepath)
    
    def load_material_schedule(self, schedule_id: str) -> Dict[str, Any]:
        """Load a material schedule configuration"""
        filepath = self.config_dir / "material_schedules" / f"{schedule_id}.json"
        return self.load_json_file(filepath)
    
    def load_hardware_schedule(self, schedule_id: str) -> Dict[str, Any]:
        """Load a hardware schedule configuration"""
        filepath = self.config_dir / "hardware_schedules" / f"{schedule_id}.json"
        return self.load_json_file(filepath)
    
    def load_profile(self, profile_type: str, profile_id: str) -> Dict[str, Any]:
        """Load a profile configuration"""
        if profile_type == "edges":
            filepath = self.config_dir / "profile_library" / "edges" / f"{profile_id}.json"
        elif profile_type == "panels":
            filepath = self.config_dir / "profile_library" / "panels" / f"{profile_id}.json"
        else:
            return {}
        return self.load_json_file(filepath)
    
    def load_door_style(self, style_number: str) -> Optional[DoorConfiguration]:
        """Load complete configuration for a door style"""
        # Load the door style definition
        style_filepath = self.config_dir / "door_styles" / f"{style_number}.json"
        style_data = self.load_json_file(style_filepath)
        
        if not style_data:
            print(f"[ERROR] Door style {style_number} not found")
            return None
        
        # Load all referenced components
        components = style_data.get('components', {})
        
        construction_method = self.load_construction_method(
            components.get('construction_method', '')
        )
        
        material_schedule = self.load_material_schedule(
            components.get('material_schedule', '')
        )
        
        hardware_schedule = self.load_hardware_schedule(
            components.get('hardware_schedule', '')
        )
        
        # Load profiles
        profiles = {}
        profile_refs = components.get('profiles', {})
        for profile_type, profile_id in profile_refs.items():
            if profile_type in ['outside_edge', 'inside_edge']:
                profiles[profile_type] = self.load_profile('edges', profile_id)
            elif profile_type == 'panel':
                profiles[profile_type] = self.load_profile('panels', profile_id)
        
        # Create configuration object
        return DoorConfiguration(
            style_number=style_number,
            style_data=style_data,
            construction_method=construction_method,
            material_schedule=material_schedule,
            hardware_schedule=hardware_schedule,
            profiles=profiles
        )
    
    def get_legacy_compatible_data(self, style_number: str) -> Dict[str, Any]:
        """
        Get configuration data in a format compatible with existing system.
        This allows gradual migration from text specs to JSON configs.
        """
        config = self.load_door_style(style_number)
        if not config:
            return {}
        
        # Format data to match what existing system expects
        return {
            'stile_width': config.get_stile_width(),
            'rail_width': config.get_rail_width(),
            'is_cope_and_stick': config.construction_method.get('method_id') == 'cope_and_stick',
            'is_mitre_cut': config.construction_method.get('method_id') == 'mitre_cut',
            'stile_length_oversize': config.get_oversize('stile', 'length'),
            'stile_width_oversize': config.get_oversize('stile', 'width'),
            'material_thickness': config.style_data['style_specific_overrides'].get('material_thickness', 0.8125),
            'stick_depth': config.style_data['style_specific_overrides'].get('stick_depth', 0.5),
            'raw_text': f"Generated from JSON config for style {style_number}"
        }


# Testing function
def test_configuration_system():
    """Test the configuration system with existing door styles"""
    print("=" * 60)
    print("TESTING DOOR CONFIGURATION SYSTEM")
    print("=" * 60)
    
    loader = DoorConfigurationLoader()
    
    # Test loading door style 103
    print("\nTesting Door Style 103:")
    print("-" * 40)
    config_103 = loader.load_door_style("103")
    if config_103:
        print(f"Style Name: {config_103.style_data['style_name']}")
        print(f"Construction: {config_103.construction_method['name']}")
        print(f"Material: {config_103.material_schedule['name']}")
        print(f"Stile Width: {config_103.get_stile_width()}")
        print(f"Rail Width: {config_103.get_rail_width()}")
        
        # Test calculations
        door_width, door_height = 15.5, 30.25
        print(f"\nTest door size: {door_width} x {door_height}")
        print(f"Stile Length: {config_103.calculate_stile_length(door_height)}")
        print(f"Rail Length: {config_103.calculate_rail_length(door_width)}")
        panel_w, panel_h = config_103.calculate_panel_dimensions(door_width, door_height)
        print(f"Panel Size: {panel_w:.3f} x {panel_h:.3f}")
    
    # Test loading door style 231
    print("\n\nTesting Door Style 231:")
    print("-" * 40)
    config_231 = loader.load_door_style("231")
    if config_231:
        print(f"Style Name: {config_231.style_data['style_name']}")
        print(f"Construction: {config_231.construction_method['name']}")
        print(f"Material: {config_231.material_schedule['name']}")
        print(f"Stile Width: {config_231.get_stile_width()}")
        print(f"Rail Width: {config_231.get_rail_width()}")
        
        # Test calculations
        print(f"\nTest door size: {door_width} x {door_height}")
        print(f"Stile Length: {config_231.calculate_stile_length(door_height)}")
        print(f"Rail Length: {config_231.calculate_rail_length(door_width)}")
        panel_w, panel_h = config_231.calculate_panel_dimensions(door_width, door_height)
        print(f"Panel Size: {panel_w:.3f} x {panel_h:.3f}")
    
    # Test legacy compatibility
    print("\n\nTesting Legacy Compatibility:")
    print("-" * 40)
    legacy_data = loader.get_legacy_compatible_data("103")
    print(f"Legacy format data: {json.dumps(legacy_data, indent=2)}")
    
    print("\n" + "=" * 60)
    print("Configuration system test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_configuration_system()