#!/usr/bin/env python3
"""
Simple script to test that the configuration file can be loaded successfully.
"""

import yaml
import os
from pathlib import Path

def test_config_loading():
    """Test loading the experiment configuration file."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "experiment_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Configuration loaded successfully!")
        print(f"📁 Config path: {config_path}")
        print(f"🎯 Random seed: {config['random_seed']}")
        print(f"🤖 Prompt encoder model: {config['prompt_encoder_model']}")
        print(f"📊 Min votes per user: {config['min_votes_per_user']}")
        
        return config
        
    except FileNotFoundError:
        print(f"❌ Configuration file not found at: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML file: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

if __name__ == "__main__":
    config = test_config_loading()
    if config:
        print("\n🎉 Project scaffolding setup complete!")
    else:
        print("\n💥 Setup verification failed!")
        exit(1)