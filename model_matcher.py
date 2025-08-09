#!/usr/bin/env python3
"""
Intelligent model matching between Ollama and Chatbot Arena datasets.
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
import ollama
from difflib import SequenceMatcher

class ModelMatcher:
    def __init__(self):
        self.arena_models = self._load_arena_models()
        self.ollama_models = self._load_ollama_models()
    
    def _load_arena_models(self) -> List[str]:
        """Load all model names from Chatbot Arena dataset."""
        print("Loading Arena models...")
        dataset = load_dataset("mathewhe/chatbot-arena-elo")
        models = set()
        for item in dataset['train']:
            models.add(item['Model'])
        return sorted(list(models))
    
    def _load_ollama_models(self) -> List[str]:
        """Load all available Ollama models."""
        print("Loading Ollama models...")
        try:
            client = ollama.Client()
            models = client.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            # Return common models as fallback
            return ['llama3.1:8b', 'gemma2:9b', 'qwen2:7b', 'phi3:3.8b', 'mistral:7b']
    
    def _normalize_name(self, name: str) -> str:
        """Normalize model name for comparison."""
        # Convert to lowercase and remove common separators
        normalized = name.lower()
        normalized = re.sub(r'[-_\.\s]+', '', normalized)
        
        # Normalize version patterns
        normalized = re.sub(r'v(\d+)', r'\1', normalized)  # v0.3 -> 03
        normalized = re.sub(r'(\d+)\.(\d+)', r'\1\2', normalized)  # 3.1 -> 31
        
        # Remove common suffixes/prefixes
        normalized = re.sub(r'^(meta|google|microsoft|mistral|qwen)', '', normalized)
        normalized = re.sub(r'(instruct|chat|base)$', '', normalized)
        
        return normalized
    
    def _extract_key_components(self, name: str) -> Dict[str, str]:
        """Extract key components from model name."""
        components = {
            'base_name': '',
            'version': '',
            'size': '',
            'variant': ''
        }
        
        name_lower = name.lower()
        
        # Extract base model name
        if 'llama' in name_lower:
            components['base_name'] = 'llama'
            # Extract version (3, 3.1, etc.)
            version_match = re.search(r'llama[^0-9]*([0-9\.]+)', name_lower)
            if version_match:
                components['version'] = version_match.group(1)
        elif 'gemma' in name_lower:
            components['base_name'] = 'gemma'
            version_match = re.search(r'gemma[^0-9]*([0-9\.]+)', name_lower)
            if version_match:
                components['version'] = version_match.group(1)
        elif 'qwen' in name_lower:
            components['base_name'] = 'qwen'
            version_match = re.search(r'qwen[^0-9]*([0-9\.]+)', name_lower)
            if version_match:
                components['version'] = version_match.group(1)
        elif 'phi' in name_lower:
            components['base_name'] = 'phi'
            version_match = re.search(r'phi[^0-9]*([0-9\.]+)', name_lower)
            if version_match:
                components['version'] = version_match.group(1)
        elif 'mistral' in name_lower:
            components['base_name'] = 'mistral'
            version_match = re.search(r'v?([0-9\.]+)', name_lower)
            if version_match:
                components['version'] = version_match.group(1)
        
        # Extract size (7b, 8b, 9b, etc.)
        size_match = re.search(r'([0-9\.]+)b', name_lower)
        if size_match:
            components['size'] = size_match.group(1) + 'b'
        
        return components
    
    def _calculate_match_score(self, ollama_name: str, arena_name: str) -> float:
        """Calculate match score between two model names."""
        # Component-based matching
        ollama_comp = self._extract_key_components(ollama_name)
        arena_comp = self._extract_key_components(arena_name)
        
        score = 0.0
        
        # Base name match (most important)
        if ollama_comp['base_name'] == arena_comp['base_name'] and ollama_comp['base_name']:
            score += 0.4
        
        # Version match
        if ollama_comp['version'] == arena_comp['version'] and ollama_comp['version']:
            score += 0.3
        elif ollama_comp['version'] and arena_comp['version']:
            # Partial version match (e.g., "3" matches "3.0")
            if ollama_comp['version'] in arena_comp['version'] or arena_comp['version'] in ollama_comp['version']:
                score += 0.15
        
        # Size match
        if ollama_comp['size'] == arena_comp['size'] and ollama_comp['size']:
            score += 0.2
        
        # String similarity as tiebreaker
        similarity = SequenceMatcher(None, self._normalize_name(ollama_name), 
                                   self._normalize_name(arena_name)).ratio()
        score += similarity * 0.1
        
        return score
    
    def find_best_matches(self, min_score: float = 0.5) -> Dict[str, Dict]:
        """Find best matches between Ollama and Arena models."""
        matches = {}
        
        print(f"\nMatching {len(self.ollama_models)} Ollama models with {len(self.arena_models)} Arena models...")
        
        for ollama_model in self.ollama_models:
            best_match = None
            best_score = 0.0
            
            for arena_model in self.arena_models:
                score = self._calculate_match_score(ollama_model, arena_model)
                if score > best_score:
                    best_score = score
                    best_match = arena_model
            
            if best_score >= min_score:
                matches[ollama_model] = {
                    'arena_name': best_match,
                    'confidence': best_score,
                    'ollama_tag': ollama_model
                }
                print(f"✅ {ollama_model} -> {best_match} (confidence: {best_score:.2f})")
            else:
                print(f"❌ {ollama_model} -> No good match (best: {best_match}, score: {best_score:.2f})")
        
        return matches
    
    def generate_config(self, matches: Dict[str, Dict]) -> str:
        """Generate Python config code for the matches."""
        config_lines = ["DEFAULT_MODELS = {"]
        
        for ollama_tag, match_info in matches.items():
            key = ollama_tag.replace(':', '_').replace('-', '_')
            config_lines.append(f"    '{ollama_tag}': {{")
            config_lines.append(f"        'arena_name': '{match_info['arena_name']}',")
            config_lines.append(f"        'elo': None,  # Will be loaded from dataset")
            config_lines.append(f"        'ollama_tag': '{match_info['ollama_tag']}',")
            config_lines.append(f"        'confidence': {match_info['confidence']:.3f}")
            config_lines.append(f"    }},")
        
        config_lines.append("}")
        return "\n".join(config_lines)

def main():
    matcher = ModelMatcher()
    
    print("\n" + "="*60)
    print("AVAILABLE MODELS")
    print("="*60)
    
    print(f"\nOllama Models ({len(matcher.ollama_models)}):")
    for model in matcher.ollama_models[:10]:  # Show first 10
        print(f"  - {model}")
    if len(matcher.ollama_models) > 10:
        print(f"  ... and {len(matcher.ollama_models) - 10} more")
    
    print(f"\nArena Models (showing relevant ones):")
    relevant_arena = [m for m in matcher.arena_models 
                     if any(keyword in m.lower() for keyword in ['llama', 'gemma', 'qwen', 'phi', 'mistral'])]
    for model in relevant_arena[:15]:
        print(f"  - {model}")
    
    print("\n" + "="*60)
    print("MATCHING RESULTS")
    print("="*60)
    
    matches = matcher.find_best_matches(min_score=0.3)
    
    print(f"\n" + "="*60)
    print("GENERATED CONFIG")
    print("="*60)
    
    config_code = matcher.generate_config(matches)
    print(config_code)
    
    # Save to file
    with open('matched_models.json', 'w') as f:
        json.dump(matches, f, indent=2)
    
    print(f"\n✅ Saved {len(matches)} matches to 'matched_models.json'")
    
    return matches

if __name__ == "__main__":
    main()