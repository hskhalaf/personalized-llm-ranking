"""
Data generation module for personalized LLM ranking experiment.
Handles prompt generation, model response collection, and preference evaluation.
"""

import json
import os
import random
from typing import Dict, List, Tuple, Any
import ollama
from datasets import load_dataset
import time

from config import (
    DEFAULT_MODELS, PERSONAS, EXPERIMENT_CONFIG, MODEL_PATHS,
    DATA_PATHS, CACHE_FILES, get_cache_file_path, get_config_hash
)


class DataGenerator:
    def __init__(self, models: Dict = None, personas: Dict = None):
        """Initialize the data generator with models and personas."""
        self.models = models or DEFAULT_MODELS
        self.personas = personas or PERSONAS
        self.config_hash = get_config_hash()
        
        # Load Arena ELO scores
        self.arena_scores = self._load_arena_scores()
        
        # Update model ELO scores
        for model_key, model_info in self.models.items():
            arena_name = model_info['arena_name']
            if arena_name in self.arena_scores:
                model_info['elo'] = self.arena_scores[arena_name]
            else:
                print(f"Warning: {arena_name} not found in Arena dataset")
    
    def _load_arena_scores(self) -> Dict[str, float]:
        """Load ELO scores from Chatbot Arena dataset."""
        cache_file = get_cache_file_path(CACHE_FILES['arena_elo_scores'])
        
        # Check if cached
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Load from dataset
        print("Loading Arena ELO scores from dataset...")
        dataset = load_dataset(DATA_PATHS['arena_dataset'])
        
        # Extract scores
        scores = {}
        for item in dataset['train']:
            model_name = item['model']
            elo_score = item.get('Arena Score', item.get('score', item.get('elo', None)))
            if elo_score is not None:
                scores[model_name] = float(elo_score)
        
        # Cache the scores
        with open(cache_file, 'w') as f:
            json.dump(scores, f, indent=2)
        
        return scores
    
    def _load_prompt_template(self, template_name: str) -> str:
        """Load a prompt template from file."""
        template_path = os.path.join(DATA_PATHS['prompts_dir'], f"{template_name}.txt")
        with open(template_path, 'r') as f:
            return f.read()
    
    def _call_ollama(self, model: str, prompt: str, system: str = None) -> str:
        """Make a call to Ollama API."""
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = ollama.chat(model=model, messages=messages)
            return response['message']['content']
        except Exception as e:
            print(f"Error calling Ollama with model {model}: {e}")
            return None
    
    def _parse_json_response(self, response: str) -> Any:
        """Parse JSON response from LLM, handling common formatting issues."""
        try:
            # Try to extract JSON from the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx == -1:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response: {response}")
            return None
    
    def generate_prompts(self, persona: str) -> List[Dict]:
        """Generate prompts for a specific persona."""
        cache_file = get_cache_file_path(CACHE_FILES['generated_prompts'], persona)
        
        # Check if cached and config hasn't changed
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if cached_data.get('config_hash') == self.config_hash:
                    print(f"Using cached prompts for persona: {persona}")
                    return cached_data['prompts']
        
        print(f"Generating prompts for persona: {persona}")
        
        persona_info = self.personas[persona]
        template = self._load_prompt_template('prompt_generation')
        
        # Format template
        prompt = template.format(
            persona_name=persona_info['name'],
            persona_description=persona_info['description'],
            expertise=', '.join(persona_info['expertise']),
            communication_style=persona_info['communication_style'],
            num_prompts=EXPERIMENT_CONFIG['num_prompts']
        )
        
        # Generate prompts using Ollama
        response = self._call_ollama(MODEL_PATHS['generator_model'], prompt)
        if not response:
            raise Exception("Failed to generate prompts")
        
        prompts = self._parse_json_response(response)
        if not prompts or not isinstance(prompts, list):
            raise Exception("Failed to parse generated prompts")
        
        # Cache the prompts
        cache_data = {
            'config_hash': self.config_hash,
            'persona': persona,
            'prompts': prompts,
            'generated_at': time.time()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        return prompts
    
    def collect_model_responses(self, persona: str, prompts: List[Dict]) -> Dict:
        """Collect responses from all models for all prompts."""
        cache_file = get_cache_file_path(CACHE_FILES['model_responses'], persona)
        
        # Check if cached and config hasn't changed
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if cached_data.get('config_hash') == self.config_hash:
                    print(f"Using cached responses for persona: {persona}")
                    return cached_data['responses']
        
        print(f"Collecting responses from all models for persona: {persona}")
        
        responses = {}
        
        for prompt in prompts:
            prompt_id = prompt['id']
            prompt_text = prompt['text']
            responses[prompt_id] = {}
            
            for model_key, model_info in self.models.items():
                print(f"Getting response from {model_key} for prompt {prompt_id}")
                
                response = self._call_ollama(model_info['ollama_tag'], prompt_text)
                if response:
                    responses[prompt_id][model_key] = response
                else:
                    print(f"Failed to get response from {model_key}")
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.5)
        
        # Cache the responses
        cache_data = {
            'config_hash': self.config_hash,
            'persona': persona,
            'responses': responses,
            'generated_at': time.time()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        return responses
    
    def generate_preferences(self, persona: str, prompts: List[Dict], responses: Dict) -> List[Dict]:
        """Generate pairwise preferences using the evaluator model."""
        cache_file = get_cache_file_path(CACHE_FILES['preference_data'], persona)
        
        # Check if cached and config hasn't changed
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if cached_data.get('config_hash') == self.config_hash:
                    print(f"Using cached preferences for persona: {persona}")
                    return cached_data['preferences']
        
        print(f"Generating preferences for persona: {persona}")
        
        persona_info = self.personas[persona]
        template = self._load_prompt_template('preference_evaluation')
        
        # Split prompts into train/test
        random.seed(EXPERIMENT_CONFIG['random_seed'])
        random.shuffle(prompts)
        split_idx = int(len(prompts) * EXPERIMENT_CONFIG['train_test_split'])
        train_prompts = prompts[:split_idx]
        test_prompts = prompts[split_idx:]
        
        preferences = []
        
        # Generate preferences for training prompts
        for prompt in train_prompts:
            prompt_id = prompt['id']
            prompt_text = prompt['text']
            
            # Get all model responses for this prompt
            prompt_responses = responses[prompt_id]
            model_keys = list(prompt_responses.keys())
            
            # Generate all possible pairs
            for i in range(len(model_keys)):
                for j in range(i + 1, len(model_keys)):
                    model_a = model_keys[i]
                    model_b = model_keys[j]
                    
                    # Format evaluation prompt
                    eval_prompt = template.format(
                        persona_name=persona_info['name'],
                        persona_description=persona_info['description'],
                        expertise=', '.join(persona_info['expertise']),
                        communication_style=persona_info['communication_style'],
                        prompt=prompt_text,
                        response_a=prompt_responses[model_a],
                        response_b=prompt_responses[model_b]
                    )
                    
                    # Get evaluation
                    eval_response = self._call_ollama(MODEL_PATHS['evaluator_model'], eval_prompt)
                    if not eval_response:
                        continue
                    
                    evaluation = self._parse_json_response(eval_response)
                    if not evaluation or 'winner' not in evaluation:
                        continue
                    
                    # Determine winner and loser
                    if evaluation['winner'] == 'A':
                        winner = model_a
                        loser = model_b
                    else:
                        winner = model_b
                        loser = model_a
                    
                    preference = {
                        'prompt_id': prompt_id,
                        'prompt_text': prompt_text,
                        'winner': winner,
                        'loser': loser,
                        'reasoning': evaluation.get('reasoning', ''),
                        'confidence': evaluation.get('confidence', 'medium'),
                        'split': 'train'
                    }
                    
                    preferences.append(preference)
        
        # Add test prompts info
        test_info = {
            'test_prompts': [p['id'] for p in test_prompts],
            'train_prompts': [p['id'] for p in train_prompts]
        }
        
        # Cache the preferences
        cache_data = {
            'config_hash': self.config_hash,
            'persona': persona,
            'preferences': preferences,
            'test_info': test_info,
            'generated_at': time.time()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        return preferences
    
    def generate_ground_truth_rankings(self, persona: str, test_prompts: List[str], responses: Dict, prompts: List[Dict] = None) -> Dict:
        """Generate ground truth rankings for test prompts."""
        persona_info = self.personas[persona]
        template = self._load_prompt_template('ground_truth_ranking')
        
        # Create prompt_id to prompt_text mapping
        prompt_mapping = {}
        if prompts:
            prompt_mapping = {p['id']: p['text'] for p in prompts}
        
        rankings = {}
        
        for prompt_id in test_prompts:
            prompt_responses = responses[prompt_id]
            model_keys = list(prompt_responses.keys())
            
            # Get the actual prompt text
            prompt_text = prompt_mapping.get(prompt_id, prompt_id)
            
            # Format responses list
            responses_list = ""
            for i, model_key in enumerate(model_keys):
                responses_list += f"MODEL {chr(65+i)} ({model_key}):\n{prompt_responses[model_key]}\n\n"
            
            # Format ranking prompt
            ranking_prompt = template.format(
                persona_name=persona_info['name'],
                persona_description=persona_info['description'],
                expertise=', '.join(persona_info['expertise']),
                communication_style=persona_info['communication_style'],
                prompt=prompt_text,
                responses_list=responses_list,
                num_models=len(model_keys)
            )
            
            # Get ranking
            ranking_response = self._call_ollama(MODEL_PATHS['evaluator_model'], ranking_prompt)
            if not ranking_response:
                continue
            
            ranking_data = self._parse_json_response(ranking_response)
            if not ranking_data or 'ranking' not in ranking_data:
                continue
            
            rankings[prompt_id] = {
                'ranking': ranking_data['ranking'],
                'reasoning': ranking_data.get('reasoning', ''),
                'confidence': ranking_data.get('confidence', 'medium')
            }
        
        return rankings
    
    def generate_all_data(self, persona: str) -> Dict:
        """Generate all data for a persona: prompts, responses, and preferences."""
        print(f"Generating all data for persona: {persona}")
        
        # Generate prompts
        prompts = self.generate_prompts(persona)
        
        # Collect model responses
        responses = self.collect_model_responses(persona, prompts)
        
        # Generate preferences
        preferences = self.generate_preferences(persona, prompts, responses)
        
        # Get test prompt IDs
        cache_file = get_cache_file_path(CACHE_FILES['preference_data'], persona)
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            test_prompts = cached_data['test_info']['test_prompts']
        
        # Generate ground truth rankings
        ground_truth = self.generate_ground_truth_rankings(persona, test_prompts, responses, prompts)
        
        return {
            'prompts': prompts,
            'responses': responses,
            'preferences': preferences,
            'ground_truth': ground_truth,
            'test_prompts': test_prompts
        }


if __name__ == "__main__":
    # Test the data generator
    generator = DataGenerator()
    
    # Generate data for coder persona
    data = generator.generate_all_data('coder')
    print(f"Generated {len(data['prompts'])} prompts")
    print(f"Generated {len(data['preferences'])} preferences")
    print(f"Generated {len(data['ground_truth'])} ground truth rankings") 