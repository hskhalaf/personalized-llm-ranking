"""
Minimal Working LLM Evaluation System
Requirements: pip install flask flask-cors requests numpy
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import requests
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import ast
import threading
from concurrent.futures import ThreadPoolExecutor
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

app = Flask(__name__)
CORS(app)

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"

# Global state (in production, use a database)
sessions = {}

# Fetch global ELOs from Hugging Face at server start
GLOBAL_ELOS = {}
def fetch_global_elos():
    global GLOBAL_ELOS
    if load_dataset is None:
        print("datasets library not installed, using hardcoded ELOs.")
        return
    try:
        dataset = load_dataset("mathewhe/chatbot-arena-elo", split="train")
        for row in dataset:
            model_name = row["Model"].lower()  # store as lowercase
            elo = row["Arena Score"]
            GLOBAL_ELOS[model_name] = elo
        print("Fetched global ELOs from Hugging Face.")
        print("GLOBAL_ELOS keys:", list(GLOBAL_ELOS.keys())[:10])
    except Exception as e:
        print(f"Error fetching ELOs: {e}")

fetch_global_elos()

print("GLOBAL_ELOS keys:", list(GLOBAL_ELOS.keys()))

ARENA_TO_OLLAMA = {
    "Gemma-2B-it": "gemma:2b",
    "Qwen1.5-4B-Chat": "qwen:4b",
    "Gemma-3-4B-it": "gemma3:4b",
    "SmolLM2-1.7B-Instruct": "smollm2:1.7b",
    "Llama-2-7B-chat": "llama2:7b",
}

# Use exact dataset names for ELO lookup
ARENA_TO_DATASET = {
    "Gemma-2B-it": "Gemma-2B-it",
    "Qwen1.5-4B-Chat": "Qwen1.5-4B-Chat",
    "Gemma-3-4B-it": "Gemma-3-4B-it",
    "SmolLM2-1.7B-Instruct": "SmolLM2-1.7B-Instruct",
    "Llama-2-7B-chat": "Llama-2-7B-chat",
}

INSTALLED_MODELS = ARENA_TO_OLLAMA.copy()

MODEL_DISPLAY_NAMES = list(ARENA_TO_OLLAMA.keys())

def get_global_elo(arena_name):
    # Fuzzy search for best match in GLOBAL_ELOS (all lowercased)
    search_key = arena_name.lower().replace('-', '').replace('b', '')
    for dataset_key in GLOBAL_ELOS:
        if search_key in dataset_key.replace('-', '').replace('b', ''):
            return GLOBAL_ELOS[dataset_key]
    return 1200

# Print all model names from the Arena ELO dataset at server start
print("All model names in the Arena ELO dataset:")
for name in GLOBAL_ELOS:
    print(name)

# Print global ELOs for selected models at server start
print("\nGlobal ELOs for selected models:")
for arena_name, ollama_tag in ARENA_TO_OLLAMA.items():
    dataset_name = ARENA_TO_DATASET.get(arena_name, arena_name)
    elo = GLOBAL_ELOS.get(dataset_name.lower(), 1200)
    print(f"{arena_name} (Ollama: {ollama_tag}) | Dataset: {dataset_name} | Arena Score: {elo}")

@dataclass
class Model:
    key: str
    name: str
    ollama_name: str
    global_elo: float
    personal_elo: float = 1500.0
    wins: int = 0
    losses: int = 0

@dataclass
class Comparison:
    prompt: str
    model_a: str
    model_b: str
    response_a: str
    response_b: str
    winner: Optional[str] = None

class EvaluationSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.behavior = ""
        self.prompts = []
        self.comparisons = []
        self.current_comparison = 0
        self.models = {}
        for arena_name, ollama_name in INSTALLED_MODELS.items():
            self.models[arena_name] = Model(
                key=arena_name,
                name=arena_name,
                ollama_name=ollama_name,
                global_elo=get_global_elo(arena_name)
            )
        self.model_generations = {}  # arena_name -> list of generations
    
    def generate_all_generations(self):
        """Generate and store all generations for each model and prompt using parallel requests per model."""
        self.model_generations = {}
        for arena_name, model in self.models.items():
            def get_response(prompt):
                try:
                    response = requests.post(
                        f"{OLLAMA_BASE_URL}/api/generate",
                        json={
                            "model": model.ollama_name,
                            "prompt": prompt,
                            "stream": False,
                            "temperature": 0.7,
                            "options": {"num_predict": 150}
                        },
                        timeout=30
                    )
                    if response.status_code == 200:
                        return response.json()['response'].strip()
                    else:
                        return f"[Error getting response from {arena_name}]"
                except Exception as e:
                    print(f"Error getting model response: {e}")
                    return f"Mock response from {arena_name} for: {prompt}"
            with ThreadPoolExecutor(max_workers=5) as executor:
                self.model_generations[arena_name] = list(executor.map(get_response, self.prompts))
    
    def generate_all_comparisons(self, pairs_per_prompt=3):
        """For each prompt, randomly select a subset of model pairs to compare."""
        model_keys = list(self.models.keys())
        self.comparisons = []
        for prompt_idx, prompt in enumerate(self.prompts):
            pairs = set()
            while len(pairs) < min(pairs_per_prompt, len(model_keys)*(len(model_keys)-1)//2):
                i, j = sorted(random.sample(range(len(model_keys)), 2))
                pairs.add((i, j))
            for i, j in pairs:
                a, b = model_keys[i], model_keys[j]
                response_a = self.model_generations[a][prompt_idx]
                response_b = self.model_generations[b][prompt_idx]
                # Randomly swap to avoid bias
                if random.random() > 0.5:
                    a, b = b, a
                    response_a, response_b = response_b, response_a
                self.comparisons.append(Comparison(
                    prompt=prompt,
                    model_a=a,
                    model_b=b,
                    response_a=response_a,
                    response_b=response_b
                ))
        random.shuffle(self.comparisons)
    
    def generate_all_responses(self):
        """Pre-generate and store responses for all comparisons"""
        for comparison in self.comparisons:
            try:
                # Get response from model A
                response_a = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": self.models[comparison.model_a].ollama_name,
                        "prompt": comparison.prompt,
                        "stream": False,
                        "temperature": 0.7,
                        "options": {"num_predict": 150}
                    },
                    timeout=30
                )
                if response_a.status_code == 200:
                    comparison.response_a = response_a.json()['response'].strip()
                else:
                    comparison.response_a = f"[Error getting response from {comparison.model_a}]"

                # Get response from model B
                response_b = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": self.models[comparison.model_b].ollama_name,
                        "prompt": comparison.prompt,
                        "stream": False,
                        "temperature": 0.7,
                        "options": {"num_predict": 150}
                    },
                    timeout=30
                )
                if response_b.status_code == 200:
                    comparison.response_b = response_b.json()['response'].strip()
                else:
                    comparison.response_b = f"[Error getting response from {comparison.model_b}]"
            except Exception as e:
                print(f"Error getting model responses: {e}")
                comparison.response_a = f"Mock response from {comparison.model_a} for: {comparison.prompt}"
                comparison.response_b = f"Mock response from {comparison.model_b} for: {comparison.prompt}"
            # Randomly swap positions to avoid bias
            if random.random() > 0.5:
                comparison.model_a, comparison.model_b = comparison.model_b, comparison.model_a
                comparison.response_a, comparison.response_b = comparison.response_b, comparison.response_a
    
    def update_elo(self, winner: str, loser: str, k: float = 32):
        """Update ELO ratings based on comparison result"""
        winner_model = self.models[winner]
        loser_model = self.models[loser]
        
        # Update win/loss counts
        winner_model.wins += 1
        loser_model.losses += 1
        
        # Calculate expected scores
        expected_winner = 1 / (1 + 10 ** ((loser_model.personal_elo - winner_model.personal_elo) / 400))
        expected_loser = 1 - expected_winner
        
        # Update ELOs
        winner_model.personal_elo += k * (1 - expected_winner)
        loser_model.personal_elo += k * (0 - expected_loser)

# Routes
@app.route('/')
def index():
    """Serve the HTML interface and print global ELOs for selected models on the starting page."""
    print("\n--- Global ELOs for selected models (on start page) ---")
    for arena_name, ollama_tag in ARENA_TO_OLLAMA.items():
        dataset_name = ARENA_TO_DATASET.get(arena_name, arena_name)
        elo = GLOBAL_ELOS.get(dataset_name.lower(), 1200)
        print(f"{arena_name} (Ollama: {ollama_tag}) | Dataset: {dataset_name} | Arena Score: {elo}")
    # Prepare data for template rendering
    models = list(ARENA_TO_OLLAMA.items())
    elos = {k: GLOBAL_ELOS.get(ARENA_TO_DATASET.get(k, k).lower(), 1200) for k in ARENA_TO_OLLAMA}
    from jinja2 import Template
    template = Template(HTML_TEMPLATE)
    return template.render(models=models, elos=elos)

@app.route('/api/generate-prompts', methods=['POST'])
def generate_prompts():
    """Generate test prompts based on behavior description"""
    data = request.json
    behavior = data.get('behavior', '')
    session_id = data.get('session_id', str(random.randint(1000, 9999)))
    # Create or get session
    if session_id not in sessions:
        sessions[session_id] = EvaluationSession(session_id)
    session = sessions[session_id]
    session.behavior = behavior
    # Enhanced meta-prompt and use llama3.2:3b
    meta_prompt = f'''
You are a prompt engineer. Your task is to create 5 diverse, specific, and challenging test prompts that would best evaluate if a language model exhibits the following behavior:

BEHAVIOR: {behavior}

Each prompt should be directly related to this behavior, and should be clear, specific, and likely to reveal whether the model has this characteristic. Do NOT include any explanation or extra text, only return a valid JSON array of 5 double-quoted strings.

Example output:
["prompt 1", "prompt 2", "prompt 3", "prompt 4", "prompt 5"]
'''
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": "llama3.2:3b",  # Use 3B model for better prompt generation
                "prompt": meta_prompt,
                "stream": False,
                "temperature": 0.7
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            response_text = result['response'].strip()
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                prompts = json.loads(json_match.group())
                print("Parsed prompts:", prompts)
                print("Number of prompts:", len(prompts))
                session.prompts = prompts[:5]  # Limit to 5 prompts
            else:
                print("Prompt generation response (no JSON match):", repr(response_text))
                raise ValueError("Could not parse prompts")
        else:
            raise Exception("Ollama request failed")
    except Exception as e:
        print(f"Error generating prompts: {e}")
        # Fallback prompts
        session.prompts = [
            "How can I improve my productivity at work?",
            "What's the best way to learn Python programming?",
            "Explain quantum computing in simple terms",
            "What should I consider when buying a laptop?",
            "How do I start a small garden?"
        ]
    # Only return prompts, do not generate model responses yet
    return jsonify({
        'session_id': session_id,
        'prompts': session.prompts
    })

@app.route('/api/generate-responses', methods=['POST'])
def generate_responses():
    data = request.json
    session_id = data.get('session_id')
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    session = sessions[session_id]
    # Generate all generations and comparisons
    session.generate_all_generations()
    session.generate_all_comparisons()
    return jsonify({'success': True})

@app.route('/api/get-comparison', methods=['POST'])
def get_comparison():
    """Get the next comparison with pre-generated model responses"""
    data = request.json
    session_id = data.get('session_id')
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    session = sessions[session_id]
    if session.current_comparison >= len(session.comparisons):
        return jsonify({'completed': True})
    comparison = session.comparisons[session.current_comparison]
    return jsonify({
        'comparison_index': session.current_comparison,
        'total_comparisons': len(session.comparisons),
        'prompt': comparison.prompt,
        'response_a': comparison.response_a,
        'response_b': comparison.response_b,
        'progress': (session.current_comparison / len(session.comparisons)) * 100
    })

@app.route('/api/submit-vote', methods=['POST'])
def submit_vote():
    """Submit vote for current comparison"""
    data = request.json
    session_id = data.get('session_id')
    winner = data.get('winner')  # 'A', 'B', or 'tie'
    
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = sessions[session_id]
    comparison = session.comparisons[session.current_comparison]
    
    # Record winner and update ELO
    if winner == 'A':
        comparison.winner = comparison.model_a
        session.update_elo(comparison.model_a, comparison.model_b)
    elif winner == 'B':
        comparison.winner = comparison.model_b
        session.update_elo(comparison.model_b, comparison.model_a)
    else:
        comparison.winner = 'tie'
    
    # Move to next comparison
    session.current_comparison += 1
    
    return jsonify({'success': True})

@app.route('/api/get-results', methods=['POST'])
def get_results():
    """Get final results with both scoring methods"""
    data = request.json
    session_id = data.get('session_id')
    alpha = data.get('alpha', 0.7)
    
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = sessions[session_id]
    results = []
    
    for key, model in session.models.items():
        # Method A: Weighted average
        method_a = alpha * model.global_elo + (1 - alpha) * model.personal_elo
        
        # Method B: Bayesian update
        n = model.wins + model.losses
        if n > 0:
            prior_variance = 25 ** 2
            data_variance = 100 ** 2 / n
            method_b = (
                (model.global_elo / prior_variance + model.personal_elo / data_variance) /
                (1 / prior_variance + 1 / data_variance)
            )
        else:
            method_b = model.global_elo
        
        results.append({
            'key': key,
            'name': key,
            'global_elo': model.global_elo,
            'personal_elo': model.personal_elo,
            'method_a': method_a,
            'method_b': method_b,
            'wins': model.wins,
            'losses': model.losses
        })
    
    return jsonify({
        'results': results,
        'behavior': session.behavior,
        'total_comparisons': len(session.comparisons)
    })

@app.route('/api/responses-ready', methods=['POST'])
def responses_ready():
    data = request.json
    session_id = data.get('session_id')
    if session_id not in sessions:
        return jsonify({'ready': False})
    session = sessions[session_id]
    # If any comparison has empty responses, not ready
    ready = all(c.response_a and c.response_b for c in session.comparisons)
    return jsonify({'ready': ready})

# HTML Template (embedded for simplicity)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Personalized LLM Evaluation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header { background: #2c3e50; color: white; padding: 30px 0; margin-bottom: 30px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1 { text-align: center; font-size: 2.5em; font-weight: 300; }
        .subtitle { text-align: center; opacity: 0.8; margin-top: 10px; }
        .phase-container { background: white; border-radius: 8px; padding: 30px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        .phase-title { font-size: 1.5em; margin-bottom: 20px; color: #2c3e50; display: flex; align-items: center; gap: 10px; }
        .phase-number { background: #3498db; color: white; width: 35px; height: 35px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; }
        .behavior-input { width: 100%; padding: 15px; border: 2px solid #ddd; border-radius: 5px; font-size: 16px; margin-bottom: 20px; transition: border-color 0.3s; }
        .behavior-input:focus { outline: none; border-color: #3498db; }
        .model-elo-table { margin: 20px 0 30px 0; border-collapse: collapse; width: 100%; max-width: 600px; }
        .model-elo-table th, .model-elo-table td { padding: 8px 12px; border-bottom: 1px solid #eee; text-align: left; }
        .model-elo-table th { background: #3498db; color: #fff; font-weight: 600; }
        .model-elo-table tr:last-child td { border-bottom: none; }
        .btn { background: #3498db; color: white; border: none; padding: 12px 30px; border-radius: 5px; font-size: 16px; cursor: pointer; transition: background 0.3s; }
        .btn:hover { background: #2980b9; }
        .btn:disabled { background: #95a5a6; cursor: not-allowed; }
        .btn-secondary { background: #95a5a6; }
        .btn-secondary:hover { background: #7f8c8d; }
        .prompts-list { background: #f8f9fa; border-radius: 5px; padding: 20px; margin: 20px 0; }
        .prompt-item { padding: 10px; margin: 5px 0; background: white; border-radius: 3px; border-left: 4px solid #3498db; }
        .comparison-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
        .response-box { background: #f8f9fa; padding: 20px; border-radius: 5px; border: 2px solid transparent; transition: border-color 0.3s; cursor: pointer; position: relative; min-height: 150px; }
        .response-box:hover { border-color: #3498db; }
        .response-box.selected { border-color: #2ecc71; background: #e8f8f5; }
        .response-label { font-weight: bold; color: #2c3e50; margin-bottom: 10px; font-size: 1.2em; }
        .response-text { color: #555; line-height: 1.6; white-space: pre-wrap; }
        .vote-buttons { display: flex; gap: 10px; justify-content: center; margin-top: 20px; }
        .progress-bar { background: #ecf0f1; height: 30px; border-radius: 15px; overflow: hidden; margin: 20px 0; }
        .progress-fill { background: #3498db; height: 100%; width: 0%; transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }
        .leaderboard { overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #34495e; color: white; font-weight: 500; }
        tr:hover { background: #f5f5f5; }
        .rank-change { font-size: 0.9em; margin-left: 5px; }
        .rank-up { color: #27ae60; }
        .rank-down { color: #e74c3c; }
        .loading { text-align: center; padding: 40px; color: #7f8c8d; }
        .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 20px auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .alpha-slider { margin: 20px 0; }
        .slider { width: 100%; height: 5px; border-radius: 5px; background: #ddd; outline: none; -webkit-appearance: none; }
        .slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 20px; height: 20px; border-radius: 50%; background: #3498db; cursor: pointer; }
        .slider::-moz-range-thumb { width: 20px; height: 20px; border-radius: 50%; background: #3498db; cursor: pointer; }
        .alpha-value { text-align: center; margin-top: 10px; font-weight: bold; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <header>
        <h1>Personalized LLM Evaluation System</h1>
        <p class="subtitle">Combine global Arena rankings with your personal preferences</p>
    </header>
    
    <div class="container">
        <!-- Phase 1: Behavior Description -->
        <div id="phase1" class="phase-container">
            <h2 class="phase-title">
                <span class="phase-number">1</span>
                Describe Your Desired Model Behavior
            </h2>
            <p style="margin-bottom: 20px;">Describe in natural language what kind of behavior you want from the LLM:</p>
            <input type="text" id="behaviorInput" class="behavior-input" 
                   placeholder="e.g., I want concise, practical responses without unnecessary elaboration"
                   value="I want concise, practical responses without unnecessary elaboration" />
            <!-- Global ELO Table -->
            <div style="margin-top: 20px;">
                <h3 style="margin-bottom: 10px;">Global Arena ELOs</h3>
                <table class="model-elo-table">
                    <thead>
                        <tr><th>Model</th><th>Global Arena ELO</th></tr>
                    </thead>
                    <tbody>
                        {% for model, ollama in models %}
                        <tr>
                            <td>{{ model }}</td>
                            <td>{{ elos[model] }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            <button class="btn" onclick="generatePrompts()">Generate Test Prompts</button>
        </div>
        
        <!-- Phase 2: Generated Prompts -->
        <div id="phase2" class="phase-container hidden">
            <h2 class="phase-title">
                <span class="phase-number">2</span>
                Generated Test Prompts
            </h2>
            <div id="promptsList" class="prompts-list"></div>
            <button class="btn" id="approvePromptsBtn" onclick="approvePrompts()">Approve Prompts</button>
            <div id="responseGenSpinner" class="loading hidden">
                <div class="spinner"></div>
                <p>Generating all model responses. This may take a minute...</p>
            </div>
            <button class="btn" id="startEvalBtn" onclick="startEvaluation()" disabled>Start Evaluation</button>
            <button class="btn btn-secondary" onclick="generatePrompts()">Regenerate Prompts</button>
        </div>
        
        <!-- Phase 3: Pairwise Comparisons -->
        <div id="phase3" class="phase-container hidden">
            <h2 class="phase-title">
                <span class="phase-number">3</span>
                Compare Model Responses
            </h2>
            <div class="progress-bar">
                <div id="progressFill" class="progress-fill">0%</div>
            </div>
            <h3 id="currentPrompt" style="margin-bottom: 20px;"></h3>
            <div id="loadingSpinner" class="loading">
                <div class="spinner"></div>
                <p>Getting model responses...</p>
            </div>
            <div id="comparisonContent" class="hidden">
                <div class="comparison-container">
                    <div class="response-box" onclick="selectResponse('A')">
                        <div class="response-label">Response A</div>
                        <div id="responseA" class="response-text"></div>
                    </div>
                    <div class="response-box" onclick="selectResponse('B')">
                        <div class="response-label">Response B</div>
                        <div id="responseB" class="response-text"></div>
                    </div>
                </div>
                <div class="vote-buttons">
                    <button class="btn" onclick="submitVote('A')">Choose A</button>
                    <button class="btn btn-secondary" onclick="submitVote('tie')">Tie</button>
                    <button class="btn" onclick="submitVote('B')">Choose B</button>
                </div>
            </div>
        </div>
        
        <!-- Phase 4: Results -->
        <div id="phase4" class="phase-container hidden">
            <h2 class="phase-title">
                <span class="phase-number">4</span>
                Personalized Leaderboard
            </h2>
            <div class="alpha-slider">
                <label>Adjust weight (α) between global and personal scores:</label>
                <input type="range" id="alphaSlider" class="slider" min="0" max="100" value="70" oninput="updateAlpha()">
                <div class="alpha-value">α = <span id="alphaValue">0.70</span> (70% global, 30% personal)</div>
            </div>
            <div class="leaderboard">
                <table id="leaderboardTable">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Global Rank</th>
                            <th>Method A (Weighted)</th>
                            <th>Method B (Bayesian)</th>
                            <th>Your Votes</th>
                            <th>Personal ELO</th>
                        </tr>
                    </thead>
                    <tbody id="leaderboardBody"></tbody>
                </table>
            </div>
            <div style="margin-top: 30px; text-align: center;">
                <button class="btn" onclick="location.reload()">Start New Evaluation</button>
            </div>
        </div>
    </div>
    
    <script>
        let sessionId = null;
        let currentAlpha = 0.7;
        
        async function generatePrompts() {
            const behavior = document.getElementById('behaviorInput').value;
            if (!behavior) {
                alert('Please describe your desired model behavior');
                return;
            }
            try {
                const response = await fetch('/api/generate-prompts', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({behavior: behavior, session_id: sessionId})
                });
                const data = await response.json();
                sessionId = data.session_id;
                // Display prompts
                const promptsList = document.getElementById('promptsList');
                promptsList.innerHTML = data.prompts.map((p, i) => 
                    `<div class="prompt-item">${i + 1}. ${p}</div>`
                ).join('');
                // Show phase 2
                document.getElementById('phase1').classList.add('hidden');
                document.getElementById('phase2').classList.remove('hidden');
                // Hide spinner and disable start button until approval
                document.getElementById('responseGenSpinner').classList.add('hidden');
                document.getElementById('startEvalBtn').disabled = true;
                document.getElementById('approvePromptsBtn').disabled = false;
            } catch (error) {
                alert('Error generating prompts: ' + error);
            }
        }
        async function approvePrompts() {
            document.getElementById('approvePromptsBtn').disabled = true;
            document.getElementById('responseGenSpinner').classList.remove('hidden');
            document.getElementById('startEvalBtn').disabled = true;
            // Call backend to generate all responses
            await fetch('/api/generate-responses', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: sessionId})
            });
            // Poll backend for readiness
            await pollForResponsesReady();
            document.getElementById('responseGenSpinner').classList.add('hidden');
            document.getElementById('startEvalBtn').disabled = false;
        }
        async function pollForResponsesReady() {
            // Poll a new endpoint to check if all responses are ready
            while (true) {
                const resp = await fetch('/api/responses-ready', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: sessionId})
                });
                const data = await resp.json();
                if (data.ready) {
                    document.getElementById('responseGenSpinner').classList.add('hidden');
                    document.getElementById('startEvalBtn').disabled = false;
                    // Auto-advance to evaluation phase
                    startEvaluation();
                    return;
                }
                await new Promise(r => setTimeout(r, 1500));
            }
        }
        
        function startEvaluation() {
            document.getElementById('phase2').classList.add('hidden');
            document.getElementById('phase3').classList.remove('hidden');
            showNextComparison();
        }
        
        async function showNextComparison() {
            // Show loading
            document.getElementById('loadingSpinner').classList.remove('hidden');
            document.getElementById('comparisonContent').classList.add('hidden');
            
            try {
                const response = await fetch('/api/get-comparison', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: sessionId})
                });
                
                const data = await response.json();
                
                if (data.completed) {
                    showResults();
                    return;
                }
                
                // Update progress
                document.getElementById('progressFill').style.width = data.progress + '%';
                document.getElementById('progressFill').textContent = Math.round(data.progress) + '%';
                
                // Show prompt and responses
                document.getElementById('currentPrompt').textContent = `Prompt: "${data.prompt}"`;
                document.getElementById('responseA').textContent = data.response_a;
                document.getElementById('responseB').textContent = data.response_b;
                
                // Clear selection
                document.querySelectorAll('.response-box').forEach(box => 
                    box.classList.remove('selected')
                );
                
                // Show comparison
                document.getElementById('loadingSpinner').classList.add('hidden');
                document.getElementById('comparisonContent').classList.remove('hidden');
                
            } catch (error) {
                alert('Error getting comparison: ' + error);
            }
        }
        
        function selectResponse(choice) {
            document.querySelectorAll('.response-box').forEach(box => 
                box.classList.remove('selected')
            );
            
            if (choice === 'A') {
                document.querySelectorAll('.response-box')[0].classList.add('selected');
            } else if (choice === 'B') {
                document.querySelectorAll('.response-box')[1].classList.add('selected');
            }
        }
        
        async function submitVote(winner) {
            try {
                await fetch('/api/submit-vote', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: sessionId, winner: winner})
                });
                
                // Show next comparison
                showNextComparison();
                
            } catch (error) {
                alert('Error submitting vote: ' + error);
            }
        }
        
        async function showResults() {
            document.getElementById('phase3').classList.add('hidden');
            document.getElementById('phase4').classList.remove('hidden');
            
            await updateLeaderboard();
        }
        
        async function updateLeaderboard() {
            try {
                const response = await fetch('/api/get-results', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: sessionId, alpha: currentAlpha})
                });
                
                const data = await response.json();
                const results = data.results;
                
                // Sort by Method A score
                results.sort((a, b) => b.method_a - a.method_a);
                
                // Calculate ranks
                const globalRanks = [...results].sort((a, b) => b.global_elo - a.global_elo);
                const methodBRanks = [...results].sort((a, b) => b.method_b - a.method_b);
                
                // Display results
                const tbody = document.getElementById('leaderboardBody');
                tbody.innerHTML = results.map((model, i) => {
                    const globalRank = globalRanks.findIndex(m => m.key === model.key) + 1;
                    const methodBRank = methodBRanks.findIndex(m => m.key === model.key) + 1;
                    const methodARank = i + 1;
                    
                    return `
                        <tr>
                            <td>${model.name}</td>
                            <td>#${globalRank} (${model.global_elo.toFixed(0)})</td>
                            <td>#${methodARank} (${model.method_a.toFixed(0)})
                                ${getRankChange(globalRank, methodARank)}</td>
                            <td>#${methodBRank} (${model.method_b.toFixed(0)})
                                ${getRankChange(globalRank, methodBRank)}</td>
                            <td>${model.wins}-${model.losses}</td>
                            <td>${model.personal_elo.toFixed(0)}</td>
                        </tr>
                    `;
                }).join('');
                
            } catch (error) {
                alert('Error getting results: ' + error);
            }
        }
        
        function getRankChange(oldRank, newRank) {
            const diff = oldRank - newRank;
            if (diff > 0) {
                return `<span class="rank-change rank-up">↑${diff}</span>`;
            } else if (diff < 0) {
                return `<span class="rank-change rank-down">↓${Math.abs(diff)}</span>`;
            }
            return '';
        }
        
        async function updateAlpha() {
            const slider = document.getElementById('alphaSlider');
            currentAlpha = slider.value / 100;
            document.getElementById('alphaValue').textContent = currentAlpha.toFixed(2);
            // Update the percentage display
            const percentGlobal = Math.round(currentAlpha * 100);
            const percentPersonal = 100 - percentGlobal;
            document.getElementById('alphaValue').parentNode.innerHTML = `α = <span id="alphaValue">${currentAlpha.toFixed(2)}</span> (${percentGlobal}% global, ${percentPersonal}% personal)`;
            await updateLeaderboard();
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("Starting LLM Evaluation Server...")
    print("Make sure Ollama is running at http://localhost:11434")
    print("Install at least a few of these models:")
    print("  - ollama pull gemma:2b")
    print("  - ollama pull qwen:4b")
    print("  - ollama pull gemma3:4b")
    print("  - ollama pull smollm2:1.7b")
    print("  - ollama pull llama2:7b")
    print("\nServer starting at http://localhost:5050")
    app.run(debug=True, port=5050)
