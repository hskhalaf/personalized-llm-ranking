#!/usr/bin/env python3
"""
Data Preparation Script for Personalized LLM Preference Model

This script downloads, processes, and caches the training data:
1. Loads LMSYS Arena and Anthropic HH-RLHF datasets
2. Filters and cleans the Arena data
3. Builds model vocabulary and filters low-frequency models
4. Encodes prompts using sentence-transformers
5. Creates processed DataFrames and saves to cache
6. Identifies users for personalization
"""

import yaml
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Set
import logging

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreparer:
    """Main class for data preparation pipeline."""
    
    def __init__(self, config_path: str = "configs/experiment_config.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.cache_dir = Path(self.config['cache_dir'])
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.prompt_encoder = None
        self.model_vocab = {}
        self.processed_data = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load experiment configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Arena and HH-RLHF datasets."""
        logger.info("Loading datasets...")
        
        # Load LMSYS Arena dataset
        logger.info(f"Loading Arena dataset: {self.config['data_source_arena']}")
        arena_dataset = load_dataset(self.config['data_source_arena'], split='train')
        arena_df = arena_dataset.to_pandas()
        logger.info(f"Arena dataset loaded: {len(arena_df)} rows")
        
        # Load HH-RLHF dataset
        logger.info(f"Loading HH-RLHF dataset: {self.config['data_source_hh']}")
        hh_dataset = load_dataset(self.config['data_source_hh'], split='train')
        hh_df = hh_dataset.to_pandas()
        logger.info(f"HH-RLHF dataset loaded: {len(hh_df)} rows")
        
        return arena_df, hh_df
    
    def filter_arena_data(self, arena_df: pd.DataFrame) -> pd.DataFrame:
        """Filter Arena data to remove invalid comparisons."""
        logger.info("Filtering Arena data...")
        initial_count = len(arena_df)
        
        # Remove rows that are not pairwise comparisons
        # Keep only rows where we have winner/loser information
        filtered_df = arena_df.dropna(subset=['winner', 'model_a', 'model_b']).copy()
        logger.info(f"After removing non-pairwise: {len(filtered_df)} rows")
        
        # Remove toxic content (if toxic flags exist)
        toxic_columns = [col for col in filtered_df.columns if 'toxic' in col.lower()]
        if toxic_columns:
            for col in toxic_columns:
                filtered_df = filtered_df[filtered_df[col] != True]
            logger.info(f"After removing toxic content: {len(filtered_df)} rows")
        
        # Remove moderation errors (if moderation flags exist)
        mod_columns = [col for col in filtered_df.columns if 'moderation' in col.lower()]
        if mod_columns:
            for col in mod_columns:
                filtered_df = filtered_df[filtered_df[col] != True]
            logger.info(f"After removing moderation errors: {len(filtered_df)} rows")
        
        # Remove ties (keep only clear winners)
        if 'winner' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['winner'].isin(['model_a', 'model_b'])]
            logger.info(f"After removing ties: {len(filtered_df)} rows")
        
        logger.info(f"Filtered {initial_count - len(filtered_df)} rows ({100*(initial_count - len(filtered_df))/initial_count:.1f}%)")
        return filtered_df
    
    def build_model_vocabulary(self, arena_df: pd.DataFrame) -> Dict[str, int]:
        """Build vocabulary of model names and filter low-frequency models."""
        logger.info("Building model vocabulary...")
        
        # Count model appearances
        model_counts = Counter()
        if 'model_a' in arena_df.columns and 'model_b' in arena_df.columns:
            model_counts.update(arena_df['model_a'].values)
            model_counts.update(arena_df['model_b'].values)
        
        # Filter models with fewer than 100 appearances
        min_appearances = 100
        frequent_models = {model: count for model, count in model_counts.items() 
                          if count >= min_appearances}
        
        logger.info(f"Found {len(model_counts)} unique models")
        logger.info(f"Filtered to {len(frequent_models)} models with >= {min_appearances} appearances")
        
        # Create vocabulary mapping
        model_vocab = {model: idx for idx, model in enumerate(sorted(frequent_models.keys()))}
        
        # Save vocabulary
        vocab_path = self.cache_dir / "model_vocab.json"
        with open(vocab_path, 'w') as f:
            json.dump(model_vocab, f, indent=2)
        logger.info(f"Model vocabulary saved to {vocab_path}")
        
        return model_vocab
    
    def encode_prompts(self, arena_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Encode unique prompts using sentence-transformers."""
        logger.info("Encoding prompts...")
        
        # Initialize sentence transformer
        model_name = self.config['prompt_encoder_model']
        logger.info(f"Loading sentence transformer: {model_name}")
        self.prompt_encoder = SentenceTransformer(model_name)
        
        # Get unique prompts from conversations
        prompt_texts = set()
        
        # Extract prompts from conversation_a and conversation_b
        for col in ['conversation_a', 'conversation_b']:
            if col in arena_df.columns:
                for idx, conversation in arena_df[col].items():
                    if conversation is not None and len(conversation) > 0:
                        # conversation is a numpy array of dicts
                        # Find the first user message
                        for message in conversation:
                            if isinstance(message, dict) and message.get('role') == 'user':
                                content = message.get('content', '')
                                if content and isinstance(content, str) and content.strip():
                                    prompt_texts.add(content.strip())
                                break  # Only take the first user message
        
        prompt_texts = list(prompt_texts)
        logger.info(f"Found {len(prompt_texts)} unique prompts")
        
        # Encode prompts in batches
        batch_size = 32
        prompt_embeddings = {}
        
        for i in tqdm(range(0, len(prompt_texts), batch_size), desc="Encoding prompts"):
            batch = prompt_texts[i:i+batch_size]
            embeddings = self.prompt_encoder.encode(batch, convert_to_numpy=True)
            
            for prompt, embedding in zip(batch, embeddings):
                prompt_embeddings[prompt] = embedding
        
        # Save embeddings
        embeddings_path = self.cache_dir / "prompt_embeddings.pkl"
        with open(embeddings_path, 'wb') as f:
            pickle.dump(prompt_embeddings, f)
        logger.info(f"Prompt embeddings saved to {embeddings_path}")
        
        return prompt_embeddings
    
    def create_processed_dataframe(self, arena_df: pd.DataFrame, 
                                 model_vocab: Dict[str, int]) -> pd.DataFrame:
        """Create processed DataFrame with model IDs and clean structure."""
        logger.info("Creating processed DataFrame...")
        
        processed_rows = []
        valid_models = set(model_vocab.keys())
        
        for idx, row in tqdm(arena_df.iterrows(), total=len(arena_df), desc="Processing rows"):
            # Skip if models not in vocabulary
            if (row.get('model_a') not in valid_models or 
                row.get('model_b') not in valid_models):
                continue
            
            # Extract prompt text
            prompt_text = self._extract_prompt_text(row)
            if not prompt_text:
                continue
            
            # Determine winner and loser
            if row.get('winner') == 'model_a':
                winner_model = row['model_a']
                loser_model = row['model_b']
            elif row.get('winner') == 'model_b':
                winner_model = row['model_b']
                loser_model = row['model_a']
            else:
                continue  # Skip ties or invalid winners
            
            processed_row = {
                'prompt_text': prompt_text,
                'winner_model_id': model_vocab[winner_model],
                'loser_model_id': model_vocab[loser_model],
                'winner_model_name': winner_model,
                'loser_model_name': loser_model,
                'user_id': row.get('judge', 'unknown')
            }
            processed_rows.append(processed_row)
        
        processed_df = pd.DataFrame(processed_rows)
        logger.info(f"Created processed DataFrame with {len(processed_df)} rows")
        
        # Save processed data
        processed_path = self.cache_dir / "arena_processed.parquet"
        processed_df.to_parquet(processed_path, index=False)
        logger.info(f"Processed data saved to {processed_path}")
        
        return processed_df
    
    def _extract_prompt_text(self, row) -> str:
        """Extract prompt text from conversation format."""
        # Try conversation_a first, then conversation_b
        for col in ['conversation_a', 'conversation_b']:
            if col in row:
                conversation = row[col]
                if conversation is not None and len(conversation) > 0:
                    # Find the first user message
                    for message in conversation:
                        if isinstance(message, dict) and message.get('role') == 'user':
                            content = message.get('content', '')
                            if content and isinstance(content, str) and content.strip():
                                return content.strip()
        return ""
    
    def identify_personalization_users(self, processed_df: pd.DataFrame) -> List[str]:
        """Identify users with sufficient votes for personalization."""
        logger.info("Identifying users for personalization...")
        
        min_votes = self.config['min_votes_per_user']
        user_counts = processed_df['user_id'].value_counts()
        eligible_users = user_counts[user_counts >= min_votes].index.tolist()
        
        logger.info(f"Found {len(eligible_users)} users with >= {min_votes} votes")
        logger.info(f"Total votes from eligible users: {user_counts[eligible_users].sum()}")
        
        # Save user list
        users_path = self.cache_dir / "personalization_users.json"
        with open(users_path, 'w') as f:
            json.dump(eligible_users, f, indent=2)
        logger.info(f"Personalization users saved to {users_path}")
        
        return eligible_users
    
    def run_pipeline(self):
        """Run the complete data preparation pipeline."""
        logger.info("Starting data preparation pipeline...")
        
        # Step 1: Load datasets
        arena_df, hh_df = self.load_datasets()
        
        # Step 2: Filter Arena data
        filtered_arena = self.filter_arena_data(arena_df)
        
        # Step 3: Build model vocabulary
        self.model_vocab = self.build_model_vocabulary(filtered_arena)
        
        # Step 4: Encode prompts
        prompt_embeddings = self.encode_prompts(filtered_arena)
        
        # Step 5: Create processed DataFrame
        self.processed_data = self.create_processed_dataframe(filtered_arena, self.model_vocab)
        
        # Step 6: Identify personalization users
        personalization_users = self.identify_personalization_users(self.processed_data)
        
        logger.info("Data preparation pipeline completed successfully!")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Files created:")
        logger.info(f"  - model_vocab.json")
        logger.info(f"  - prompt_embeddings.pkl")
        logger.info(f"  - arena_processed.parquet")
        logger.info(f"  - personalization_users.json")

def main():
    """Main function to run data preparation."""
    try:
        preparer = DataPreparer()
        preparer.run_pipeline()
        print("✅ Data preparation completed successfully!")
    except Exception as e:
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        print(f"❌ Data preparation failed: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())