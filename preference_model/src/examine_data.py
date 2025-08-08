#!/usr/bin/env python3
"""
Quick script to examine the Arena dataset structure
"""

from datasets import load_dataset
import pandas as pd

# Load a small sample
print("Loading Arena dataset sample...")
arena_dataset = load_dataset("lmsys/chatbot_arena_conversations", split='train[:100]')
arena_df = arena_dataset.to_pandas()

print(f"Dataset shape: {arena_df.shape}")
print(f"Columns: {list(arena_df.columns)}")
print("\nSample row:")
print(arena_df.iloc[0].to_dict())

print("\nColumn types:")
print(arena_df.dtypes)

print("\nFirst few values for key columns:")
for col in ['model_a', 'model_b', 'winner', 'conversation_a', 'conversation_b']:
    if col in arena_df.columns:
        print(f"\n{col}:")
        print(arena_df[col].head(2).values)