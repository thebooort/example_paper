#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@File    :   experiment.py
@Version :   1.0
@License :   CC-BY-SA or GPL3
@Desc    :   This script computes similarity statistics between textual descriptions of insect species.
             It compares morphological and behavioral summaries using sentence embeddings.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from scipy.stats import pearsonr, ttest_ind
from tqdm import tqdm

# --- Load data ---

# Load the CSV containing species summaries
df = pd.read_csv("species_summary_v4.csv")

# Drop rows where either morphological or behavioral summaries are missing
df = df.dropna(subset=['summary_morpho', 'summary_behavior'])

# Optional: limit dataset size to reduce memory usage
# df = df.sample(n=500, random_state=42)

# --- Compute text lengths ---

# Number of words in each morphological description
df['len_morpho'] = df['summary_morpho'].apply(lambda x: len(str(x).split()))
# Number of words in each behavioral description
df['len_behavior'] = df['summary_behavior'].apply(lambda x: len(str(x).split()))

# --- Load sentence embedding model ---

# Using a compact pre-trained sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings for both types of summaries
emb_morpho = model.encode(df['summary_morpho'].tolist(), convert_to_tensor=True)
emb_behavior = model.encode(df['summary_behavior'].tolist(), convert_to_tensor=True)

# --- Compute pairwise cosine similarities ---

# Cosine similarity between each pair of morphological embeddings
sim_morpho_matrix = util.cos_sim(emb_morpho, emb_morpho).cpu().numpy()
# Cosine similarity between each pair of behavioral embeddings
sim_behavior_matrix = util.cos_sim(emb_behavior, emb_behavior).cpu().numpy()

# Consider only the upper triangle of the similarity matrices (excluding diagonal)
triu_indices = np.triu_indices_from(sim_morpho_matrix, k=1)

# Extract similarity values for each type
morpho_sims = sim_morpho_matrix[triu_indices]
behavior_sims = sim_behavior_matrix[triu_indices]

# --- Compute mean and std of similarities ---

mean_morpho = morpho_sims.mean()
std_morpho = morpho_sims.std()
mean_behavior = behavior_sims.mean()
std_behavior = behavior_sims.std()

print(f"Morpho similarity - mean: {mean_morpho:.4f}, std: {std_morpho:.4f}")
print(f"Behavior similarity - mean: {mean_behavior:.4f}, std: {std_behavior:.4f}")

# --- Correlation between text length and average similarity ---

# Average similarity of each text with all others
avg_sim_morpho_per_text = sim_morpho_matrix.mean(axis=1)
avg_sim_behavior_per_text = sim_behavior_matrix.mean(axis=1)

# Pearson correlation between length and average similarity
corr_morpho_len, pval_morpho = pearsonr(df['len_morpho'], avg_sim_morpho_per_text)
corr_behavior_len, pval_behavior = pearsonr(df['len_behavior'], avg_sim_behavior_per_text)

print(f"Correlation (morpho length vs avg similarity): {corr_morpho_len:.4f} (p = {pval_morpho:.4e})")
print(f"Correlation (behavior length vs avg similarity): {corr_behavior_len:.4f} (p = {pval_behavior:.4e})")

# --- T-test: short vs long texts ---

# Threshold to define "short" texts (in words)
threshold = 20

# Store average similarity per text in the dataframe
df['avg_sim_morpho'] = avg_sim_morpho_per_text
df['avg_sim_behavior'] = avg_sim_behavior_per_text

# Split data based on length threshold
short_morpho = df[df['len_morpho'] < threshold]['avg_sim_morpho']
long_morpho = df[df['len_morpho'] >= threshold]['avg_sim_morpho']

short_behavior = df[df['len_behavior'] < threshold]['avg_sim_behavior']
long_behavior = df[df['len_behavior'] >= threshold]['avg_sim_behavior']

# Display mean similarities
print(f"Avg similarity (short morpho): {short_morpho.mean():.4f}")
print(f"Avg similarity (long morpho): {long_morpho.mean():.4f}")
print(f"Avg similarity (short behavior): {short_behavior.mean():.4f}")
print(f"Avg similarity (long behavior): {long_behavior.mean():.4f}")

# Perform Welch’s t-test (assumes unequal variances)
t_morpho, p_morpho = ttest_ind(short_morpho, long_morpho, equal_var=False)
t_behavior, p_behavior = ttest_ind(short_behavior, long_behavior, equal_var=False)

print(f"T-test morpho: t = {t_morpho:.4f}, p = {p_morpho:.4e}")
print(f"T-test behavior: t = {t_behavior:.4f}, p = {p_behavior:.4e}")