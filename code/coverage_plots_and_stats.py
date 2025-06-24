#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Version :   1.0
@License :   CC-BY-SA or GPL3
@Desc    :   This script analyzes and cleans morphological and behavioral descriptions
of UK insect taxa at species, family, and genus levels. It generates statistics
and histograms showing the distribution of word counts in descriptions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# Global plot settings
plt.rcParams.update({
    "text.usetex": False,
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})


def clean_text(text: str) -> str:
    """
    Normalize a string by removing unnecessary whitespace and sentences
    containing low-informative phrases like 'not detailed'.
    """
    if pd.isna(text):
        return ""
    # Remove sentences with 'not detailed'
    sentences = [s for s in str(text).split('. ') if 'not detailed' not in s]
    cleaned = '. '.join(sentences)
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Replace multiple spaces with single space
    return cleaned.strip()


def clean_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the text was not filtered previously this function
    Remove rows containing generic or non-informative phrases
    from both morphological and behavioral descriptions that are likely to be 
    missed outcomes from the LLM
    """
    patterns_to_remove = [
        'please|Please', 'The provided text', 'provided',
        'provide', 'the text', 'The text', 'given text'
    ]
    for pattern in patterns_to_remove:
        df = df[~df['summary_morpho'].str.contains(pattern, na=False)]
        df = df[~df['summary_behavior'].str.contains(pattern, na=False)]

    df['summary_morpho'] = df['summary_morpho'].apply(clean_text)
    df['summary_behavior'] = df['summary_behavior'].apply(clean_text)
    return df


def analyze_descriptions(df: pd.DataFrame, level_name: str, save_path: str = None) -> None:
    """
    Generate statistics and histograms of word counts for morphological
    and behavioral descriptions.
    
    Parameters:
    - df: DataFrame containing the descriptions
    - level_name: name of the taxonomic level (e.g., 'Species', 'Family')
    - save_path: optional path to save the histogram as an image
    """
    df['morpho_description_size'] = df['summary_morpho'].apply(lambda x: len(str(x).split()))
    df['behavioral_description_size'] = df['summary_behavior'].apply(lambda x: len(str(x).split()))

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(df['morpho_description_size'], bins=100, alpha=0.5, label='Morpho Description', color='blue')
    plt.hist(df['behavioral_description_size'], bins=100, alpha=0.5, label='Behavioral Description', color='orange')
    plt.title(f'Histogram of Description Sizes ({level_name} UK)')
    plt.xlabel('Description Size (number of words)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    # plt.show()  # Uncomment if running interactively

    # Print summary statistics
    print(f"\n=== {level_name.upper()} ===")
    morpho_avg = df['morpho_description_size'].mean()
    morpho_std = df['morpho_description_size'].std()
    behavioral_avg = df['behavioral_description_size'].mean()
    behavioral_std = df['behavioral_description_size'].std()
    print(f'Morpho Description - Average: {morpho_avg:.2f}, Std: {morpho_std:.2f}')
    print(f'Behavioral Description - Average: {behavioral_avg:.2f}, Std: {behavioral_std:.2f}')

    morpho_short = df[df['morpho_description_size'] < 20].shape[0]
    behavioral_short = df[df['behavioral_description_size'] < 20].shape[0]
    morpho_pct = (morpho_short / len(df)) * 100
    behavioral_pct = (behavioral_short / len(df)) * 100
    print(f'Morpho Description - Less than 20 words: {morpho_short} ({morpho_pct:.2f}%)')
    print(f'Behavioral Description - Less than 20 words: {behavioral_short} ({behavioral_pct:.2f}%)')


if __name__ == "__main__":
    datasets = [
        {
            "csv": "species_summary.csv",
            "label": "Species",
            "output": "species_histogram.png",
            "clean": True
        },
        {
            "csv": "famillies_summary.csv",
            "label": "Family",
            "output": "families_histogram.png",
            "clean": True,
            "export": "families_summary.csv"
        },
        {
            "csv": "genus_summary.csv",
            "label": "Genus",
            "output": "genus_histogram.png",
            "clean": True,
            "export": "genus_summary.csv"
        }
    ]

    for data in datasets:
        print(f"\nProcessing: {data['label']}")
        df = pd.read_csv(data["csv"])

        if data.get("clean", False):
            df = clean_descriptions(df)
            if "export" in data:
                df.to_csv(data["export"], index=False)

        analyze_descriptions(df, data["label"], data["output"])
