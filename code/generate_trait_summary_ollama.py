#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@File    :   experiment_ollama.py
@Version :   1.0
@License :   CC-BY-SA or GPL3
@Desc    :   This script extracts structured morphological traits from textual moth descriptions
             using a local language model served by Ollama (e.g. LLaMA 3.1 or Mistral).
             It outputs clean JSON files with standardized fields.
"""

import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import json
from tqdm import tqdm
import re

# Initialize a local open-source LLM via Ollama
# Requires `ollama run llama3:instruct` (LLaMA 3.1). You can also try:
# - `ollama run llama3:instruct`     (LLaMA 3.1)
# - `ollama run llama3:instruct-3.3` (LLaMA 3.3 — newer, if available in your Ollama installation)
llm = ChatOllama(model="llama3:instruct")

# Fields to extract from the morphological description
FIELDS = [
    "main colors",
    "pattern description",
    "details colors",
    "antennae size",
    "antennae description",
    "antennae colors",
    "head color",
    "abdomen color",
    "wingspan",
    "dimorphism",
    "dimorphism description",
    "forewings size",
    "hindwing size",
    "forewing colors",
    "hindwing colors",
    "similar species"
]

# Prompt template for structured JSON output
prompt = ChatPromptTemplate.from_template("""
You are an expert on insect morphology. From the following moth description, extract the following fields.

If any field is missing or not mentioned, write "NaN". DO NOT INVENT ANYTHING.

Respond **only** in **valid JSON** with exactly these keys:

{fields}

Example of desired format:
```json
{{
  "main colors": ["orange-brown"],
  "pattern description": "brown bands on forewings",
  "details colors": ["brown"],
  "antennae size": "short",
  "antennae description": "feathery",
  "antennae colors": ["brown"],
  "head color": "orange",
  "abdomen color": "light brown",                      
  "wingspan": "10-12mm",
  "dimorphism": "yes",
  "dimorphism description": "females are smaller and have more pronounced patterns",
  "forewings size": "9-12mm",
  "hindwing size": "8-9mm",
  "forewing colors": ["orange", "cream"],
  "hindwing colors": ["gray"],
  "similar species": ["Hepialus lupulinus"]
}}
{context}
""")

# Create the LangChain document-to-JSON pipeline
chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# Load species datasets
df = pd.read_csv("species_summary.csv")  # Requires 'species_name' and 'summary_morpho'
df2 = pd.read_csv("species_summary_v3.csv", encoding="utf-8")  # Optional full descriptions

results = {}

# Process each species description
for _, row in tqdm(df.iterrows(), total=len(df)):
    species = row["species_name"]
    description = row["summary_morpho"]
    document = Document(page_content=description)

    try:
        # Get structured JSON output
        response = chain.invoke({
            "fields": ", ".join(FIELDS),
            "context": [document]
        })

        # Clean output if wrapped in markdown-style code block
        clean_response = re.sub(r"^```json\n?|```$", "", response.strip(), flags=re.MULTILINE)
        result = json.loads(clean_response)

        results[species] = result

    except Exception as e:
        print(f"Error processing {species}: {e}")
        continue

    # Add full descriptions if available
    row2 = df2[df2["species_name"] == species]
    if not row2.empty:
        results[species]["full_morphological_description"] = row2["summary_morpho"].values[0]
        results[species]["full_behavioral_description"] = row2["summary_behavior"].values[0]
    else:
        results[species]["full_morphological_description"] = "No full description available"
        results[species]["full_behavioral_description"] = "No full description available"

# Save results to JSON file
with open("morphological_traits_summary_ollama.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
