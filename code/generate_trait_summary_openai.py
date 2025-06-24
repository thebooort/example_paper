import pandas as pd
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
import os
import json
from tqdm import tqdm
import re

# Make sure your OpenAI API key is set as an environment variable
# Do NOT hardcode the key in your script!
# Example (Linux/Mac): export OPENAI_API_KEY='your-key-here'
# Example (Windows): set OPENAI_API_KEY=your-key-here

# Initialize the GPT-4o-mini model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Fields to extract from the insect morphological description
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

# Prompt template with an example for the model to follow
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

# Create the document analysis chain
chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# Load dataset with species and morphological descriptions
df = pd.read_csv("species_summary.csv")  # Must contain 'species_name' and 'summary_morpho'
df2 = pd.read_csv("species_summary_v3.csv", encoding="utf-8")  # For merging full descriptions

results = {}

# Iterate through each species and process the morphological summary
for _, row in tqdm(df.iterrows(), total=len(df)):
    species = row["species_name"]
    description = row["summary_morpho"]
    document = Document(page_content=description)

    try:
        # Generate structured JSON from description
        response = chain.invoke({
            "fields": ", ".join(FIELDS),
            "context": [document]
        })

        # Clean response if wrapped in triple backticks
        clean_response = re.sub(r"^```json\n?|```$", "", response.strip(), flags=re.MULTILINE)
        result = json.loads(clean_response)

        results[species] = result
        print(results[species])   

    except Exception as e:
        print(f"Error processing {species}: {e}")
        continue

    # Merge full morphological and behavioral descriptions if available
    row2 = df2[df2["species_name"] == species]
    if not row2.empty:
        results[species]["full_morphological_description"] = row2["summary_morpho"].values[0]
        results[species]["full_behavioral_description"] = row2["summary_behavior"].values[0]
    else:
        results[species]["full_morphological_description"] = "No full description available"
        results[species]["full_behavioral_description"] = "No full description available"

    print(results)

# Save all extracted results to JSON
with open("morphological_traits_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
