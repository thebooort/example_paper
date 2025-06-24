#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   get_summary.py
@Time    :   2025/05/07 14:01:43
@Author  :   Bart Ortiz 
@Version :   1.0
@Contact :   bortiz@ugr.es
@License :   CC-BY-SA or GPL3
@Desc    :   This script generates morphological and behavioral summaries for insect species
             by retrieving and processing text descriptions using a language model via LangChain.
'''

import pandas as pd
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import getpass
import os
from langchain_core.documents import Document

# Set your OpenAI API key (ideally use environment variables in production)
os.environ["OPENAI_API_KEY"] = "-your key-"

# Initialize the LLM (GPT-4o-mini in this case) using a LangChain wrapper
from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Load species description dataset
df_descriptions = pd.read_csv("species_descriptions_translated_cleaned.csv")


def get_all_descriptions(species_name):
    """
    Retrieve all available descriptions for a given species.

    Only sources 'nrm.se' and 'artfakta.se' use the 'description_translated' column.
    All others use the original 'description' column.

    Returns a list of non-empty description strings.
    """
    descriptions = []
    if species_name in df_descriptions["taxon"].values:
        filtered_df = df_descriptions[df_descriptions["taxon"] == species_name]
        for source in filtered_df["source"]:
            if source in ["nrm.se", "artfakta.se"]:
                desc = filtered_df[filtered_df["source"] == source]["description_translated"].values[0]
                if desc:
                    descriptions.append(desc)
            else:
                desc = filtered_df[filtered_df["source"] == source]["description"].values[0]
                if desc:
                    descriptions.append(desc)
        # Remove empty or invalid entries
        descriptions = [desc for desc in descriptions if isinstance(desc, str) and desc.strip() != ""]
        return descriptions
    else:
        print(f"Species '{species_name}' not found in the dataset.")
        return None


def summary_behavior_from_text(text):
    """
    Use the LLM to generate a behavioral summary from a list of input descriptions.

    Focus is on behavior only: months, flying patterns, mating behavior, activity period, food preferences.
    """
    documents = [Document(page_content=desc, metadata={"title": "species_description"}) for desc in text]

    prompt_template = (
        "You are an entomological assistant. Your task is to summarize scientific species descriptions by extracting only behavioral details "
        "(months, flying habits, mating behavior, activity periods, food preferences, plant species). Disregard information about color, size, or morphology.\n\n"
        "Example:\n"
        "Input: 'This species has brown wings with red spots. Adults fly during dusk and are attracted to light. They are mostly active in spring and are "
        "known to rest on tree trunks during the day.'\n"
        "Output: 'Adults fly during dusk, are attracted to light, and rest on tree trunks during the day. They are active in spring.'\n\n"
        "Now summarize the following input in the same way, keeping only the behavioral description and writing a single short paragraph:\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = create_stuff_documents_chain(llm, prompt)
    result = chain.invoke({"context": documents})
    return result


def summary_morpho_from_text(text):
    """
    Use the LLM to generate a morphological summary from a list of input descriptions.

    Focus is on adult morphology only: color, shape, wing patterns.
    Explicitly ignore behavior, phenology, larval stages, months, wingspan, etc.
    """
    documents = [Document(page_content=desc, metadata={"title": "species_description"}) for desc in text]

    prompt_template = (
        "You are an entomological assistant. Your task is to summarize scientific species descriptions by extracting only "
        "morphological details (e.g., color, antennae shape, wing patterns). Merge if they appear repeated. Remove information about behavior, reproduction, "
        "phenology, eggs, or larvae, months, or wingspan mm. Do not put descriptions of the larvae, non adult specimens\n\n"
        "Example:\n"
        "Input: 'This species is often found in shaded forested areas. The adult has bright green forewings of 12 - 11mm with two dark bands, and a yellow abdomen. "
        "Males are slightly smaller than females. They usually fly at night during June and July. The larvae are whitish with a dark brown head and neck plate, and eat plants'\n"
        "Output: 'Adults have bright green forewings with two dark bands, a yellow abdomen.'\n\n"
        "Now summarize the following input in the same way, keeping only the morphological description and writing a single short paragraph:\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = create_stuff_documents_chain(llm, prompt)
    result = chain.invoke({"context": documents})
    return result


def get_summary_morpho(species_name):
    """
    Generate morphological summary for a species if found in the dataset.
    """
    filtered_df = df_descriptions[df_descriptions["taxon"] == species_name]
    if not filtered_df.empty:
        descriptive_text = get_all_descriptions(species_name)
        summary = summary_morpho_from_text(descriptive_text)
        return summary
    else:
        print(f"Species '{species_name}' not found in the dataset.")
        return None


def get_summarty_behavior(species_name):
    """
    Generate behavioral summary for a species if found in the dataset.
    """
    filtered_df = df_descriptions[df_descriptions["taxon"] == species_name]
    if not filtered_df.empty:
        descriptive_text = get_all_descriptions(species_name)
        summary = summary_behavior_from_text(descriptive_text)
        return summary
    else:
        print(f"Species '{species_name}' not found in the dataset.")
        return None


from tqdm import tqdm


if __name__ == "__main__":
    # Load previous results (if any)
    df_results = pd.read_csv("species_summary.csv")
    
    # Get all species in the descriptions file
    species_list = df_descriptions["taxon"].unique()
    
    # Skip species already processed
    already_processed = df_results["species_name"].unique()
    species_list = [species for species in species_list if species not in already_processed]

    print(f"Species to process: {species_list}")

    # Iterate over all species and generate summaries
    for species in tqdm(species_list, desc="Processing species", unit="species"):
        all_descriptions = get_all_descriptions(species)
        print(f"All descriptions for {species}:")

        # Generate morphological summary
        summary = get_summary_morpho(species)
        print(f"Summary of the dataset for {species}: Morphology")
        print(summary)

        # Generate behavioral summary
        summary2 = get_summarty_behavior(species)
        print(f"Summary of the dataset for {species}: Behavior")
        print(summary2)

        # Save result for current species
        results = {
            'species_name': species,
            'summary_morpho': summary,
            'summary_behavior': summary2
        }

        # Append to result dataframe and save
        df_results = pd.concat([df_results, pd.DataFrame([results])], ignore_index=True)
        df_results.to_csv("species_summary_v3.csv", index=False)
