import streamlit as st
import requests
from googletrans import Translator
from pinecone import Pinecone
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize Pinecone
pc = Pinecone(st.secrets["PINECONE_API_KEY"])
index = pc.Index("newsbot2")

# Initialize Google Translator
translator = Translator()

# Hugging Face model initialization
HUGGING_FACE_TOKEN = st.secrets["HUGGING_FACE_TOKEN"]
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

def extract_tags(text):
    """
    Extract tags using Mixtral model through Hugging Face API
    """
    prompt = f"""
    Task: Extract relevant keywords or tags from the following text. Return only the tags as a comma-separated list.

    Text: {text}

    Tags:"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.3,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        tags = response.json()[0]["generated_text"].strip().split(",")
        return [tag.strip() for tag in tags if tag.strip()]
    else:
        raise Exception(f"Error from API: {response.text}")

def parse_date(date_str):
    """
    Parse different date formats
    """
    try:
        # Try YYYY-MM-DD format
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        try:
            # Try "Feb DD, YYYY HH:MM pm" format
            return datetime.strptime(date_str, "%b %d, %Y %I:%M %p")
        except ValueError:
            return datetime.now()  # Return current date as fallback

# Streamlit app
st.title("Gujarati News Search App")
st.write("Enter a query to search for relevant Gujarati news articles.")

# User input
user_input = st.text_input("Enter your query:")

if st.button("Search"):
    if user_input:
        try:
            # Step 1: Extract tags using Mixtral
            with st.spinner("Extracting tags from your input..."):
                tags = extract_tags(user_input)
                st.write("Extracted Tags:", tags)

            # Step 2: Translate tags to Gujarati
            with st.spinner("Translating tags to Gujarati..."):
                translated_tags = [translator.translate(tag, src="en", dest="gu").text for tag in tags]
                st.write("Translated Tags:", translated_tags)

            # Step 3: Search Pinecone index
            with st.spinner("Searching for articles..."):
                all_results = []

                # Search in each namespace
                namespaces = ["divyabhasker", "sandesh"]
                for namespace in namespaces:
                    for tag in translated_tags:
                        query_response = index.query(
                            vector=[0.0] * 1536,  # Replace with actual vector dimension
                            filter={"text": {"$contains": tag}},
                            top_k=10,
                            namespace=namespace,
                            include_metadata=True
                        )
                        all_results.extend(query_response.matches)

            # Step 4: Sort results by date
            with st.spinner("Sorting results..."):
                sorted_results = sorted(
                    all_results,
                    key=lambda x: parse_date(x.metadata.get("date", "1900-01-01")),
                    reverse=True
                )

            # Step 5: Display results
            if sorted_results:
                st.write(f"Found {len(sorted_results)} articles:")
                for result in sorted_results:
                    with st.expander(f"Article from {result.metadata.get('date', 'Unknown date')}"):
                        st.write(f"**Date:** {result.metadata.get('date', 'Unknown date')}")
                        if 'title' in result.metadata and result.metadata['title']:
                            st.write(f"**Title:** {result.metadata['title']}")
                        st.write(f"**Content:** {result.metadata.get('text', 'No content available')}")
                        if 'link' in result.metadata and result.metadata['link']:
                            st.write(f"[Read more]({result.metadata['link']})")
            else:
                st.warning("No articles found for the given query.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please enter a query.")

# Add CSS for better formatting
st.markdown("""
    <style>
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
