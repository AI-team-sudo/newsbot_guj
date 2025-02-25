import streamlit as st
import requests
from googletrans import Translator
from pinecone import Pinecone
from datetime import datetime
import asyncio
import nest_asyncio

# Apply nest_asyncio to handle event loop issues
nest_asyncio.apply()

# Initialize Pinecone
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")
# List all indexes to verify
indexes = pc.list_indexes()
st.write("Available indexes:", indexes)  # This will help debug which indexes are actually available

# Initialize Google Translator
translator = Translator()

# Hugging Face configuration
HUGGING_FACE_TOKEN = "YOUR_HUGGING_FACE_TOKEN"
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}

def extract_tags(text):
    """Extract tags using Mixtral model"""
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

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        tags = response.json()[0]["generated_text"].strip().split(",")
        return [tag.strip() for tag in tags if tag.strip()]
    except Exception as e:
        st.error(f"Error extracting tags: {str(e)}")
        return []

def parse_date(date_str):
    """Parse different date formats"""
    date_formats = [
        "%Y-%m-%d",
        "%b %d, %Y %I:%M %p",
        "%Y-%m-%d %H:%M:%S"
    ]

    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return datetime.now()

# Streamlit app
st.title("Gujarati News Search App")

# Debug information
st.sidebar.write("Debug Information")
if st.sidebar.checkbox("Show available indexes"):
    st.sidebar.write(indexes)

# User input
user_input = st.text_input("Enter your query:")

if st.button("Search"):
    if user_input:
        try:
            # Step 1: Extract tags
            with st.spinner("Extracting tags..."):
                tags = extract_tags(user_input)
                if tags:
                    st.write("Extracted Tags:", tags)
                else:
                    st.warning("No tags were extracted. Please try a different query.")
                    st.stop()

            # Step 2: Translate tags
            with st.spinner("Translating tags..."):
                translated_tags = []
                for tag in tags:
                    try:
                        translated = translator.translate(tag, src='en', dest='gu').text
                        translated_tags.append(translated)
                    except Exception as e:
                        st.warning(f"Could not translate tag '{tag}': {str(e)}")

                if translated_tags:
                    st.write("Translated Tags:", translated_tags)
                else:
                    st.warning("No tags could be translated.")
                    st.stop()

            # Step 3: Search in Pinecone
            with st.spinner("Searching articles..."):
                all_results = []
                namespaces = ["divyabhasker", "sandesh"]

                # Get the correct index name from available indexes
                index_name = indexes[0].name if indexes else None
                if not index_name:
                    st.error("No Pinecone indexes available")
                    st.stop()

                index = pc.Index(index_name)

                for namespace in namespaces:
                    try:
                        for tag in translated_tags:
                            query_response = index.query(
                                vector=[0.0] * 1536,  # Adjust dimension as needed
                                filter={"text": {"$contains": tag}},
                                top_k=10,
                                namespace=namespace,
                                include_metadata=True
                            )
                            all_results.extend(query_response.matches)
                    except Exception as e:
                        st.warning(f"Error searching in namespace {namespace}: {str(e)}")

            # Step 4: Display results
            if all_results:
                sorted_results = sorted(
                    all_results,
                    key=lambda x: parse_date(x.metadata.get("date", "1900-01-01")),
                    reverse=True
                )

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
                st.warning("No articles found matching the search criteria.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your Pinecone configuration and try again.")
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
