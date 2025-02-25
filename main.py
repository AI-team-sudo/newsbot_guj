import streamlit as st
import requests
from googletrans import Translator
from pinecone import Pinecone
from datetime import datetime
import asyncio
import nest_asyncio

# Apply nest_asyncio to handle event loop issues
nest_asyncio.apply()

# Secure configuration using st.secrets
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    HUGGING_FACE_TOKEN = st.secrets["HUGGING_FACE_TOKEN"]
except Exception as e:
    st.error("Missing required secrets. Please check your Streamlit secrets configuration.")
    st.stop()

# Initialize Pinecone with error handling
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    indexes = pc.list_indexes()
except Exception as e:
    st.error(f"Failed to initialize Pinecone: {str(e)}")
    st.stop()

# Initialize Google Translator
translator = Translator()

# Hugging Face configuration
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
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        tags = response.json()[0]["generated_text"].strip().split(",")
        return [tag.strip() for tag in tags if tag.strip()]
    except requests.Timeout:
        st.error("Request to Hugging Face API timed out. Please try again.")
        return []
    except requests.RequestException as e:
        st.error(f"Error accessing Hugging Face API: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Error extracting tags: {str(e)}")
        return []

def parse_date(date_str):
    """Parse different date formats with error handling"""
    if not date_str:
        return datetime.now()

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

    st.warning(f"Could not parse date format: {date_str}")
    return datetime.now()

# Streamlit app
st.title("Gujarati News Search App")

# Debug information in development
if st.secrets.get("ENVIRONMENT") == "development":
    st.sidebar.write("Debug Information")
    if st.sidebar.checkbox("Show available indexes"):
        st.sidebar.write(indexes)

# User input
user_input = st.text_input("Enter your query:")

if st.button("Search"):
    if not user_input:
        st.error("Please enter a query.")
        st.stop()

    try:
        # Step 1: Extract tags
        with st.spinner("Extracting tags..."):
            tags = extract_tags(user_input)
            if not tags:
                st.warning("No tags were extracted. Please try a different query.")
                st.stop()
            st.write("Extracted Tags:", tags)

        # Step 2: Translate tags
        with st.spinner("Translating tags..."):
            translated_tags = []
            for tag in tags:
                try:
                    translated = translator.translate(tag, src='en', dest='gu').text
                    translated_tags.append(translated)
                except Exception as e:
                    st.warning(f"Could not translate tag '{tag}': {str(e)}")

            if not translated_tags:
                st.warning("No tags could be translated.")
                st.stop()
            st.write("Translated Tags:", translated_tags)

        # Step 3: Search in Pinecone
        with st.spinner("Searching articles..."):
            all_results = []
            namespaces = ["sandesh"]

            index_name = indexes[0].name if indexes else None
            if not index_name:
                st.error("No Pinecone indexes available")
                st.stop()

            index = pc.Index(index_name)

            for namespace in namespaces:
                try:
                    for tag in translated_tags:
                        query_response = index.query(
                            vector=[0.0] * 1536,
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
        if st.secrets.get("ENVIRONMENT") == "development":
            st.exception(e)

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
