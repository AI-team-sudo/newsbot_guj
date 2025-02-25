import streamlit as st
import requests
from googletrans import Translator
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime

# Initialize Pinecone with new method
pc = Pinecone(
    api_key=st.secrets["PINECONE_API_KEY"],
)
index = pc.Index("newsbot2")

# Initialize Google Translator
translator = Translator()

# Hugging Face API details
HUGGING_FACE_API_URL = "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer {st.secrets["HUGGING_FACE_TOKEN"]}"}

# Function to parse different date formats
def parse_date(date_str):
    date_formats = [
        "%Y-%m-%d",
        "%b %d, %Y %I:%M %p"  # For "Feb 22, 2025 05:46 pm" format
    ]

    for date_format in date_formats:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    return None

# Streamlit app
st.title("Gujarati News Search App")

# User input
user_input = st.text_area("Enter your query:", placeholder="Type something...")

if st.button("Search"):
    if not user_input.strip():
        st.error("Please enter a valid query.")
    else:
        with st.spinner("Extracting tags from your input..."):
            # Step 1: Extract tags using Mixtral
            response = requests.post(HUGGING_FACE_API_URL, headers=headers, json={"inputs": user_input})
            if response.status_code == 200:
                tags = response.json()
                st.write("Extracted Tags:", tags)
            else:
                st.error("Error extracting tags. Please check your Hugging Face API configuration.")
                st.stop()

        with st.spinner("Translating tags to Gujarati..."):
            # Step 2: Translate tags to Gujarati
            translated_tags = [translator.translate(tag, src="en", dest="gu").text for tag in tags]
            st.write("Translated Tags:", translated_tags)

        with st.spinner("Searching for articles..."):
            # Step 3: Search Pinecone index
            query_results = []
            for namespace in ["divyabhasker", "gujratsamachar", "sandesh"]:
                for tag in translated_tags:
                    try:
                        results = index.query(
                            namespace=namespace,
                            top_k=10,
                            include_metadata=True,
                            vector=[0] * 512,  # Replace with actual vector representation
                            filter={"text": {"$contains": tag}}
                        )
                        query_results.extend(results["matches"])
                    except Exception as e:
                        st.warning(f"Error searching in namespace {namespace}: {str(e)}")

            # Step 4: Remove duplicates based on text content
            seen_texts = set()
            unique_results = []
            for result in query_results:
                text = result["metadata"]["text"]
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_results.append(result)

            # Step 5: Sort results by date
            sorted_results = sorted(
                unique_results,
                key=lambda x: parse_date(x["metadata"]["date"]) or datetime.min,
                reverse=True
            )

            # Step 6: Display results
            if sorted_results:
                st.write(f"Found {len(sorted_results)} unique articles:")
                for result in sorted_results:
                    with st.expander(f"{result['metadata'].get('title', 'No Title')} - {result['metadata']['date']}"):
                        st.write("**Date:**", result["metadata"]["date"])
                        st.write("**Content:**", result["metadata"]["text"])
                        if "link" in result["metadata"]:
                            st.write("**Link:**", result["metadata"]["link"])
                        st.write("**Source:**", result["metadata"]["filename"])
            else:
                st.write("No articles found matching your query.")
