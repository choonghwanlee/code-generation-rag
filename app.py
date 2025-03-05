import os
from dotenv import load_dotenv
from pinecone import Pinecone
import streamlit as st
import prompts
from openai import OpenAI
import prompts
from itertools import chain 
import voyageai
import numpy as np
from datasets import load_dataset

load_dotenv()
 
# Use st.cache_resource for loading resources that should persist across sessions
@st.cache_resource
def load_pinecone_data():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index('arxiv-code-voyage')
    ## get list of all IDs
    nested_ids = list(index.list())
    all_ids = list(chain(*nested_ids))

    ## batch fetch vectors from pinecone
    all_vectors = {}
    for i in range(0, 10000, 100):
        batch_ids = all_ids[i : i + 100]
        fetched = index.fetch(batch_ids)
        all_vectors.update(fetched.vectors)

    ## store vector DB embeddings in a list, along with the IDs in a parallel list
    vectordb_embeddings = []
    vectordb_ids = []
    for id, embeddings in all_vectors.items():
        vectordb_ids.append(int(id.split('_')[1]))
        vectordb_embeddings.append(embeddings.values)
    
    return vectordb_embeddings, vectordb_ids

@st.cache_resource
def load_dataset_and_subset():
    dataset = load_dataset("AlgorithmicResearchGroup/ArXivDLInstruct", split='train')
    filtered_dataset = dataset.filter(lambda example: len(example["function"]) <= 1000)
    return filtered_dataset.shuffle(seed=42).select(range(100000))

@st.cache_resource
def initialize_clients():
    vo = voyageai.Client(api_key=st.secrets["VOYAGE_API_KEY"])
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    return vo, client

# Initialize global resources on app start
vectordb_embeddings, vectordb_ids = load_pinecone_data()
vectordb_subset = load_dataset_and_subset()
vo, client = initialize_clients()

def retrieve_k_most_similar(query, k=3):
    query_embed = vo.embed([query], model='voyage-code-3', input_type='query', output_dimension=256).embeddings[0]
    similarities = np.array(query_embed) @ np.array(vectordb_embeddings).T
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return np.array(vectordb_ids)[top_k_indices]

def fetch_and_format(indices):
    ### given an array of indices to fetch from, fetch the data and format it in few-shot format
    retrieved = vectordb_subset.select(indices)
    few_shot_prompt = ""
    for desc, snippet in zip(retrieved['description'], retrieved['function']):
        few_shot_prompt += f"**Description**: {desc}\n**Code Snippet**: {snippet}\n\n"
    return few_shot_prompt

# Simple function for chatbot response (can be replaced with an actual model)
def chatbot_response(user_input):
    # For now, a simple response generation
    top_k_indices = retrieve_k_most_similar(user_input)
    rag_prompt = fetch_and_format(top_k_indices)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages= [
            {"role": "system", "content": prompts.CODE_GENERATION_RAG_SYSTEM},
            {"role": "user", "content": prompts.CODE_GENERATION_RAG.format(rag_retrieved= rag_prompt, instruction=user_input)}
        ],
        max_completion_tokens = 1024,
        temperature = 0,
    ) 
    completion = response.choices[0].message.content
    return completion


# Streamlit interface
def run_chatbot():
    st.title("Code Generation RAG System")

    # Display a textbox for user input
    user_input = st.text_input("Your Instructions: ", "", help='Enter instructions on research code to generate!')

    if user_input:
        # Display user message
        st.write(f"You: {user_input}")
        
        # Get and display chatbot response
        bot_response = chatbot_response(user_input)
        st.code(bot_response, language='python')

# Run the chatbot app
if __name__ == "__main__":
    run_chatbot()
