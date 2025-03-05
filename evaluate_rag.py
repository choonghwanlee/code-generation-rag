'''
Steps

1. Load the ArXiVDLInstruct dataset, obtain the in-domain subset from vector DB and an out-of-domain "validation" subset
2. Use GPT-4o to write code for the 'instructions' column in the validation subset <= performance *without* RAG
3. Use voyage-code-3 to embed the 'instructions' column of the validation subset as a query
4. calculate cosine similarity with code embeddings in vector DB, retrieve top 5 by metadata ID
5. Use metadata ID to fetch description and code chunk from in-domain subset 
6. Use GPT-4o to write code with the description and code chunks added
'''
 
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from matplotlib import pyplot as plt
import voyageai
import numpy as np
from openai import OpenAI
import prompts
from itertools import chain 

def evaluate_code(predicted: str, ground_truth: str):
    '''
    Evaluate predicted source code and ground truth source code using code-to-code similiarity.  
    '''
    predicted_embed = vo.embed([predicted], model='voyage-code-3', input_type='query', output_dimension=256).embeddings[0]
    groundtruth_embed = vo.embed([ground_truth], model='voyage-code-3', input_type='document', output_dimension=256).embeddings[0]
    similarity = np.dot(groundtruth_embed, predicted_embed)
    return similarity

def generate_code_vanilla(example):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages= [
            {"role": "system", "content": prompts.CODE_GENERATION_VANILLA_SYSTEM},
            {"role": "user", "content": prompts.CODE_GENERATION_VANILLA.format(instruction=example['prompt'])}
        ],
        max_completion_tokens = 1024,
        temperature = 0,
    ) 
    completion = response.choices[0].message.content
    example['vanilla_output'] = completion
    example['vanilla_similarity'] = evaluate_code(completion, example['function'])
    return example

def generate_code_rag(example):
    top_k_indices = retrieve_k_most_similar(example['prompt'], vectordb_embeddings, vectordb_ids, vo)
    rag_prompt = fetch_and_format(top_k_indices, vectordb_subset)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages= [
            {"role": "system", "content": prompts.CODE_GENERATION_RAG_SYSTEM},
            {"role": "user", "content": prompts.CODE_GENERATION_RAG.format(rag_retrieved= rag_prompt, instruction=example['prompt'])}
        ],
        max_completion_tokens = 1024,
        temperature = 0,
    ) 
    completion = response.choices[0].message.content
    example['rag_output'] = completion
    example['rag_similarity'] = evaluate_code(completion, example['function'])
    return example

def retrieve_k_most_similar(query, vectordb_embeddings, vectordb_ids, client, k=3):
    query_embed = client.embed([query], model='voyage-code-3', input_type='query', output_dimension=256).embeddings[0]
    similarities = np.array(query_embed) @ np.array(vectordb_embeddings).T
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return np.array(vectordb_ids)[top_k_indices]

def fetch_and_format(indices, vectordb_subset):
    ### given an array of indices to fetch from, fetch the data and format it in few-shot format
    retrieved = vectordb_subset.select(indices)
    few_shot_prompt = ""
    for desc, snippet in zip(retrieved['description'], retrieved['function']):
        few_shot_prompt += f"**Description**: {desc}\n**Code Snippet**: {snippet}\n\n"
    return few_shot_prompt


if __name__ == "__main__":
    load_dotenv()

    pc = Pinecone(api_key = os.getenv("PINECONE_API_KEY"))

    index = pc.Index('arxiv-code-voyage')

    ## get list of all IDs
    nested_ids = list(index.list())
    all_ids = list(chain(*nested_ids))

    ## batch fetch vectors from pinecone
    all_vectors = {}
    for i in range(0, len(all_ids), 100):
        batch_ids = all_ids[i : i + 100]
        fetched = index.fetch(batch_ids)
        all_vectors.update(fetched.vectors)

    ## store vector DB embeddings in a list, along with the IDs in a parallel list
    vectordb_embeddings = []
    vectordb_ids = []
    for id, embeddings in all_vectors.items():
        vectordb_ids.append(int(id.split('_')[1]))
        vectordb_embeddings.append(embeddings.values)

    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    client = OpenAI()

    dataset = load_dataset("AlgorithmicResearchGroup/ArXivDLInstruct", split='train')

    filtered_dataset = dataset.filter(lambda example: len(example["function"]) <= 1000)

    vectordb_subset = filtered_dataset.shuffle(seed=42).select(range(100000))

    validation_subset = filtered_dataset.shuffle(seed=42).select(range(100000, 100100))

    without_rag = validation_subset.map(generate_code_vanilla, batched=False)
    with_rag = validation_subset.map(generate_code_rag, batched=False)
    print("Without RAG: ", np.mean(without_rag['vanilla_similarity']))
    print("With RAG: ", np.mean(with_rag['rag_similarity']))

    without_rag.push_to_hub("jasonhwan/ArXiVDLInstruct-embed", "Vanilla", private=True)
    with_rag.push_to_hub("jasonhwan/ArXiVDLInstruct-embed", "RAG", private=True)


