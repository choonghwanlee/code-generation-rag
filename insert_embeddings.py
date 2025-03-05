import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
import voyageai

load_dotenv()
 
pc = Pinecone(api_key = os.getenv("PINECONE_API_KEY"))

index = pc.Index('arxiv-code-voyage')

vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

dataset = load_dataset("AlgorithmicResearchGroup/ArXivDLInstruct", split='train')

filtered_dataset = dataset.filter(lambda example: len(example["function"]) <= 1000)

filtered_subset = filtered_dataset.shuffle(seed=42).select(range(100000))

# Function to get embeddings from Anthropic
def embed_text(examples: dict):
    texts = examples['function']
    results = vo.embed(texts, model='voyage-code-3', input_type='document', output_dimension=256)
    examples["embedding"] = results.embeddings
    return examples

embedded_dataset = filtered_subset.map(
    embed_text,
    batched=True,
    batch_size = 500,
    desc="Generating embeddings",
)

# Upload embeddings to Pinecone
print("Uploading embeddings to Pinecone...")
vectors_to_upsert = []
batch_size = 100 

for i, example in enumerate(tqdm(embedded_dataset)):
    # Skip examples without embeddings
    if example["embedding"] is None:
        continue
    
    vectors_to_upsert.append({
        "id": f"code_{i}",
        "values": example["embedding"],
        "metadata": {
            "original_index": i ## use the original_index to fetch the actual doc chunk at inference time  
        }
    })
    
    # Upsert in batches
    if len(vectors_to_upsert) >= batch_size or i == len(embedded_dataset) - 1:
        try:
            index.upsert(vectors=vectors_to_upsert)
            print(f"Upserted batch ending at index {i}")
            vectors_to_upsert = []  # Reset after upsert
        except Exception as e:
            print(f"Error upserting batch to Pinecone: {e}")

print("Embedding and indexing process complete!")


