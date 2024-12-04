# Comparing Loading the FAISS object from CSV with an already existing embeddings
# and from a stored FAISS index.
# Results for the full films CSV (approx. 750 films with embeddings 55MB) is:
# CSV Load Time: 3.816 seconds
# FAISS index Load Time: 0.004 seconds

import time
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

FILMS_CSV_FILE = "data/fp_enrichment_14films_09-2024_with_academic_categories_enriched.csv"
FAISS_INDEX_PATH = "data/faiss_index"
EMBEDDING_MODEL_TYPE = "text-embedding-3-large"
embeddings = OpenAIEmbeddings(model = EMBEDDING_MODEL_TYPE)

def benchmark_csv_load():
    start_time = time.time()
    df = pd.read_csv(FILMS_CSV_FILE)
    text_embeddings = df['academic_value_embedding'].apply(eval).tolist()
    texts = df['academic_value']
    metadata = [{'film_id': row[1]['film_id'], 'film_name': row[1]['film_name']} 
                for row in df.iterrows()]
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, text_embeddings)),
        metadatas=metadata,
        embedding=embeddings
    )
    end_time = time.time()
    vector_store.save_local(FAISS_INDEX_PATH)
    return end_time - start_time

def benchmark_faiss_load():
    start_time = time.time()
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization = True)
    end_time = time.time()
    return end_time - start_time

# Run benchmarks
csv_time = benchmark_csv_load()
faiss_time = benchmark_faiss_load()

print(f"CSV Load Time: {csv_time:.3f} seconds")
print(f"FAISS Load Time: {faiss_time:.3f} seconds")