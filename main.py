import stoper
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

stoper = stoper.stoper()
stoper.start()

# Same embedding model that used to generate the embeddings in the CSV file.
EMBEDDING_MODEL_TYPE = "text-embedding-3-large" 
FILMS_CSV_URL = "data/fp_enrichment_14films_09-2024_with_academic_categories_enriched.csv"
dataframe = pd.read_csv(FILMS_CSV_URL)

stoper.lap('after CSV load')

embeddings = OpenAIEmbeddings(model = EMBEDDING_MODEL_TYPE)
text_embeddings = dataframe['academic_value_embedding'].apply(eval).tolist()

stoper.lap('after convert embeddings column')

texts = dataframe['academic_value']
metadata = [{'film_id': row[1]['film_id'], 'film_name': row[1]['film_name']} for row in dataframe.iterrows()]

stoper.lap('before vector store creation')

# Initialize FAISS vector store with embeddings and documents
vector_store = FAISS.from_embeddings(text_embeddings=list(zip(texts, text_embeddings)),
                                      metadatas=metadata, embedding=embeddings)

stoper.lap('after vector store creation')

# Example query 
query_embedding = embeddings.embed_query("A film about social issues in the USA")

stoper.lap('after query embedding')

results = vector_store.similarity_search_by_vector(query_embedding, k=8)

stoper.lap('got results')

# Print the results
for result in results:
    print(f"\n\nFilm ID:{result.metadata['film_id']}, Film name: {result.metadata['film_name']}, Academic value: {result.page_content}")

# Print timing information
print("\n" + stoper.get_data_str())