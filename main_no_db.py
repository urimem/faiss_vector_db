import stoper
import pandas as pd
from langchain_openai import OpenAIEmbeddings
import numpy as np

stoper = stoper.stoper()
stoper.start()

EMBEDDING_MODEL_TYPE = "text-embedding-3-large"
embed_model = OpenAIEmbeddings(model = EMBEDDING_MODEL_TYPE)

df = pd.read_csv("data/fp_enrichment_14films_09-2024_with_academic_categories_enriched.csv")
df.head()

stoper.lap("after CSV load")

# Creating embeddings for user request/prompt
user_request = "A film about social issues in the USA"
user_request_embedding = embed_model.embed_query(user_request)

stoper.lap("after query embedding creation")

# Calculating similarity/distance from user_request to each film
def get_similarity(film_embedding):
    film_embedding_list = eval(film_embedding) if isinstance(film_embedding, str) else film_embedding
    return np.dot(film_embedding_list, user_request_embedding)

# Adding similarity column 
df['similarity'] = df['academic_value_embedding'].apply(get_similarity)

stoper.lap("after similarity calc")

# Sort the records by similarity and take the first X records
df.sort_values('similarity', ascending = False, inplace = True)

stoper.lap("after sorting df")

for i in range(0,8):
    print(str(i) + ". Film name: " + df['film_name'].iloc[i] + "  - Similarity score: " + str(df['similarity'].iloc[i]))

print("\n" + stoper.get_data_str())