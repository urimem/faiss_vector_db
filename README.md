# FAISS - Embeddings Analysis Project

This project implements semantic search functionality for film analysis using OpenAI embeddings and FAISS vector store.

## Overview

The system allows for semantic searching through a film database using embeddings generated from academic value descriptions. It utilizes OpenAI's text-embedding-3-large model for generating embeddings and FAISS for efficient similarity search.

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Project Structure

```
.
├── data/
│   └── fp_enrichment_14films_09-2024_with_academic_categories_enriched.csv
├── notebooks/
│   └── film_embeddings.ipynb
├── main.py
├── stoper.py
├── requirements.txt
└── README.md
```

## Features

- Load and process film data from CSV
- Utilize pre-computed embeddings for efficient search
- Perform semantic similarity search using FAISS
- Performance timing using custom Stoper class
- Jupyter notebook for interactive analysis

## Usage

### Command Line Interface

Run the main script:
```bash
python main.py
```

### Jupyter Notebook

Start Jupyter notebook:
```bash
jupyter notebook
```
Navigate to `notebooks/film_embeddings.ipynb` for interactive analysis.

## Data Format

The CSV file should contain the following columns:
- `film_id`: Unique identifier for each film
- `film_name`: Name of the film
- `academic_value`: Text description of the film's academic value
- `academic_value_embedding`: Pre-computed embeddings for the academic value text

## Performance Monitoring

The project includes a custom `Stoper` class for performance monitoring:
- Tracks execution time between different stages
- Provides detailed timing information
- Helps identify potential bottlenecks

## Example Usage

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Create vector store
vector_store = FAISS.from_embeddings(
    text_embeddings=list(zip(texts, text_embeddings)),
    metadatas=metadata,
    embedding=embeddings
)

# Perform similarity search
results = vector_store.similarity_search_by_vector(query_embedding, k=8)
```

## Dependencies

```
pandas
langchain
langchain-openai
faiss-cpu
jupyter
notebook
openai
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
