# Document Vectorized Recommender

## Objective

The Document Vectorized Recommender is designed to identify and rank the most relevant documents from a corpus based on the similarity of their vectorized representations. This system is particularly useful for retrieving documents that closely match a given input, whether it is a full document or a free text query.

## Dependencies

Check the `requirements.txt` file for the necessary dependencies.

## Execution steps

1. Ensure all dependencies are installed by running:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the recommender script:
   ```bash
   python recommender_embeddings.py
   ```

## Usage example

Enter your query when prompted by the script:

```text
Paste your text (leave an empty line to search, or type 'exit'):
> YOUR_QUERY_HERE
```

Results will be displayed based on the similarity of your query to the documents in the corpus.
