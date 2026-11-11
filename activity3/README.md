# Activity 3 — Document Vectorized Recommender

## Objective

Identify and rank the most relevant documents in a corpus using semantic embeddings and cosine similarity. This is the original recommender used by [Activity 4](../activity4/README.md).

## Implementation

- **Approach:** Sentence Transformer embeddings (Path B)
- **Pretrained model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Input:** Free-text queries (main interface) or a document ID (Python example)
- **Output:** Top-5 ranked documents with similarity scores
- **Corpus:** `corpus.csv`, containing 25 documents from the Policy/Process and Support/Incidents categories

The model computes document embeddings and uses cosine similarity to recommend the most relevant documents. When searching by document ID, the seed document is excluded.

## Requirements and execution

Install Python and then, from the repository root:

```bash
python -m pip install -r activity3/requirements.txt
python activity3/recommender_embeddings.py
```

Enter a query at the prompt and then press Enter on a blank line:

```text
Paste your text (leave an empty line to search, or type 'exit'):
> YOUR_QUERY_HERE
```

Run the example document-ID search with:

```bash
python activity3/example.py
```

The corpus path is derived from the script's folder, so both scripts continue to locate `activity3/corpus.csv` even when executed from the repository root.

## Activity 3 deliverables

- [`recommender_embeddings.py`](recommender_embeddings.py): Main recommender
- [`example.py`](example.py): Document-ID search example
- [`corpus.csv`](corpus.csv): Document corpus
- [`requirements.txt`](requirements.txt): Python dependency
- [`evidencias/`](evidencias/README.md): Screenshots and comments from the three original test scenarios
- [`Act3_KM.pdf`](Act3_KM.pdf): Original report
- [`Activity 3 Instruccions.txt`](Activity%203%20Instruccions.txt): Original assignment instructions and rubric

## Continuity

Activity 4 reuses these source files unchanged. Its relevance labels and report are kept separately in [`../activity4/`](../activity4/README.md).
