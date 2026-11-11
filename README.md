# Document Vectorized Recommender

This repository contains two connected academic activities about semantic document recommendation and its evaluation.

## Activities

- **[Activity 3 — Document Recommender](activity3/README.md):** The original Python recommendation engine, document corpus, installation instructions, evidence, and final report.
- **[Activity 4 — Recommendation Quality Evaluation](activity4/README.md):** An ongoing evaluation of the same recommender using manually classified relevance labels, Precision, Recall, F1-score, and visual evidence.

The Activity 4 evaluation uses the original Activity 3 model and dataset; it does not replace or modify them.

## Run the recommender (Activity 3)

From the repository root:

```bash
python -m pip install -r activity3/requirements.txt
python activity3/recommender_embeddings.py
```

To run the document-ID example:

```bash
python activity3/example.py
```

The recommender looks for `activity3/corpus.csv` automatically, regardless of the current working directory.

## Project organization

```text
documentVectorized/
├── activity3/                 # Original recommender and Activity 3 deliverables
│   ├── recommender_embeddings.py
│   ├── example.py
│   ├── corpus.csv
│   ├── requirements.txt
│   ├── evidencias/
│   ├── Act3_KM.pdf
│   └── README.md
├── activity4/                 # Evaluation, validation table, and report in progress
│   ├── report/
│   ├── validation.csv
│   ├── figures/
│   ├── evidence/
│   ├── infographic/
│   └── README.md
├── .gitignore                 # Shared repository-level exclusions
└── README.md                  # This project index
```

For detailed requirements, evidence, and current completion status, use each activity's README.
