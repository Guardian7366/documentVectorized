# Activity 4 — Evaluation of the Document Recommender

## Status

**Work in progress.** Sections 1 and 2 of the report (model description, test scenario, relevance criteria, and validation table) were supplied in `report/Act4_KM.docx`. The remaining evaluation deliverables are not yet completed.

This activity evaluates the recommender from **Activity 3**. It does not rebuild or change the original model.

## Reused components from Activity 3

- **Engine:** [`../activity3/recommender_embeddings.py`](../activity3/recommender_embeddings.py)
- **Corpus:** [`../activity3/corpus.csv`](../activity3/corpus.csv) (25 documents; `DOC001` through `DOC025`)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Method:** semantic embeddings, cosine similarity, Top-5 results

## Selected evaluation scenario

The text query described in the Word report is a fragment of **DOC014**:

> Several users reported that they could not sign in to the client portal after completing the password reset process. The reset page displayed a success message, but the new password was rejected on the login screen

The information need concerns user authentication, password resets, login failures, and account access. The report states that relevance was judged manually by reviewing all 25 corpus documents, independently of their categories.

## Available files

- [`report/Act4_KM.docx`](report/Act4_KM.docx): existing Word report, with sections 1 and 2 completed.
- [`validation.csv`](validation.csv): labels transcribed from the report's validation table.
  - `document_id`: original corpus document ID.
  - `ground_truth_relevant`: manually judged relevant (1) or not relevant (0).
  - `retrieved_by_system`: marked retrieved (1) or not retrieved (0) in the report.

**Evidence note:** The retrieval flags in `validation.csv` reproduce the results reported in the supplied Word document; they have not been independently rerun or verified in this repository setup commit.

## Reserved folders and upcoming deliverables

- `figures/`: confusion matrix and performance charts (pending).
- `evidence/`: execution screenshots and console output (pending).
- `infographic/`: organizational-value infographic (pending).
- `report/`: continue the existing report with metrics and interpretation (pending).
- `evaluate_recommender.py` or a Jupyter notebook: evaluation with `scikit-learn` (pending; not yet created).

## Working principles

1. Keep the original Activity 3 engine and dataset unchanged.
2. Compare the original 25 validation labels against actual output for the same query.
3. Make evaluation results reproducible from code before reporting Precision, Recall, F1-score, or the confusion matrix.
4. Add the source code and its dependencies when the evaluation is implemented.
5. Do not present pending charts or runtime evidence as completed.

The Word copy in this repository omits bundled font binaries to avoid publishing font files; its text, tables, graphics, and four-page rendering were checked against the supplied document.
