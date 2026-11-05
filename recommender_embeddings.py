"""Semantic Recommender Implementation (Activity 3, Intruction 2, Option B: Embeddings).

This module only implements the recommender functions for the execution interface.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CORPUS = Path(__file__).resolve().parent / "corpus.csv"
TOP_K = 5
EXIT_COMMANDS = {"salir", "exit", "q"}


def load_corpus(csv_path: str | Path = DEFAULT_CORPUS) -> list[dict[str, str]]:
    """Load and validate corpus data without modifying"""
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"Corpus not found: {path}")

    documents: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"id", "title", "text"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Missing mandatory columns on corpus.csv: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            doc_id = (row.get("id") or "").strip().upper()
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            category = (row.get("category") or "").strip()

            if not doc_id or not title or not text:
                raise ValueError(
                    f"Row {row_number}: id, title and text can not be empty."
                )
            if doc_id in seen_ids:
                raise ValueError(f"Duplicated ID in row {row_number}: {doc_id}")

            seen_ids.add(doc_id)
            documents.append(
                {"id": doc_id, "title": title, "text": text, "category": category}
            )

    if len(documents) < 2:
        raise ValueError("Corpus must contain at least two documents.")

    return documents


class EmbeddingRecommender:
    """Generate embeddings and search similar documents"""

    def __init__(
        self,
        corpus_path: str | Path = DEFAULT_CORPUS,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.documents = load_corpus(corpus_path)
        self.id_to_index = {
            document["id"]: index for index, document in enumerate(self.documents)
        }
        self.model = SentenceTransformer(model_name)

        document_texts = [
            f'{document["title"]}. {document["text"]}'
            for document in self.documents
        ]
        self.embeddings = self.model.encode(
            document_texts, convert_to_tensor=True, show_progress_bar=False
        )

    def recommend_by_id(
        self, document_id: str, top_k: int = TOP_K
    ) -> list[dict[str, str | float]]:
        """Top-K per document seed, excluding the used document"""
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Must provide a valid document ID.")

        normalized_id = document_id.strip().upper()
        if normalized_id not in self.id_to_index:
            raise ValueError(f"No document with the ID: {normalized_id}")

        seed_index = self.id_to_index[normalized_id]
        scores = util.cos_sim(self.embeddings[seed_index], self.embeddings)[0]
        return self._top_results(scores, top_k=top_k, exclude_index=seed_index)

    def recommend_by_text(
        self, query: str, top_k: int = TOP_K
    ) -> list[dict[str, str | float]]:
        """Top-K per text query with the same model."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Text query can not be empty.")

        query_embedding = self.model.encode(
            query.strip(), convert_to_tensor=True, show_progress_bar=False
        )
        scores = util.cos_sim(query_embedding, self.embeddings)[0]
        return self._top_results(scores, top_k=top_k)

    def _top_results(
        self, scores, top_k: int, exclude_index: int | None = None
    ) -> list[dict[str, str | float]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive number.")

        ranked = sorted(
            (
                (index, float(score))
                for index, score in enumerate(scores)
                if index != exclude_index
            ),
            key=lambda item: (-item[1], self.documents[item[0]]["id"]),
        )

        results: list[dict[str, str | float]] = []
        for index, score in ranked[:top_k]:
            document = self.documents[index]
            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "category": document["category"],
                    "similarity": round(score, 4),
                }
            )
        return results


"""Semantic recommender (Activity 3, point 2, option B: Embeddings).

Includes the recommendation engine and the execution interface for point 3,
option 2: the user pastes a short text and the system recommends the Top-5.
"""

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CORPUS = Path(__file__).resolve().parent / "corpus.csv"
TOP_K = 5
EXIT_COMMANDS = {"salir", "exit", "q"}


def load_corpus(csv_path: str | Path = DEFAULT_CORPUS) -> list[dict[str, str]]:
    """Loads and validates the corpus documents without modifying them."""
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"Corpus not found: {path}")

    documents: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"id", "title", "text"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Missing required columns in corpus.csv: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            doc_id = (row.get("id") or "").strip().upper()
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            category = (row.get("category") or "").strip()

            if not doc_id or not title or not text:
                raise ValueError(
                    f"Row {row_number}: id, title and text cannot be empty."
                )
            if doc_id in seen_ids:
                raise ValueError(f"Duplicate ID in row {row_number}: {doc_id}")

            seen_ids.add(doc_id)
            documents.append(
                {"id": doc_id, "title": title, "text": text, "category": category}
            )

    if len(documents) < 2:
        raise ValueError("The corpus must contain at least two documents.")

    return documents


class EmbeddingRecommender:
    """Generates embeddings once and searches for the most similar documents."""

    def __init__(
        self,
        corpus_path: str | Path = DEFAULT_CORPUS,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.documents = load_corpus(corpus_path)
        self.id_to_index = {
            document["id"]: index for index, document in enumerate(self.documents)
        }
        self.model = SentenceTransformer(model_name)

        document_texts = [
            f'{document["title"]}. {document["text"]}'
            for document in self.documents
        ]
        self.embeddings = self.model.encode(
            document_texts, convert_to_tensor=True, show_progress_bar=False
        )

    def recommend_by_id(
        self, document_id: str, top_k: int = TOP_K
    ) -> list[dict[str, str | float]]:
        """Top-K for a seed document, excluding the document itself."""
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("You must provide a valid document ID.")

        normalized_id = document_id.strip().upper()
        if normalized_id not in self.id_to_index:
            raise ValueError(f"Document does not exist: {normalized_id}")

        seed_index = self.id_to_index[normalized_id]
        scores = util.cos_sim(self.embeddings[seed_index], self.embeddings)[0]
        return self._top_results(scores, top_k=top_k, exclude_index=seed_index)

    def recommend_by_text(
        self, query: str, top_k: int = TOP_K
    ) -> list[dict[str, str | float]]:
        """Top-K for free text, using the same pretrained model."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("The text query cannot be empty.")

        query_embedding = self.model.encode(
            query.strip(), convert_to_tensor=True, show_progress_bar=False
        )
        scores = util.cos_sim(query_embedding, self.embeddings)[0]
        return self._top_results(scores, top_k=top_k)

    def _top_results(
        self, scores, top_k: int, exclude_index: int | None = None
    ) -> list[dict[str, str | float]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        ranked = sorted(
            (
                (index, float(score))
                for index, score in enumerate(scores)
                if index != exclude_index
            ),
            key=lambda item: (-item[1], self.documents[item[0]]["id"]),
        )

        results: list[dict[str, str | float]] = []
        for index, score in ranked[:top_k]:
            document = self.documents[index]
            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "category": document["category"],
                    "similarity": round(score, 4),
                }
            )
        return results


# ---------------------------------------------------------------------------
# Execution interface (point 3, option 2): free text -> Top-5
# ---------------------------------------------------------------------------

def read_pasted_text() -> str | None:
    """Reads pasted text (one or more lines) up to an empty line.

    Returns None if the user types an exit command or closes the input.
    """
    print("\nPaste your text (leave an empty line to search, or type 'exit'):")
    lines: list[str] = []
    while True:
        try:
            line = input("> " if not lines else "  ")
        except EOFError:
            break

        if not lines and line.strip().lower() in EXIT_COMMANDS:
            return None
        if not line.strip():
            if lines:
                break
            continue  # Ignore empty lines before the user starts typing.
        lines.append(line.strip())

    if not lines:
        return None
    return " ".join(lines)


def print_results(query: str, results: list[dict[str, str | float]]) -> None:
    """Displays the Top-K results as a readable table."""
    preview = query if len(query) <= 80 else query[:77] + "..."
    print(f'\nTop-{len(results)} recommendations for: "{preview}"\n')
    print(f'{"#":<3} {"ID":<8} {"Similarity":<11} {"Category":<18} Title')
    print("-" * 80)
    for position, result in enumerate(results, start=1):
        print(
            f'{position:<3} {result["id"]:<8} {result["similarity"]:<11.4f} '
            f'{(result["category"] or "-"):<18} {result["title"]}'
        )


def main() -> None:
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS

    print("Loading model and generating corpus embeddings...")
    try:
        recommender = EmbeddingRecommender(corpus_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error loading the corpus: {error}")
        sys.exit(1)

    print(f"Ready. Documents loaded: {len(recommender.documents)}")
    print("=== Semantic recommender: free-text search ===")

    while True:
        query = read_pasted_text()
        if query is None:
            print("\nGoodbye.")
            break

        try:
            results = recommender.recommend_by_text(query, top_k=TOP_K)
        except ValueError as error:
            print(f"Error: {error}")
            continue

        print_results(query, results)


if __name__ == "__main__":
    main()