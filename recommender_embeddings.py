"""Recomendador semántico (Actividad 3, punto 2, opción B: Embeddings).

Este módulo implementa solamente el motor de recomendación.
No incluye interfaz interactiva, evidencias ni entregables de otros puntos.
"""

from __future__ import annotations

import csv
from pathlib import Path

from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CORPUS = Path(__file__).resolve().parent / "corpus.csv"


def load_corpus(csv_path: str | Path = DEFAULT_CORPUS) -> list[dict[str, str]]:
    """Carga y valida los documentos del corpus sin modificarlos."""
    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el corpus: {path}")

    documents: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"id", "title", "text"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Faltan columnas obligatorias en corpus.csv: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            doc_id = (row.get("id") or "").strip().upper()
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            category = (row.get("category") or "").strip()

            if not doc_id or not title or not text:
                raise ValueError(
                    f"Fila {row_number}: id, title y text no pueden estar vacíos."
                )
            if doc_id in seen_ids:
                raise ValueError(f"ID duplicado en fila {row_number}: {doc_id}")

            seen_ids.add(doc_id)
            documents.append(
                {"id": doc_id, "title": title, "text": text, "category": category}
            )

    if len(documents) < 2:
        raise ValueError("El corpus debe contener al menos dos documentos.")

    return documents


class EmbeddingRecommender:
    """Genera embeddings una vez y busca los documentos más similares."""

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
        self, document_id: str, top_k: int = 5
    ) -> list[dict[str, str | float]]:
        """Top-K por documento semilla, excluyendo el propio documento."""
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Debes proporcionar un ID de documento válido.")

        normalized_id = document_id.strip().upper()
        if normalized_id not in self.id_to_index:
            raise ValueError(f"No existe el documento: {normalized_id}")

        seed_index = self.id_to_index[normalized_id]
        scores = util.cos_sim(self.embeddings[seed_index], self.embeddings)[0]
        return self._top_results(scores, top_k=top_k, exclude_index=seed_index)

    def recommend_by_text(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, str | float]]:
        """Top-K por texto libre, mediante el mismo modelo preentrenado."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("La consulta de texto no puede estar vacía.")

        query_embedding = self.model.encode(query.strip(), convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.embeddings)[0]
        return self._top_results(scores, top_k=top_k)

    def _top_results(
        self, scores, top_k: int, exclude_index: int | None = None
    ) -> list[dict[str, str | float]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k debe ser un entero positivo.")

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
