
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# 1. File locations
BASE_DIR = Path(__file__).resolve().parent
CORPUS_FILE = BASE_DIR.parent / "activity3" / "corpus.csv"
VALIDATION_FILE = BASE_DIR / "validation.csv"
FIGURES_DIR = BASE_DIR / "figures"
EVIDENCE_DIR = BASE_DIR / "evidence"


# 2. Read and check the validation data
def load_validation():
    with VALIDATION_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {
            "document_id",
            "ground_truth_relevant",
            "retrieved_by_system",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing columns in validation.csv")

        rows = list(reader)

    if len(rows) != 25:
        raise ValueError("Expected 25 validation documents.")

    document_ids = [row["document_id"] for row in rows]

    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Duplicate document IDs found.")

    with CORPUS_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        corpus_ids = {
            row["id"] for row in csv.DictReader(file)
        }

    if set(document_ids) != corpus_ids:
        raise ValueError("Validation IDs do not match the corpus.")

    for row in rows:
        for column in ("ground_truth_relevant", "retrieved_by_system"):
            if row[column] not in ("0", "1"):
                raise ValueError(
                    f"Invalid value in {row['document_id']}: {column}"
                )

    actual = [int(row["ground_truth_relevant"]) for row in rows]
    predicted = [int(row["retrieved_by_system"]) for row in rows]

    return rows, actual, predicted


# 3. Calculate evaluation metrics
def calculate_metrics(actual, predicted):
    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])

    return precision, recall, f1, matrix


# 4. Generate the confusion matrix image
def plot_confusion_matrix(matrix):
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No", "Yes"],
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title("Confusion Matrix - Document Recommendations")
    ax.set_xlabel("Retrieved by the system")
    ax.set_ylabel("Ground truth")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


# 5. Generate a simple chart
def plot_metrics(precision, recall, f1):
    names = ["Precision", "Recall", "F1-score"]
    values = [precision, recall, f1]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Recommendation System Performance")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2%}",
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=180)
    plt.close(fig)


# 6. Run the evaluation and save the results
def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    EVIDENCE_DIR.mkdir(exist_ok=True)

    rows, actual, predicted = load_validation()
    precision, recall, f1, matrix = calculate_metrics(actual, predicted)

    tn, fp, fn, tp = matrix.ravel()

    lines = [
        "ACTIVITY 4 - RECOMMENDATION EVALUATION",
        "=" * 53,
        "",
        "Document   Relevant   Retrieved",
        "-" * 36,
    ]

    for row in rows:
        lines.append(
            f"{row['document_id']:<10} "
            f"{row['ground_truth_relevant']:<10} "
            f"{row['retrieved_by_system']}"
        )

    lines.extend([
        "",
        "CONFUSION MATRIX RESULTS",
        f"True Positives:  {tp}",
        f"False Positives: {fp}",
        f"False Negatives: {fn}",
        f"True Negatives:  {tn}",
        "",
        "EVALUATION METRICS",
        f"Precision: {precision:.2%}",
        f"Recall:    {recall:.2%}",
        f"F1-score:  {f1:.2%}",
    ])

    output = "\n".join(lines)
    print(output)

    (EVIDENCE_DIR / "evaluation_output.txt").write_text(
        output + "\n", encoding="utf-8"
    )

    plot_confusion_matrix(matrix)
    plot_metrics(precision, recall, f1)

    print("\nEvaluation finished. Charts and results were saved.")


if __name__ == "__main__":
    main()
