"""Evaluate a small text-classification prototype on synthetic skill answers."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sample_answers.csv"
REPORT_DIR = ROOT / "reports"
LABELS = ["incorrect", "partial", "correct"]
RANDOM_STATE = 42


def read_examples() -> list[dict[str, str]]:
    with DATA_PATH.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    required = {"question_id", "topic", "prompt", "answer", "label", "rubric_terms", "reason"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("The sample dataset is empty or missing required columns.")
    invalid = sorted({row["label"] for row in rows} - set(LABELS))
    if invalid:
        raise ValueError(f"Unexpected labels: {invalid}")
    return rows


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )


def rubric_prediction(answer: str, rubric_terms: str) -> str:
    answer_words = set(re.findall(r"[a-z0-9]+", answer.casefold()))
    terms = {term.strip().casefold() for term in rubric_terms.split("|") if term.strip()}
    if not terms:
        return "incorrect"
    coverage = len(answer_words & terms) / len(terms)
    if coverage >= 0.75:
        return "correct"
    if coverage >= 0.40:
        return "partial"
    return "incorrect"


def metric_summary(actual: list[str], predicted: list[str]) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=LABELS, zero_division=0
    )
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0),
        "per_label": {
            label: {"precision": float(p), "recall": float(r), "f1": float(score), "support": int(n)}
            for label, p, r, score, n in zip(LABELS, precision, recall, f1, support)
        },
        "matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
    }


def write_disagreements(rows: list[dict[str, str]]) -> None:
    path = REPORT_DIR / "disagreements.csv"
    fields = ["question_id", "topic", "prompt", "answer", "label", "model_prediction", "model_confidence", "rubric_prediction"]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation() -> None:
    examples = read_examples()
    indices = list(range(len(examples)))
    groups = [row["question_id"] for row in examples]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=RANDOM_STATE)
    train_indices, test_indices = next(splitter.split(indices, groups=groups))
    train = [examples[i] for i in train_indices]
    test = [examples[i] for i in test_indices]

    model = make_model()
    model.fit([row["prompt"] + "\n" + row["answer"] for row in train], [row["label"] for row in train])
    test_text = [row["prompt"] + "\n" + row["answer"] for row in test]
    actual = [row["label"] for row in test]
    predicted = model.predict(test_text).tolist()
    probabilities = model.predict_proba(test_text)
    model_metrics = metric_summary(actual, predicted)

    baseline_predictions = [rubric_prediction(row["answer"], row["rubric_terms"]) for row in test]
    baseline_metrics = metric_summary(actual, baseline_predictions)

    disagreements: list[dict[str, str]] = []
    for row, truth, guess, probs, baseline_guess in zip(test, actual, predicted, probabilities, baseline_predictions):
        if truth != guess or truth != baseline_guess:
            confidence = max(float(value) for value in probs)
            disagreements.append(
                {
                    "question_id": row["question_id"],
                    "topic": row["topic"],
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                    "label": truth,
                    "model_prediction": guess,
                    "model_confidence": f"{confidence:.3f}",
                    "rubric_prediction": baseline_guess,
                }
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_disagreements(disagreements)
    test_groups = sorted({row["question_id"] for row in test})
    report = [
        "# SkillCheck Eval Lab - Evaluation Report",
        "",
        "> Demonstration only. The examples are synthetic and the metrics do not establish real-world grading or hiring accuracy.",
        "",
        "## Setup",
        "",
        f"- Examples: {len(examples)} synthetic responses across {len(set(groups))} questions.",
        f"- Training rows: {len(train)}; held-out rows: {len(test)}.",
        f"- Held-out question IDs: {', '.join(test_groups)}.",
        "- Split method: group holdout by question ID (random seed 42).",
        "- Model: TF-IDF word unigrams/bigrams + logistic regression.",
        "- Baseline: keyword coverage against the sample rubric terms.",
        "",
        "## Held-out metrics",
        "",
        "| Approach | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
        f"| TF-IDF + logistic regression | {model_metrics['accuracy']:.3f} | {model_metrics['macro_f1']:.3f} |",
        f"| Keyword rubric baseline | {baseline_metrics['accuracy']:.3f} | {baseline_metrics['macro_f1']:.3f} |",
        "",
        "Macro-F1 weights each label equally. It is shown alongside accuracy so a single overall score does not hide weaker labels.",
        "",
        f"On this single synthetic-data split, the keyword baseline scored higher macro-F1 ({baseline_metrics['macro_f1']:.3f}) than the text model ({model_metrics['macro_f1']:.3f}). The gap is not evidence of general performance; it is a failure case to investigate with more reviewed examples.",
        "",
        "## Per-label metrics",
        "",
        "| Label | Model precision | Model recall | Model F1 | Support | Rubric precision | Rubric recall | Rubric F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        model_row = model_metrics["per_label"][label]
        baseline_row = baseline_metrics["per_label"][label]
        report.append(
            f"| {label} | {model_row['precision']:.3f} | {model_row['recall']:.3f} | {model_row['f1']:.3f} | {model_row['support']} | "
            f"{baseline_row['precision']:.3f} | {baseline_row['recall']:.3f} | {baseline_row['f1']:.3f} |"
        )
    report.extend(
        [
            "",
            "## Model confusion matrix",
            "",
            "Rows are reference labels; columns are predictions. Label order: incorrect, partial, correct.",
            "",
            "```text",
            "                 predicted",
            "actual       incorrect  partial  correct",
        ]
    )
    for label, values in zip(LABELS, model_metrics["matrix"]):
        report.append(f"{label:<13} {values[0]:>9} {values[1]:>9} {values[2]:>8}")
    report.extend(
        [
            "```",
            "",
            f"The disagreement file contains {len(disagreements)} held-out examples where at least one approach differs from the reference label.",
            "",
            "## Limitations and next steps",
            "",
            "- The dataset is small, synthetic, and labeled using a simple demonstration guide; labels were not independently adjudicated.",
            "- A text classifier can reward wording patterns rather than technical correctness. It may miss valid alternative explanations and mishandle negation.",
            "- The keyword baseline is intentionally simple and cannot understand context or synonyms.",
            "- Do not use either score for screening, ranking, or rejecting real candidates. A production evaluation would need representative consented data, independent review, subgroup/error analysis, privacy controls, and human oversight.",
            "- A useful next step is to expand the benchmark with reviewed examples and analyze failures by skill area before changing the model.",
            "",
        ]
    )
    report_path = REPORT_DIR / "evaluation_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {report_path.relative_to(ROOT)}")
    print(f"Model macro-F1: {model_metrics['macro_f1']:.3f}")
    print(f"Rubric baseline macro-F1: {baseline_metrics['macro_f1']:.3f}")
    print(f"Held-out disagreements: {len(disagreements)}")


def predict_one(prompt: str, answer: str) -> None:
    examples = read_examples()
    model = make_model()
    texts = [row["prompt"] + "\n" + row["answer"] for row in examples]
    labels = [row["label"] for row in examples]
    model.fit(texts, labels)
    input_text = prompt + "\n" + answer
    probabilities = model.predict_proba([input_text])[0]
    best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    prediction = model.classes_[best_index]
    confidence = float(probabilities[best_index])
    print(f"Prototype label: {prediction}")
    print(f"Model confidence: {confidence:.3f}")
    if confidence < 0.55:
        print("Review needed: the prototype is uncertain. Do not use this result as a decision.")
    else:
        print("Review needed: this synthetic-data prototype is not validated for real answers.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate or try the SkillCheck answer-labeling prototype.")
    parser.add_argument("--prompt", default="", help="Challenge prompt for a one-off prediction.")
    parser.add_argument("--answer", default="", help="Short answer to label; without this, run the evaluation report.")
    args = parser.parse_args()
    if args.answer:
        predict_one(args.prompt, args.answer)
    else:
        run_evaluation()


if __name__ == "__main__":
    main()

