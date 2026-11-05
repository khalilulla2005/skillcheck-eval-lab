# SkillCheck Eval Lab - Evaluation Report

> Demonstration only. The examples are synthetic and the metrics do not establish real-world grading or hiring accuracy.

## Setup

- Examples: 54 synthetic responses across 18 questions.
- Training rows: 36; held-out rows: 18.
- Held-out question IDs: baseline, confusion_matrix, drift, hallucination, leakage, py_dict_lookup.
- Split method: group holdout by question ID (random seed 42).
- Model: TF-IDF word unigrams/bigrams + logistic regression.
- Baseline: keyword coverage against the sample rubric terms.

## Held-out metrics

| Approach | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + logistic regression | 0.444 | 0.435 |
| Keyword rubric baseline | 0.556 | 0.530 |

Macro-F1 weights each label equally. It is shown alongside accuracy so a single overall score does not hide weaker labels.

On this single synthetic-data split, the keyword baseline scored higher macro-F1 (0.530) than the text model (0.435). The gap is not evidence of general performance; it is a failure case to investigate with more reviewed examples.

## Per-label metrics

| Label | Model precision | Model recall | Model F1 | Support | Rubric precision | Rubric recall | Rubric F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| incorrect | 0.500 | 0.333 | 0.400 | 6 | 0.444 | 0.667 | 0.533 |
| partial | 0.333 | 0.333 | 0.333 | 6 | 0.333 | 0.167 | 0.222 |
| correct | 0.500 | 0.667 | 0.571 | 6 | 0.833 | 0.833 | 0.833 |

## Model confusion matrix

Rows are reference labels; columns are predictions. Label order: incorrect, partial, correct.

```text
                 predicted
actual       incorrect  partial  correct
incorrect             2         2        2
partial               2         2        2
correct               0         2        4
```

The disagreement file contains 14 held-out examples where at least one approach differs from the reference label.

## Limitations and next steps

- The dataset is small, synthetic, and labeled using a simple demonstration guide; labels were not independently adjudicated.
- A text classifier can reward wording patterns rather than technical correctness. It may miss valid alternative explanations and mishandle negation.
- The keyword baseline is intentionally simple and cannot understand context or synonyms.
- Do not use either score for screening, ranking, or rejecting real candidates. A production evaluation would need representative consented data, independent review, subgroup/error analysis, privacy controls, and human oversight.
- A useful next step is to expand the benchmark with reviewed examples and analyze failures by skill area before changing the model.

