# SkillCheck Eval Lab

A small, offline Python prototype for evaluating short answers to beginner programming and machine-learning skill challenges. It is tailored to the kind of work in the AI/ML internship listing: build a small evaluation set, compare an AI/ML baseline with a simple rubric, inspect disagreements, and report limitations.

## What it does

- Uses 54 hand-authored demonstration responses across 18 questions. Every response is synthetic; there is no real applicant data.
- Compares a TF-IDF + logistic-regression text classifier with a transparent keyword-coverage rubric.
- Holds out complete questions during evaluation so near-duplicate answers to the same question do not appear in both train and test sets.
- Reports accuracy, macro-F1, per-label precision/recall, a confusion matrix, and examples where the model disagrees with the labels.
- Lets you try the trained prototype on a prompt and answer from the command line.

The current demonstration run shows the keyword baseline ahead of the text classifier on its held-out synthetic examples. The project treats that as a result to investigate, not a claim that either approach is ready for real grading.

## Run it

Use Python 3.10 or newer.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_evaluation.py
python run_evaluation.py --prompt "What is data leakage?" --answer "The model accidentally gets information from the held-out data during training."
```

The evaluation writes `reports/evaluation_report.md` and `reports/disagreements.csv`.

## Data and limits

The examples and labels are synthetic and authored for this demonstration. The score is not evidence of real-world hiring accuracy. The dataset is small, the labels are not independently adjudicated, and TF-IDF cannot reliably judge technical meaning, equivalent code, or context by itself. Do not use this prototype to screen or reject candidates. A real evaluation should use consented, representative examples, a written rubric, independent label review, privacy safeguards, subgroup analysis, and human oversight.

This prototype makes no external model/API calls and needs no API key. It was created with AI coding assistance; review the code, rerun the project yourself, and be ready to explain the design and its limitations before presenting it.

## References

- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [scikit-learn GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

