# Scikit-Learn Baseline for The Webpage Classification Task

This is a scikit-learn baseline for the webpage classification task.
Find more details (including a trained classifier) at the [overview repository](https://github.com/OpenWebSearch/wows-code/tree/main/irixys23/webpage-classification/baselines/sklearn-baseline).

## Submission

Submit this via the Github Action (navigate to "Actions" -> "Upload Docker Software to TIRA") via the following fields (required that you placed the sk_model.pkl in this directory, e.g., via `wget 'https://github.com/OpenWebSearch/wows-code/raw/main/irixys23/webpage-classification/baselines/sklearn-baseline/sk_model.pkl'`):

- path: `sklearn-baseline`
- command: `python3 /app/sklearn_baseline.py -i $inputDataset/inputs.jsonl -o $outputDir -m /app/sk_model.pkl`
- Dockerfile: `Dockerfile`

