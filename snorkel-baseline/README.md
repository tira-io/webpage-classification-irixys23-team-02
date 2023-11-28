# Snorkel Baseline for The Webpage Classification Task

This is a snorkel baseline for the webpage classification task.
Find more details (including a trained classifier) at the [overview repository](https://github.com/OpenWebSearch/wows-code/tree/main/irixys23/webpage-classification/baselines/snorkel-baseline).

## Submission

Submit this via the Github Action (navigate to "Actions" -> "Upload Docker Software to TIRA") via the following fields (required that you placed the snrkl_model.pkl in this directory, e.g., via `wget 'https://github.com/OpenWebSearch/wows-code/raw/main/irixys23/webpage-classification/baselines/snorkel-baseline/snrkl_model.pkl'`):

- path: `snorkel-baseline`
- command: `python3 /app/snorkel_baseline.py -i $inputDataset/inputs.jsonl -o $outputDir/predictions.jsonl -m /app/snrkl_model.pkl`
- Dockerfile: `Dockerfile`

