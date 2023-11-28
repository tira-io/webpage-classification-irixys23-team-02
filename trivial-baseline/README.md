# Trivial Baseline The Webpage Classification Task

This is a trivial baseline for the webpage classification task that always predicts the passed label (e.g., `Benign` for --prediction Benign).

## Submission

Submit this via the Github Action (navigate to "Actions" -> "Upload Docker Software to TIRA") via the following fields:

- path: `trivial-baseline`
- command: `python3 /baseline.py -i $inputDataset/inputs.jsonl -o $outputDir/predictions.jsonl`
- Dockerfile: `Dockerfile`

