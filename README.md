# Submission for the Webpage Classification Task at [IRIXYS'23](https://irixys.uni-passau.de/workshops-summer-schools/)

This is a repository containing submissions to the [webpage classification task at IRIXYS'23](https://github.com/OpenWebSearch/wows-code/tree/main/irixys23/webpage-classification). All submissions will be made to [TIRA.io](https://www.tira.io/task-overview/webpage-classification/) where you can submit your working software to improve reproducibility (you can also upload run files in case of problems with a software submission).

## Submissions to TIRA

TIRA expects software to be submitted as Docker image. For the hackathon, we have prepared a Github Action that automatically builds the Docker image and submits it to TIRA. Please ensure that all code and models are inside the repository and start the Github action, from there, [Maik](https://www.tira.io/u/maik_froebe/summary) will take care of the rest (e.g., fixing build of the image if required). If you have any questions, please do not hesitate to [write Maik a message](https://www.tira.io/u/maik_froebe/summary). To simplify software submissions for the hackathon as much as possible, we recommend that you work either in [Github Codespaces](https://codespaces.new/tira-io/ir-lab-jena-leipzig-wise-2023-irixys23-team-02/tree/main) or using [dev containers with Docker](https://code.visualstudio.com/docs/devcontainers/containers). Github Codespaces are an easy option to start in a few minutes (free tier of 130 compute hours per month).

## Getting the Data

Get the training/validation data via:

```
wget 'https://zenodo.org/records/10118828/files/Hackathon_data.zip?download=1' -O hackathon-data.zip
```

## Existing Starters

To simplify the start, this repository contains the official baselines so that you can directly continue with them:

- [trivial-baseline](trivial-baseline) contains a trivial baseline that always predicts the same class
- [sklearn-baseline](sklearn-baseline) contains a scikit-learn baseline
- [snorkel-baseline](snorkel-baseline) contains a baseline with with snorkel in Python
- [snorkel-jupyter-baseline](snorkel-jupyter-baseline) contains a baseline with with snorkel with Jupyter notebooks

## Developing in Github Codespaces

- [Open this repository in Github Codespaces](https://codespaces.new/tira-io/ir-lab-jena-leipzig-wise-2023-irixys23-team-02/tree/main)
- Please do not forget to commit often


## Developing in Dev Containers

A dev container (please find a suitable installation instruction [here](https://code.visualstudio.com/docs/devcontainers/containers)) allows you to directly work in the prepared Docker container so that you do not have to install the dependencies (which can sometimes be a bit tricky).

To develop with dev containers, please:

- Install [VS Code](https://code.visualstudio.com/download) and [Docker](https://docs.docker.com/engine/install/) on your machine
- Clone this repository: `git clone ...`
- Open the repository with VS Code (it should ask you to open the repository in a dev container)

