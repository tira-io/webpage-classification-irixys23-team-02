#!/usr/bin/env python3
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os
import joblib
import numpy as np
from baseline_train import load_data, preprocess_text_dataset, preprocess_url_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Webpage classification with sklearn pipeline')
    parser.add_argument(
        "-i", "--input_data", help="Path to the jsonl file for which predictions should be made.", required=True)
    parser.add_argument(
        "-m", "--model", help="The sklearn SGDClassifier model to use for the predictions.", required=True)
    parser.add_argument(
        "-o", "--output", help="Path to the directory to write the results to.", required=True)
    return parser.parse_args()


def load_model(model_file):
    # Load the trained model
    models = joblib.load(model_file)
    return models


def main(input_file, output_dir, model_file):
    # Load models
    models = load_model(model_file)
    pipeline_text = models['text']
    pipeline_url = models['url']
    corpus = models['url_corpus']
    model_final = models['final']

    # Load datasets
    test_data = load_data(input_file)

    # Text based model
    test_features1 = preprocess_text_dataset(test_data)
    test_predictions1 = pipeline_text.predict_proba(test_features1)

    # Url based model
    test_features2, corpus = preprocess_url_dataset(
        test_data, train=False, corpus=corpus)
    test_predictions2 = pipeline_text.predict_proba(test_features2)

    # Final model
    test_features_final = np.concatenate(
        [test_predictions1, test_predictions2], axis=1)
    model_final = SGDClassifier()
    test_predictions_final = model_final.predict(test_features_final)

    # Save the predictions
    test_data['prediction'] = test_predictions_final
    output_path = os.path.join(output_dir, 'predictions.jsonl')
    test_data[['uid', 'prediction']].to_json(
        output_path, orient='records', lines=True)


if __name__ == "__main__":
    args = parse_args()
    main(args.input_data, args.output, args.model)
