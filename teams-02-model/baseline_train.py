#!/usr/bin/env python3
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import os
import joblib
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from langdetect import detect, DetectorFactory
from bs4 import BeautifulSoup
import sys
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.linear_model import SGDClassifier
from collections import Counter
import re
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import wordninja

DetectorFactory.seed = 0
nltk.download('stopwords')
nltk.download('punkt')


def load_data(file_path):
    return pd.read_json(file_path, orient='records', lines=True).set_index('uid', drop=True).sort_index()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a webpage classification model')
    parser.add_argument(
        "-d", "--data_dir", help="Path to the directory containing the data subfolders.", required=True)
    parser.add_argument("-m", "--model_output",
                        help="Path to save the trained model.", required=True)
    return parser.parse_args()


def preprocess(content):
    # Placeholder for the content preprocessing
    return content


# Text-based model
language_translation = {
    'en': 'english',
    'de': 'german',
    'hr': 'hungarian',
    'nl': 'dutch',
    'ru': 'russian',
    'zh-cn': 'chinese',
    'pt': 'portuguese',
    'fr': 'french',
    'id': 'indonesia',
    'es': 'spanish',
    'tr': 'turkish',
    'it': 'italian'
}
stop_words = {}
for key, lang in language_translation.items():
    try:
        stop_words[key] = stopwords.words(lang)
    except:
        pass
stemmers = {}
for key, lang in language_translation.items():
    try:
        stemmers[key] = SnowballStemmer(language=lang)
    except:
        pass


def detect_language(text: str):
    try:
        return detect(text)
    except:
        return None


def preprocess_text(row) -> str:
    lang = row['lang']
    text = row['text']
    text = text.lower()
    tokens = [w for w in word_tokenize(text)]

    # Handle stopwords
    if lang in stop_words:
        tokens = [w for w in tokens if w not in stop_words[lang]]
    if lang in stemmers:
        stemmer = stemmers[lang]
        tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)


def preprocess_text_dataset(dataset: pd.DataFrame):
    dataset['html_soup'] = dataset['html'].apply(
        lambda html: BeautifulSoup(html, 'lxml'))
    dataset['text'] = dataset['html_soup'].apply(
        lambda soup: soup.text)
    dataset['lang'] = dataset['text'].apply(detect_language)
    dataset['processed_text'] = dataset.apply(
        preprocess_text, axis='columns')
    return dataset['processed_text']

# Url based


def extract_domain_and_path(url):
    # Supprimer le protocole s'il est présent
    url = url.replace('www', '')

    # Extraire le nom de domaine et le chemin
    parts = url.split('/')
    domain = parts[0].replace('.', ' ')
    path = ' '.join(parts[1:])

    # Concaténer le nom de domaine et le chemin
    result = f"{domain} {path}"

    return result.strip()


def separate_words(input_str):
    # Utiliser wordninja pour segmenter les mots
    words = wordninja.split(input_str)

    # Joindre les mots séparés avec des espaces
    result = ' '.join(words)

    return result


def top_words(input_str, n=1000):
    input_str = [word for word in word_tokenize(input_str) if len(word) > 1]
    input_str = ' '.join(input_str)
    # Convertir le texte en minuscules et diviser en mots
    words = re.findall(r'\b\w+\b', input_str.lower())

    # Compter les occurrences des mots
    word_counts = Counter(words)

    # Afficher les n mots les plus courants
    top_n_words = word_counts.most_common(1000)

    L = []
    for word, count in top_n_words:
        L.append(word)
    return L


def tokenize_url(url, corpus):
    v = np.zeros(len(corpus) + 3, dtype=int)
    for idx, w in enumerate(corpus):
        n = url.count(w)
        if n > 0:
            v[idx] = n
    v[-1] = 1 if url.count('/') > 2 else 0
    v[-2] = 0 if sum(c.isdigit() for c in url) == 0 else 1
    v[-3] = 1 if url.count('-') > 0 else 0
    return v


def preprocess_url_dataset(dataset: pd.DataFrame, train=True, corpus=None):
    dataset['url_words'] = dataset['url'].apply(
        lambda url: separate_words(extract_domain_and_path(url)))
    if train:
        grouped_df = dataset.groupby('label')['url_words'].apply(
            lambda x: ' '.join(x)).reset_index()
        top_words_result_A = top_words(grouped_df['url_words'][0])
        top_words_result_B = top_words(grouped_df['url_words'][1])
        top_words_result_M = top_words(grouped_df['url_words'][2])
        corpus = list(
            set(top_words_result_A + top_words_result_B + top_words_result_M))

    dataset['url_vector'] = dataset['url'].apply(
        lambda url: tokenize_url(url, corpus))
    return dataset['url_vector'].tolist(), corpus


def main(data_dir, model_output):
    # Load datasets
    train_data = load_data(os.path.join(data_dir, 'train/D1_train.jsonl'))
    train_labels = load_data(os.path.join(
        data_dir, 'train/D1_train-truth.jsonl'))['label']
    val_data = load_data(os.path.join(
        data_dir, 'validation/D1_validation.jsonl'))
    val_labels = load_data(os.path.join(
        data_dir, 'validation/D1_validation-truth.jsonl'))['label']

    train_data = pd.concat([train_data, val_data])
    train_labels = pd.concat([train_labels, val_labels])
    train_data['label'] = train_labels

    # Text based model
    train_features1 = preprocess_text_dataset(train_data)
    pipeline_text = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=15000)),
        ('clf', SGDClassifier(loss='modified_huber'))
    ])
    pipeline_text.fit(train_features1, train_labels)
    train_predictions1 = pipeline_text.predict_proba(train_features1)

    # Url based model
    train_features2, corpus = preprocess_url_dataset(train_data, train=True)
    pipeline_url = SGDClassifier(loss='modified_huber')
    pipeline_url.fit(train_features2, train_labels)
    train_predictions2 = pipeline_url.predict_proba(train_features2)

    # Final model
    train_features_final = np.concatenate(
        [train_predictions1, train_predictions2], axis=1)
    model_final = SGDClassifier()
    model_final.fit(train_features_final, train_labels)

    # Save the trained model
    joblib.dump({
        'text': pipeline_text,
        'url': pipeline_url,
        'url_corpus': corpus,
        'final': model_final
    }, model_output)


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, args.model_output)
