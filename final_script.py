# Filename: final_script.py
# Description: A script to train the model and clasify 1000 discussions
# Date: 
# Authors: Shaya Bhailal

import json
import pandas as pd
import numpy as np
import spacy
import sys
from scipy.sparse import csr_matrix
from empath import Empath
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from scipy.sparse import hstack
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score)


def load_data(path):
    """Load json file and return pandas dataframe."""
    print('Loading ' + path + '...')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('Success')
    return pd.DataFrame(data)


def convert_to_categories_series(series, num):
    """Convert Likert scale to ether 3 or 2 categroies, depending on num."""
    if num == 3:
        return series.map(lambda x: 1 if x in [1, 2] else 2 if x == 3 else 3 if x in [4, 5] else 'unk')
    elif num == 2:
        return series.map(lambda x: 1 if x in [1, 2]  else 3 if x in [3, 4, 5] else 'unk')


def conversion(data, num):
    """Puts conversion all together by going through the labels."""
    data['suspense'] = convert_to_categories_series(data['suspense'], num)
    data['curiosity'] = convert_to_categories_series(data['curiosity'], num)
    data['surprise'] = convert_to_categories_series(data['surprise'], num)
    return data


def create_model():
    """Creates the ensemble model, choice from SVM, LR, NB and RF."""
    svm_cls = SVC(kernel='linear', random_state=42, C=10, probability=True)
    lr_cls = LogisticRegression(max_iter=1000, random_state=42)
    nb_cls = MultinomialNB()
    rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)

    voting_cls = VotingClassifier(estimators=[ # Adjust to include other models
        # ('nb_cls', nb_cls),
        ('svm_cls', svm_cls),
        # ('lr_cls', lr_cls),
        ('rf_cls', rf_cls)
    ], voting='soft')
    
    # Handling multi ouputs
    multi_output_model = MultiOutputClassifier(voting_cls)
    return multi_output_model



def extract_spacy_features(texts):
    """Extracts n_sentences, n_entities, num_questions, lexical_diversity, avg_token_len."""
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe("sentencizer")
    n_entities = []
    n_sentences = []
    avg_token_len = []
    lexical_diversity_list = []
    num_questions_list = []
    for doc in nlp.pipe(texts, disable=["parser", "tagger"]):
        num_unique_tokens = len(set([token.text.lower() for token in doc if token.is_alpha]))
        lexical_diversity = num_unique_tokens / len(doc) if len(doc) else 0
        num_questions = sum(1 for sent in doc.sents if '?' in sent.text)
        n_entities.append(len(doc.ents))
        n_sentences.append(len(list(doc.sents)))
        token_lens = [len(token) for token in doc if not token.is_punct and not token.is_space]
        avg_token_len.append(sum(token_lens) / len(token_lens) if token_lens else 0)
        num_questions_list.append(num_questions)
        lexical_diversity_list.append(lexical_diversity)
    features = np.array([n_entities, n_sentences, avg_token_len, lexical_diversity_list, num_questions_list]).T
    return csr_matrix(features)


def extract_empath_features(texts, categories=None):
    """Extracts emotion per text using Empath libary."""
    lexicon = Empath()
    if categories is None:
        categories = list(lexicon.cats)
    
    feature_list = []
    for text in texts:
        features = lexicon.analyze(text, categories=categories, normalize=True)
        if features is None:
            feature_vector = [0.0] * len(categories)
        else:
            feature_vector = [features[cat] for cat in categories]
        feature_list.append(feature_vector)
    
    return csr_matrix(feature_list)


def baseline(test):
    """Baseline classifier, DummyClassifier using most_frequent."""
    labels = ['suspense', 'curiosity', 'surprise', 'story_class']
    for label in labels:
        y_test = test[label].values
        dummy_cls = DummyClassifier(strategy="most_frequent")
        dummy_X = np.zeros((len(y_test), 1))
        dummy_cls.fit(dummy_X, y_test)
        print('Dummy prediction class', label)
        pred = dummy_cls.predict(dummy_X)
        print('Accuracy', accuracy_score(y_test, pred))
        print('Precision', precision_score(y_test, pred, average='weighted', zero_division=0))
        print('Recall', recall_score(y_test, pred, average='weighted'))
        print('F1 macro:',f1_score(y_test, pred, average='macro'))
        print('F1 weighted:', f1_score(y_test, pred, average='weighted'))
    

def show_misclassified(y_test, y_pred, X_test):
    for idx, (true, pred) in enumerate(zip(y_test, y_pred)):
        if (true != pred).any():
            print(f'Index: {idx}')
            print(f'Text: {X_test[idx]}')
            print(f'True label: {true}')
            print(f'Predicted label: {pred}')
            print('-'*80)


def train_test(model, train, test):
    """"Train and test on annotated data."""
    print('Prepping data...')
    # Convert story to numeric
    train['story_class'] = train['story_class'].map({'Story': 1, 'Not Story': 0})
    test['story_class'] = test['story_class'].map({'Story': 1, 'Not Story': 0})
    X_train = train['body'].tolist()
    y_train = train[['suspense', 'curiosity', 'surprise', 'story_class']].values
    X_test = test['body'].tolist()
    y_test = test[['suspense', 'curiosity', 'surprise', 'story_class']].values
    # TF-IDF
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    # Empath
    train_empath_features = extract_empath_features(X_train)
    test_empath_features = extract_empath_features(X_test)
    # Lexical features
    train_spacy_features = extract_spacy_features(X_train)
    test_spacy_features = extract_spacy_features(X_test)
    # Combine with hstack
    X_train_combined = hstack([X_train_tfidf, train_spacy_features, train_empath_features])
    X_test_combined = hstack([X_test_tfidf, test_spacy_features, test_empath_features])
    print('Now training...')
    model.fit(X_train_combined, y_train)
    print('Predicting...')
    y_pred = model.predict(X_test_combined)
    evaluate(y_test, y_pred)
    # Uncomment to see misclassifications by model v v v
    # show_misclassified(y_test, y_pred, X_test)
    return model, tfidf

  
def evaluate(labels, pred):
    """Prints evaluation metrics for the model."""
    print('Readers perception')
    list =[]
    for i, label_name in enumerate(['suspense', 'curiosity', 'surprise', 'story_class']):
        print(f"\nMetrics for {label_name}:")
        accuracy = accuracy_score(labels[:, i], pred[:, i])
        print(f"Accuracy: {accuracy:.3f}")

        precision = precision_score(labels[:, i], pred[:, i], average="weighted", zero_division=0)
        print(f"Precision: {precision:.3f}")

        recall = recall_score(labels[:, i], pred[:, i], average="weighted")
        print(f"Recall: {recall:.3f}")

        f_score = f1_score(labels[:, i], pred[:, i], average="weighted")
        list.append(f_score)
        print(f"F1-score(weighted): {f_score:.3f}")
        
        f_score2 = f1_score(labels[:, i], pred[:, i], average="macro")
        list.append(f_score2)
        print(f"F1-score(macro): {f_score2:.3f}")
        
    print('Average Fscore:', sum(list) / len(list))


def process(data):
    """Prepare sample for classifying."""
    # Extracting relevant columns
    df = data[['name', 'author', 'body', 'num_comments', 'delta', 'selftext']].copy()
    df.loc[df['body'] == '', 'body'] = data.loc[df['body'] == '', 'selftext']
    # Filter comments
    df = df[df['author'] != 'DeltaBot']
    df = df[~df['body'].isin(['[deleted]', '[redacted]', '[removed]'])]
    # Delete selftext, everyting needed is now in 'body'
    df.drop(columns=['selftext'], inplace=True)
    df.to_json('sample_cls1.json', orient='records', lines=True)
    return df


def classify(model, data, tfidf):
    """Classifies sample."""
    texts = data['body'].tolist()
    print('Extracting features...')
    texts_tfidf = tfidf.transform(texts)
    texts_empath_features = extract_empath_features(texts)
    texts_spacy_features = extract_spacy_features(texts)
    texts_combined = hstack([texts_tfidf, texts_spacy_features, texts_empath_features])
    print('Classifying sample...')
    predictions = model.predict(texts_combined)
    label_names = ['suspense', 'curiosity', 'surprise', 'story']
    print('Labelling..')
    for i, label in enumerate(label_names):
        data[f'predicted_{label}'] = predictions[:, i]
    data.to_json('sample_clf.json', orient='records', lines=True)
    return


def main(argv):
    train_file = 'golden-standard-train.json'
    test_file = 'golden-standard-test.json'
    data_file = 'threads1000_format.json'
    if len(sys.argv) < 2:
        print('Please specify how many labels you want the model to train on. Options are 2, 3 and 5')
        exit(-1)
    mapping = int(argv[1])
    train_data = load_data(train_file)
    test_data = load_data(test_file)
    # Model training and evaluation
    if mapping != 5:
        train_data = conversion(train_data, mapping)
        test_data = conversion(test_data, mapping)
    model = create_model()
    # trained_model, tfidf = train_test(model, train_data, test_data)
    baseline(test_data)
    # Automatic classification
    discussions1k = load_data(data_file)
    data = process(discussions1k)
    classify(model, data, tfidf)
    


if __name__ == "__main__":
    main(sys.argv)
