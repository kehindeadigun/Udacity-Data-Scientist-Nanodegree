import sys
import nltk 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sqlalchemy import create_engine

import pandas as pd
import numpy as np

import re

import joblib

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM messages',engine)
    X = df[['message']]
    Y = df.drop(columns=['id','message','original','genre'])
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    text = re.sub(r'[^a-zA-Z0-9]',' ',text)
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]


def build_model():
    
    vectorizer = CountVectorizer(tokenizer=tokenize, stop_words=stopwords.words("english"))
    transformer = TfidfTransformer()
    classifier = MultiOutputClassifier(RandomForestClassifier())
    
    pipeline = Pipeline([('vect', vectorizer),
                         ('tfidf', transformer),
                         ('clf', classifier)
                        ])
    
    parameters = {'vect__ngram_range':[(1, 1),(1,2)],
                  'clf__estimator__n_estimators':list(range(1,30,5))
                 }
    model = GridSearchCV(pipeline, parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    for i in range(y_pred.shape[1]):
        class_names = ['-'.join([str(category_names[i]), str(j)]) for j in range(3)]
        print('Column ',i+1,' : ',category_names[i])
        print(classification_report(y_pred[:,i], y_test.values[:,i], target_names=class_names))
        #print lines to visually separate categories in the output print
        print('-'*65)


def save_model(model, model_filepath):

    best_model = model.best_estimator_
    filename = 'classifier.pkl'

    with open(model_filepath, 'wb') as file:
        joblib.dump(best_model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()