import sys
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sqlalchemy import create_engine

import pandas as pd
import numpy as np

import re
import joblib
from os import path

def load_data(database_filepath):
    """Loads data from a defined filepath
    
    Args:
    df pandas.Dataframe: A pandas dataframe to save
    database_filename str: A filename for the database name
    
    Returns:
    X numpy array: numpy Array: contains the message data values.
    Y numpy array: numpy Array: contains the message target values.
    category_names numpy Array:  contains the category name of each target value.
    """
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM messages',engine)
    target = df.drop(columns = ['id','message','original','genre'])
    
    X = df['message'].values
    target = df.drop(columns =['id','message','original','genre']).values
    return X, Y, category_names


def tokenize(text):
    """A function to tokenize a text string. Strips the text of punctuation, stopwords and lemmatizes each word. 
    
    Args:
    text str: A string of text
    
    Returns: 
    A list of tokens or tokenized words.
    """
    text = re.sub(r'[^a-zA-Z0-9]',' ',text)
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]


def build_model():
    """
    Builds a multioutput text classifcation model. 
    
    Returns:
    A grid search multiclassification model with a randomforest estimator as base
    """
    
    vectorizer = CountVectorizer(tokenizer=tokenize, stop_words=stopwords.words("english"))
    transformer = TfidfTransformer()
    classifier = MultiOutputClassifier(RandomForestClassifier())
    
    pipeline = Pipeline([('vect', vectorizer),
                         ('tfidf', transformer),
                         ('clf', classifier)
                        ])
    
    parameters = { 'vect__ngram_range':[(1, 1),(1,2)], 'clf__estimator__n_estimators':list(range(1,30,5)) }
    
    model = GridSearchCV(pipeline, parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the model, printing out a classification report 
    
    Args:
    model: A scikit learn estimator
    X_test numpy array, list or dataframe: Contains test values from the data
    Y_test numpy array, list or dataframe: Contains test values from the target
    category_names numpy array or list: Contains the category name of each target value.
    """
    
    for i in range(y_pred.shape[1]):
        class_names = ['-'.join([str(category_names[i]), str(j)]) for j in range(3)]
        print('Column ',i+1,' : ',category_names[i])
        print(classification_report(y_pred[:,i], y_test[:,i], target_names=class_names))
        #print lines to visually separate categories in the output print
        print('-'*65)


def save_model(model, model_filepath=''):
    """Saves an ML model to the the filepath. Extracts and saves the best estimator if presented with a grid search model. Saves file name is 'classifier.pkl'.
    
    Args:
    model A model to save
    model_filepath str: File path to save the model to. (If filepath does not exist, will save to the model to the same folder as the train_classifier script).
    """
    try:
        best_model = model.best_estimator_
    except:
        best_model = model
        
    if model_filepath != '':
    if not path.exists(model_filepath):
        print('WARNING: Save Path does not exists, Saving to default folder')

    model_filepath = model_filepath+'classifier.pkl'

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
        save_model(model, '')

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()